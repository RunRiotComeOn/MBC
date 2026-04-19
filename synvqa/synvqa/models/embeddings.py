"""Text embedding client (OpenAI-compatible)."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import numpy as np

from ..utils.io import load_env_file


class EmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self._client = None
        self._local = model.lower().startswith("local-")

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self._local:
            return
        load_env_file()
        from openai import OpenAI  # type: ignore
        self._client = OpenAI()

    def embed(self, texts: Iterable[str]) -> list[np.ndarray]:
        if self._local:
            return [self._local_embed(text) for text in texts]
        self._ensure_client()
        texts = list(texts)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            for d in resp.data:
                v = np.asarray(d.embedding, dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                out.append(v)
        return out

    def _local_embed(self, text: str, dim: int = 256) -> np.ndarray:
        """Deterministic no-key embedding for smoke tests and dedup checks."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float(np.dot(va, vb))
