"""Embedding-based near-duplicate detection for Stage 1."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..models.embeddings import EmbeddingClient


class EmbeddingDeduper:
    def __init__(self, embedder: EmbeddingClient, threshold: float = 0.85):
        self.embedder = embedder
        self.threshold = threshold
        self._vecs: list[np.ndarray] = []
        self._texts: list[str] = []

    def seed(self, texts: Iterable[str]) -> None:
        texts = list(texts)
        if not texts:
            return
        vecs = self.embedder.embed(texts)
        self._texts.extend(texts)
        self._vecs.extend(vecs)

    def _matrix(self) -> np.ndarray:
        if not self._vecs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(self._vecs)

    def filter_and_add(self, candidates: list[str]) -> tuple[list[str], list[str]]:
        """Returns (kept, rejected). Kept items are embedded and appended to the pool."""
        if not candidates:
            return [], []
        cand_vecs = self.embedder.embed(candidates)
        kept, rejected = [], []
        mat = self._matrix()
        for text, vec in zip(candidates, cand_vecs):
            if mat.shape[0] == 0:
                max_sim = 0.0
            else:
                sims = mat @ vec
                max_sim = float(sims.max())
            # also check against already-kept in this batch
            for k_vec in (self._vecs[-len(kept):] if kept else []):
                s = float(np.dot(k_vec, vec))
                if s > max_sim:
                    max_sim = s
            if max_sim < self.threshold:
                kept.append(text)
                self._vecs.append(vec)
                self._texts.append(text)
                mat = self._matrix()
            else:
                rejected.append(text)
        return kept, rejected

    @property
    def size(self) -> int:
        return len(self._texts)

    @property
    def vectors(self) -> np.ndarray:
        return self._matrix()
