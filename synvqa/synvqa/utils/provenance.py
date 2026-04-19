"""Per-stage provenance records."""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


class ProvenanceBuilder:
    def __init__(self, stage: int):
        self.stage = stage
        self._t0 = time.time()
        self._record: dict[str, Any] = {
            "stage": stage,
            "started_at": self._t0,
            "events": [],
        }

    def log(self, event: str, **fields: Any) -> None:
        self._record["events"].append({"t": time.time() - self._t0, "event": event, **fields})

    def model_call(
        self,
        *,
        role: str,
        model: str,
        prompt_text: str | None = None,
        api_call_id: str | None = None,
        extra: dict | None = None,
    ) -> str:
        call_id = api_call_id or str(uuid.uuid4())
        entry: dict[str, Any] = {
            "t": time.time() - self._t0,
            "event": "model_call",
            "role": role,
            "model": model,
            "api_call_id": call_id,
        }
        if prompt_text is not None:
            entry["prompt_hash"] = prompt_hash(prompt_text)
        if extra:
            entry["extra"] = extra
        self._record["events"].append(entry)
        return call_id

    def finalize(self, **summary: Any) -> dict[str, Any]:
        self._record["finished_at"] = time.time()
        self._record["wall_s"] = self._record["finished_at"] - self._t0
        self._record.update(summary)
        return self._record
