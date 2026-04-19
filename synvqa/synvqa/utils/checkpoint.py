"""Per-stage checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from .io import read_jsonl, write_jsonl


class CheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: int) -> Path:
        return self.output_dir / f"stage_{stage}_checkpoint.jsonl"

    def exists(self, stage: int) -> bool:
        p = self.path(stage)
        return p.exists() and p.stat().st_size > 0

    def write(self, stage: int, records: Iterable[dict]) -> int:
        return write_jsonl(self.path(stage), records)

    def read(self, stage: int) -> Iterator[dict]:
        p = self.path(stage)
        if not p.exists():
            return iter(())
        return read_jsonl(p)

    def load_if_resuming(self, stage: int, resume: bool) -> list[dict] | None:
        if resume and self.exists(stage):
            return list(self.read(stage))
        return None
