"""JSONL / YAML / prompt-loading helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from string import Template
from typing import Any, Iterable, Iterator

import yaml

_DOTENV_LOADED = False


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ.

    This is intentionally lightweight and avoids adding a new dependency.
    Existing environment variables win over values from the file.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_path = Path(path)
    if not env_path.exists():
        for parent in [Path.cwd(), *Path.cwd().parents]:
            candidate = parent / path
            if candidate.exists():
                env_path = candidate
                break
        else:
            return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: Any) -> str:
    """Substitute {var} placeholders by literal string replacement.

    Avoids str.format's format-spec parsing, which collides with the JSON
    braces used inside prompt templates. Unreferenced {var} survives verbatim.
    """
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value))
    return out
