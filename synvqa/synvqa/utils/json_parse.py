"""Tolerant JSON extraction from LLM text outputs."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_strict_json(text: str) -> dict | list:
    """Try to parse JSON out of an LLM response.

    Strategy:
      1. Strip whitespace, try json.loads directly.
      2. Extract fenced ```json ... ``` block if present.
      3. Fall back to the largest balanced { ... } or [ ... ] span.
    Raises ValueError on failure.
    """
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    m = _FENCE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        end = s.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from text: {text[:200]!r}")
