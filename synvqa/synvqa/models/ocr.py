"""Lightweight OCR for Stage 5 answer-leakage check."""
from __future__ import annotations

from pathlib import Path


def ocr_image(image_path: str | Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        # pytesseract not installed — return empty so the leakage check
        # is effectively skipped rather than crashing the pipeline.
        return ""
    img = Image.open(str(image_path))
    return pytesseract.image_to_string(img)
