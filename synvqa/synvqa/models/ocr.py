"""Lightweight OCR for Stage 5 answer-leakage check."""
from __future__ import annotations

from pathlib import Path


def ocr_image(image_path: str | Path) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:
        raise RuntimeError("pytesseract / Pillow not installed") from e
    img = Image.open(str(image_path))
    return pytesseract.image_to_string(img)
