import logging
import sys
from pathlib import Path

_INIT = False


def get_logger(name: str = "synvqa", level: str = "INFO", file: str | None = None) -> logging.Logger:
    global _INIT
    logger = logging.getLogger(name)
    if _INIT:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _INIT = True
    return logger
