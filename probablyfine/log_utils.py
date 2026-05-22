"""Shared logging factory for probablyfine modules."""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_DIR = Path.home() / ".probablyfine"


def get_module_logger(name: str, log_file: str) -> logging.Logger:
    """Create or retrieve a logger that writes to ~/.probablyfine/<log_file>.

    Idempotent: repeated calls with the same name return the existing logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_LOG_DIR / log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
