"""
utils/logger.py
───────────────
Loguru-based logger configured once and imported across the project.
"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import get_settings


def setup_logger() -> None:
    settings = get_settings()
    log_file: Path = settings.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # remove default stderr handler

    # Console handler — colourful & human-readable
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — JSON for structured log analysis
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        serialize=True,
    )


setup_logger()

__all__ = ["logger"]
