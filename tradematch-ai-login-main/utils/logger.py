"""
Logging configuration using Loguru.
"""

from loguru import logger
import sys

# Remove default handler and add a custom one
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
           "<level>{message}</level>",
    level="DEBUG",
)

# Optionally log to a file
logger.add(
    "logs/tradematch.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
)
