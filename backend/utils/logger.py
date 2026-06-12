"""
Logging configuration
"""

import logging
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler

from backend.app.config import settings


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logger with console + optional file handler
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers (important in FastAPI reload)
    if logger.handlers:
        return logger

    # Set log level
    log_level = (level or settings.LOG_LEVEL or "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    logger.propagate = False  # avoid double logging via root logger

    # -------------------------
    # Console Handler
    # -------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    # -------------------------
    # Optional File Handler
    # -------------------------
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )

        file_handler.setLevel(logging.DEBUG)

        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger