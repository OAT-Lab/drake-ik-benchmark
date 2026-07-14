"""
logging config for the ik_benchmark package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TextIO


PACKAGE_LOGGER_NAME = "ik_benchmark"
_HANDLER_MARKER = "_ik_benchmark_configured"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    mode: str = "w",
    quiet: bool = False,
    stream: TextIO | None = None,
) -> logging.Logger:
    """
    configure benchmark logger + return

    Args:
        level: minimum level, name or int 
        log_file: also write logs to this file
        mode: w overwrites, a appends
        quiet: if true, don't add console handler (file logs still written)
        stream: console stream for the console handler (default stderr)

    Returns:
        ik_benchmark logger
    """
    if mode not in {"w", "a"}:
        raise ValueError(f"log mode must be 'w' or 'a', got {mode!r}")

    resolved_level = _resolve_level(level)

    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(resolved_level)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)

    if not quiet:
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _HANDLER_MARKER, True)
        logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=mode)
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)

    return logger


def _resolve_level(level: str | int) -> int:
    """turn a level name/int into a logging level int, or raise"""
    if isinstance(level, int):
        return level

    levels = logging.getLevelNamesMapping()
    normalized = level.upper()
    if normalized in levels:
        return levels[normalized]

    valid = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    raise ValueError(f"log level must be one of {valid}, got {level!r}")