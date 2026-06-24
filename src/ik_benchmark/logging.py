"""Logging config for benchmark entry points."""

from __future__ import annotations

import logging
from datetime import datetime
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
    """Configure logging for the ik_benchmark package.

    The package logger is isolated from the root logger so benchmark logs do not
    change Drake or dependency logging behavior.
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


def make_timestamped_run_dir(
    out_dir: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Create and return a unique timestamped benchmark run directory."""
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    runs_dir = Path(out_dir) / "runs"
    run_dir = runs_dir / timestamp
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_log_file(log_file: str | Path | None, base_dir: str | Path) -> Path | None:
    """Resolve a configured log file path against an output directory."""
    if log_file is None:
        return None

    path = Path(log_file)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    levels = logging.getLevelNamesMapping()
    normalized = level.upper()
    if normalized in levels:
        return levels[normalized]

    valid = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    raise ValueError(f"log level must be one of {valid}, got {level!r}")
