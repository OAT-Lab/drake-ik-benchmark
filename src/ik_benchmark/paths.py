"""
filesystem path helpers for benchmark runs
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_timestamped_run_dir(
    out_dir: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """
    create and return a new timestamped run directory

    Produces <out_dir>/runs/<YYYY-MM-DD_HHMMSS>/
    
    If exists, suffix added
    """
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
    """
    resolve a configured log-file path against an output directory

    Absolute paths unchanged
    Relative paths put under base_dir
    None returns None
    """
    if log_file is None:
        return None

    path = Path(log_file)
    if path.is_absolute():
        return path
    return Path(base_dir) / path