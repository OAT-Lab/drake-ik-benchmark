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
    update_latest: bool = True,
) -> Path:
    """
    create and return a new timestamped run directory

    Produces <out_dir>/runs/<YYYY-MM-DD_HHMMSS>/
    If exists, suffix added.
    Repoints <out_dir>/runs/latest at the new dir (relative symlink).
    """
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    runs_dir = Path(out_dir) / "runs"
    run_dir = runs_dir / timestamp
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)

    if update_latest:
        _update_latest_symlink(runs_dir, run_dir)

    return run_dir


def _update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    """Point <runs_dir>/latest at run_dir via a relative symlink."""
    latest = runs_dir / "latest"
    # Only remove if it's a symlink (or a broken one); never delete a real dir.
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink():
            latest.unlink()
        else:
            # a real file/dir named 'latest' — leave it, don't clobber user data
            return
    try:
        latest.symlink_to(run_dir.name)  # relative: 'latest' -> '<timestamp>'
    except OSError:
        # symlinks unsupported (e.g. some Windows setups) — skip silently
        pass

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