"""
Run setup for the Drake IK benchmark.

This file intentionally stays simple: edit the constants below, run the file,
and the benchmark gets a timestamped run directory plus configured package
logging. Benchmark execution will be added here later.
"""


from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ik_benchmark.log_config import configure_logging
from ik_benchmark.paths import make_timestamped_run_dir, resolve_log_file


CONFIG_PATH = Path("configs/default.yaml")
OUT_DIR = Path("results")

logger = logging.getLogger("ik_benchmark.scripts.run_benchmark")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load the benchmark YAML config."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f) or {}


def _logging_config(cfg: dict) -> dict:
    logging_cfg = cfg.get("logging", {}) or {}
    return {
        "level": logging_cfg.get("level", "INFO"),
        "log_file": logging_cfg.get("file"),
        "mode": logging_cfg.get("mode", "w"),
        "quiet": bool(logging_cfg.get("quiet", False)),
    }


def setup_benchmark_run(
    config_path: str | Path = CONFIG_PATH,
    out_dir: str | Path = OUT_DIR,
) -> tuple[dict, Path]:
    """Create the run directory and configure benchmark logging."""
    cfg = load_config(config_path)
    logging_cfg = _logging_config(cfg)

    run_dir = make_timestamped_run_dir(out_dir)
    log_file = resolve_log_file(logging_cfg["log_file"], run_dir)
    configure_logging(
        level=logging_cfg["level"],
        log_file=log_file,
        mode=logging_cfg["mode"],
        quiet=logging_cfg["quiet"],
    )
    logger.info(
        "benchmark run configured config=%s out=%s run_dir=%s log_file=%s",
        config_path,
        out_dir,
        run_dir,
        log_file,
    )
    return cfg, run_dir


def main() -> None:
    setup_benchmark_run()
    logger.info("benchmark execution not implemented yet")


if __name__ == "__main__":
    main()
