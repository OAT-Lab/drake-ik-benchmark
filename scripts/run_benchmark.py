"""
The thin CLI conductor. 
- Contains no benchmark logic: 
    - load config
    - build plant
    - generate scenarios
    - loop over each (scenario + solver) pair through the single runner path
    - collect results
    - save
Orchestration only — all real logic lives in the src package.
"""


from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from ik_benchmark.logging import (
    configure_logging,
    make_timestamped_run_dir,
    resolve_log_file,
)


logger = logging.getLogger("ik_benchmark.cli.run_benchmark")


def _load_logging_config(config_path):
    with Path(config_path).open() as f:
        cfg = yaml.safe_load(f) or {}
    logging_cfg = cfg.get("logging", {}) or {}
    return {
        "level": logging_cfg.get("level", "INFO"),
        "log_file": logging_cfg.get("file"),
        "mode": logging_cfg.get("mode", "w"),
        "quiet": bool(logging_cfg.get("quiet", False)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Drake IK solver benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Override logging.level from the config.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Override logging.file from the config.",
    )
    parser.add_argument(
        "--log-mode",
        default=None,
        choices=["w", "a"],
        help="Override logging.mode from the config.",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override logging.quiet from the config.",
    )
    args = parser.parse_args()

    logging_cfg = _load_logging_config(args.config)
    log_level = args.log_level if args.log_level is not None else logging_cfg["level"]
    log_file_config = (
        args.log_file if args.log_file is not None else logging_cfg["log_file"]
    )
    log_mode = args.log_mode if args.log_mode is not None else logging_cfg["mode"]
    quiet = args.quiet if args.quiet is not None else logging_cfg["quiet"]
    run_dir = make_timestamped_run_dir(args.out)
    log_file = resolve_log_file(log_file_config, run_dir)
    configure_logging(level=log_level, log_file=log_file, mode=log_mode, quiet=quiet)
    logger.info(
        "starting benchmark config=%s out=%s run_dir=%s",
        args.config,
        args.out,
        run_dir,
    )

    # cfg = load_config(args.config)
    # plant, plant_context, gripper_frame = build_plant()
    # scenarios = generate_scenarios(plant, cfg.n, cfg.base_seed, cfg.method_mix, cfg.task_space)
    # results = []
    # for scenario in scenarios:
    #     prog, q = build_ik_prog(plant, plant_context, gripper_frame, scenario, cfg.orientation_tol)
    #     solvers = get_candidate_solvers(prog)
    #     options = build_shared_options(cfg.tolerances)
    #     for solver in solvers:
    #         results.append(run_one(prog, q, solver, scenario.q0, options,
    #                                cfg.n_repeats, cfg.n_warmup, cfg.feasibility_threshold))
    # df = aggregate(results)
    # save(df, args.out, metadata=collect_metadata(cfg))
    raise NotImplementedError


if __name__ == "__main__":
    main()
