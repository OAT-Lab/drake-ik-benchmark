"""
Run setup + execution for the Drake IK benchmark.

Produces a timestamped run directory with the four CSV tables, the resolved config, and provenance.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydrake
import yaml

from ik_benchmark.log_config import configure_logging
from ik_benchmark.paths import make_timestamped_run_dir, resolve_log_file
from ik_benchmark.plant import build_plant
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.results import solve_row
from ik_benchmark.runner import run_one
from ik_benchmark.scenarios import generate_scenarios
from ik_benchmark.solvers import get_candidate_solvers

CONFIG_PATH = Path("configs/default.yaml")
OUT_DIR = Path("results")

logger = logging.getLogger("ik_benchmark.scripts.run_benchmark")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
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
        config_path, out_dir, run_dir, log_file,
    )
    return cfg, run_dir


def _write_provenance(run_dir, cfg):
    import platform, socket
    try:
        try:
            from importlib.metadata import version
            drake_version = version("drake")
        except Exception:
            drake_version = "unknown"
        (run_dir / "provenance.txt").write_text(
            f"host: {socket.gethostname()}\n"
            f"platform: {platform.platform()}\n"
            f"machine: {platform.machine()}\n"
            f"python: {platform.python_version()}\n"
            f"pydrake: {pydrake.__file__}\n"
            f"drake_version: {drake_version}\n"
            f"run: {datetime.now().isoformat()}\n"
        )
    except Exception as e:
        logger.warning("provenance write failed (continuing): %s", e)

    try:
        with (run_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as e:
        logger.warning("config dump failed (continuing): %s", e)


def run_benchmark(cfg: dict, run_dir: Path) -> None:
    box = cfg["box"]
    time_step = cfg["time_step"]
    feas_threshold = cfg["compare"]["feasibility_threshold"]
    n_warmup = cfg["timing"]["n_warmup"]
    n_repeats = cfg["timing"]["n_repeats"]
    md = cfg["minimum_distance"]

    solve_rows, joint_rows, constraint_rows, timing_rows = [], [], [], []
    scenario_counter = 0

    logger.info(
        "study start: %d seeds x %d tols x %d scenarios (%d warmup + %d repeats + 1 record per solve)",
        len(cfg["seeds"]), len(cfg["orientation_tols"]), cfg["n_per_cell"],
        n_warmup, n_repeats,
    )

    for seed in cfg["seeds"]:
        for orientation_tol in cfg["orientation_tols"]:
            plant, plant_context, gripper_frame, q0 = build_plant(time_step, box)
            scenarios = generate_scenarios(
                plant, plant_context, gripper_frame, q0,
                n=cfg["n_per_cell"], base_seed=seed,
                method_mix=cfg["method_mix"],
                task_space_params=cfg["task_space"],
            )

            for scenario in scenarios:
                scenario_id = scenario_counter
                scenario_counter += 1

                # one prog per scenario; every solver sees this exact prog + q0
                prog, _ = build_ik_prog(
                    plant, plant_context, gripper_frame, scenario,
                    orientation_tol=orientation_tol,
                    use_minimum_distance=md["use_minimum_distance"],
                    min_distance_bound=md["min_distance_bound"],
                    influence_distance_offset=md["influence_distance_offset"],
                )

                for solver in get_candidate_solvers(prog):
                    solver_name = solver.solver_id().name()
                    solve_id = f"{seed}_{orientation_tol}_{scenario_id}_{solver_name}"

                    tables = run_one(
                        prog, solver, scenario,
                        solve_id=solve_id,
                        scenario_id=scenario_id,
                        seed=seed,
                        orientation_tol=orientation_tol,
                        cfg=cfg,
                        feasibility_threshold=feas_threshold,
                        n_warmup=n_warmup,
                        n_repeats=n_repeats,
                    )
                    solve_rows.append(solve_row(tables.solve))
                    joint_rows.extend(asdict(j) for j in tables.joints)
                    constraint_rows.extend(asdict(c) for c in tables.constraints)
                    timing_rows.extend(asdict(t) for t in tables.timings)

            logger.info("done seed=%s tol=%s (scenarios so far: %d)",
                        seed, orientation_tol, scenario_counter)

    pd.DataFrame(solve_rows).to_csv(run_dir / "solves.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(run_dir / "joints.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(run_dir / "constraints.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(run_dir / "timings.csv", index=False)
    logger.info(
        "wrote %d solves, %d joint rows, %d constraint rows, %d timing rows to %s",
        len(solve_rows), len(joint_rows), len(constraint_rows),
        len(timing_rows), run_dir,
    )


def main() -> None:
    cfg, run_dir = setup_benchmark_run()
    _write_provenance(run_dir, cfg)
    run_benchmark(cfg, run_dir)


if __name__ == "__main__":
    main()