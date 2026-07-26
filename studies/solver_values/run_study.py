"""for running the solver_values study

Per (seed, orientation_tol): build the prog, generate n_per_cell scenarios,
solve every available solver from the same initial guess (scenario.q0)

Each (scenario, solver) is two passes: one recorded solve with the print file
on, then n_warmup + n_repeats timed solves with printing off. SNOPT writes a
line per major iteration; timing a printing solve measures the filesystem.

Outputs (under results/<timestamp>/, with results/latest symlinked to it):
  solves.csv       one row per (scenario, solver)
  joints.csv       one row per (solve, joint)
  constraints.csv  one row per (solve, constraint row)
  timings.csv      one row per (solve, timed repeat)
  config.yaml      the config this run used
  provenance.txt   which pydrake ran it
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydrake
import yaml

from ik_benchmark.log_config import configure_logging
from ik_benchmark.plant import build_plant
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.results import solve_row
from ik_benchmark.runner import run_one
from ik_benchmark.scenarios import generate_scenarios
from ik_benchmark.solvers import get_candidate_solvers

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
RESULTS_ROOT = Path(__file__).parent / "results"
RESULTS_DIR = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _init_results_dir():
    """Timestamped dir + a latest symlink so downstream paths stay fixed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    latest = RESULTS_ROOT / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(RESULTS_DIR.name)

    try:
        from importlib.metadata import version
        drake_version = version("drake")
    except Exception:
        drake_version = "source build (no dist metadata)"

    (RESULTS_DIR / "provenance.txt").write_text(
        f"pydrake: {pydrake.__file__}\n"
        f"version: {drake_version}\n"
        f"run: {datetime.now().isoformat()}\n"
    )


def main():
    cfg = _load_config(CONFIG_PATH)
    _init_results_dir()

    log = cfg["logging"]
    configure_logging(
        level=log["level"],
        log_file=RESULTS_DIR / log["file"],
        mode=log["mode"],
        quiet=log["quiet"],
    )

    feas_threshold = cfg["compare"]["feasibility_threshold"]
    box = cfg["box"]
    time_step = cfg["time_step"]
    n_warmup = cfg["timing"]["n_warmup"]
    n_repeats = cfg["timing"]["n_repeats"]

    solve_rows, joint_rows, constraint_rows, timing_rows = [], [], [], []
    scenario_counter = 0

    logger.info("study start: %d seeds x %d tols x %d scenarios (%d+%d+1 solves per row)",
                len(cfg["seeds"]), len(cfg["orientation_tols"]), cfg["n_per_cell"],
                n_warmup, n_repeats)

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

                # build the prog ONCE per scenario; every solver sees this exact prog + q0
                prog, _ = build_ik_prog(
                    plant, plant_context, gripper_frame, scenario,
                    orientation_tol=orientation_tol,
                    use_minimum_distance=cfg["minimum_distance"]["use_minimum_distance"],
                    min_distance_bound=cfg["minimum_distance"]["min_distance_bound"],
                    influence_distance_offset=cfg["minimum_distance"]["influence_distance_offset"],
                )
                solvers = get_candidate_solvers(prog)
                for solver in solvers:
                    solver_name = solver.solver_id().name()
                    solve_id = f"{seed}_{orientation_tol}_{scenario_id}_{solver_name}"

                    tables = run_one(
                        prog,
                        solver,
                        scenario,
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

    pd.DataFrame(solve_rows).to_csv(RESULTS_DIR / "solves.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(RESULTS_DIR / "joints.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(RESULTS_DIR / "constraints.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(RESULTS_DIR / "timings.csv", index=False)
    logger.info("wrote %d solves, %d joint rows, %d constraint rows, %d timing rows to %s",
                len(solve_rows), len(joint_rows), len(constraint_rows),
                len(timing_rows), RESULTS_DIR)


if __name__ == "__main__":
    main()
