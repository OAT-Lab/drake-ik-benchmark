"""for running the solver_values study

Per (seed, orientation_tol): build the prog, generate n_per_cell scenarios,
solve every available solver from the same initial guess (scenario.q0)

Outputs (under results_dir):
  solves.csv       one row per (scenario, solver)
  joints.csv       one row per (solve, joint)
  constraints.csv  one row per (solve, constraint row)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from pydrake.solvers import (
    GetAvailableSolvers,
    GetProgramType,
    MakeSolver,
    NloptSolver,
    SolverOptions,
    CommonSolverOption,
)

from ik_benchmark.log_config import configure_logging
from ik_benchmark.plant import build_plant
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.scenarios import generate_scenarios

from record import build_solve_tables
from solver_opt import native_optimality_options, parse_native_optimality

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _solver_options(solver_id, solver_name, cfg):
    """Pinned shared tolerances + NLopt caps."""
    opts = SolverOptions()
    tol = cfg["tolerances"]     # from config.yaml
    feas, optm, iters = tol["feasibility_tol"], tol["optimality_tol"], tol["major_iter_limit"]

    if solver_name == "SNOPT":
        opts.SetOption(solver_id, "Major feasibility tolerance", feas)
        opts.SetOption(solver_id, "Major optimality tolerance", optm)
        opts.SetOption(solver_id, "Major iterations limit", iters)
    elif solver_name in ("Ipopt", "IPOPT"):
        opts.SetOption(solver_id, "constr_viol_tol", feas)
        opts.SetOption(solver_id, "tol", optm)
        opts.SetOption(solver_id, "max_iter", iters)
        opts.SetOption(solver_id, "acceptable_iter", 0)     # no acceptable soln, only desired
    elif solver_name == "NLopt":
        n = cfg["nlopt"]
        opts.SetOption(solver_id, NloptSolver.AlgorithmName(), "LD_SLSQP")      # default for nlopt but safety net
        opts.SetOption(solver_id, NloptSolver.ConstraintToleranceName(), feas)  
        opts.SetOption(solver_id, NloptSolver.XRelativeToleranceName(), 1e-8)  
        opts.SetOption(solver_id, NloptSolver.MaxEvalName(), n["max_eval"])      
        opts.SetOption(solver_id, NloptSolver.MaxTimeName(), n["max_time_sec"])
    return opts


def _solve_row(solve_record):
    """SolveRecord -> flat dict for CSV, dropping the q array (q lives in joints table)."""
    d = asdict(solve_record)
    d.pop("q", None)
    return d


def main():
    cfg = _load_config(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    solve_rows, joint_rows, constraint_rows = [], [], []
    scenario_counter = 0

    logger.info("study start: %d seeds x %d tols x %d scenarios",
                len(cfg["seeds"]), len(cfg["orientation_tols"]), cfg["n_per_cell"])

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
                solver_ids = GetAvailableSolvers(GetProgramType(prog))
                for solver_id in solver_ids:
                    solver = MakeSolver(solver_id)
                    solver_name = solver.solver_id().name()
                    sid = solver.solver_id()
                    solve_id = f"{seed}_{orientation_tol}_{scenario_id}_{solver_name}"

                    logger.debug("solve %s", solve_id)
                    base_opts = _solver_options(sid, solver_name, cfg)
                    opts, print_path = native_optimality_options(solver_name, base_opts)
                    try:
                        result = solver.Solve(prog, scenario.q0, opts)
                    except Exception as exc:  # noqa: BLE001 - record and continue
                        logger.exception("solver raised %s", solve_id)
                        solve_rows.append({
                            "solve_id": solve_id, 
                            "scenario_id": scenario_id,
                            "seed": seed, 
                            "orientation_tol": orientation_tol,
                            "solver_name": solver_name,
                            "generation_method": scenario.generation_method, 
                            "success": False,
                            "solution_result": None, 
                            "optimal_cost": float("nan"),
                            "evaluated_cost": float("nan"), 
                            "feasible": False,
                            "max_violation": float("nan"),
                            "common_residual_inf": None, 
                            "common_residual_2": None,
                            "native_optimality": None,
                            "error_type": type(exc).__name__, 
                            "error_message": str(exc),
                        })
                        continue

                    native_opt = parse_native_optimality(solver_name, print_path)
                    tables = build_solve_tables(
                        prog, result,
                        solve_id=solve_id, scenario_id=scenario_id, seed=seed,
                        orientation_tol=orientation_tol, solver_name=solver_name,
                        generation_method=scenario.generation_method, 
                        feasibility_threshold=feas_threshold,
                        native_optimality=native_opt,
                    )
                    solve_rows.append(_solve_row(tables.solve))
                    joint_rows.extend(asdict(j) for j in tables.joints)
                    constraint_rows.extend(asdict(c) for c in tables.constraints)

            logger.info("done seed=%s tol=%s (scenarios so far: %d)",
                        seed, orientation_tol, scenario_counter)

    pd.DataFrame(solve_rows).to_csv(RESULTS_DIR / "solves.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(RESULTS_DIR / "joints.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(RESULTS_DIR / "constraints.csv", index=False)
    logger.info("wrote %d solves, %d joint rows, %d constraint rows to %s",
                len(solve_rows), len(joint_rows), len(constraint_rows), RESULTS_DIR)


if __name__ == "__main__":
    main()