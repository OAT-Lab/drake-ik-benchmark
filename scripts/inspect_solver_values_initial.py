"""Diagnostic: write solver-blind solution values to CSV files.

***not part of the benchmark.
produces 
objective, variable, and constraint values in structured tables for each solver

2 seeds, 2 orientation tolerances - add more for more structured tests

doesn't use default.yaml - throwaway script
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydrake.solvers import (
    BoundingBoxConstraint,
    GetAvailableSolvers,
    GetProgramType,
    MakeSolver,
    NloptSolver,
    SolverOptions,
)

from ik_benchmark.log_config import configure_logging


OUT_DIR = Path("results/primal_inspect")
SEEDS = [0, 42]
ORIENTATION_TOLS = [0.0, 0.0001]

# logging
_LOG_LEVEL = "INFO"
_LOG_FILE = OUT_DIR / "inspect_solver_primal_values.log"
_LOG_MODE = "w"
_LOG_QUIET = False

# setup and sampling 
_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "max_tilt_rad": 0.5,
}

SUMMARY_COLUMNS = [
    "solver_name",
    "seed",
    "orientation_tol",
    "success",
    "solution_result",
    "optimal_cost",
    "evaluated_total_cost",
    "max_constraint_violation",
    "error_type",
    "error_message",
]
COST_COLUMNS = ["solver_name", "seed", "orientation_tol", "cost_label", "value"]
VARIABLE_COLUMNS = [
    "solver_name",
    "seed",
    "orientation_tol",
    "var_index",
    "var_name",
    "lower_bound",
    "value",
    "upper_bound",
]
CONSTRAINT_COLUMNS = [
    "solver_name",
    "seed",
    "orientation_tol",
    "constraint_label",
    "row_index",
    "lower_bound",
    "value",
    "upper_bound",
    "violation",
]

logger = logging.getLogger(__name__)


def _binding_label(binding) -> str:
    evaluator = binding.evaluator()
    description = evaluator.get_description()
    return description if description else type(evaluator).__name__


def _evaluate_cost_rows(prog, x, run_keys):
    rows = []
    total = 0.0
    for binding in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(binding.variables())
        values = np.atleast_1d(binding.evaluator().Eval(x[idx])).astype(float)
        label = _binding_label(binding)
        for row_index, value in enumerate(values):
            cost_label = label if len(values) == 1 else f"{label}[{row_index}]"
            scalar_value = float(value)
            rows.append({**run_keys, "cost_label": cost_label, "value": scalar_value})
            total += scalar_value
    return rows, total


def _collect_variable_bounds(prog):
    lower = np.full(prog.num_vars(), -np.inf)
    upper = np.full(prog.num_vars(), np.inf)
    for binding in prog.GetAllConstraints():
        evaluator = binding.evaluator()
        if not isinstance(evaluator, BoundingBoxConstraint):
            continue
        idx = prog.FindDecisionVariableIndices(binding.variables())
        lower[idx] = np.maximum(lower[idx], evaluator.lower_bound())
        upper[idx] = np.minimum(upper[idx], evaluator.upper_bound())
    return lower, upper


def _decision_variable_rows(prog, x, bounds, run_keys):
    lower, upper = bounds
    variables = prog.decision_variables()
    return [
        {
            **run_keys,
            "var_index": i,
            "var_name": variables[i].get_name(),
            "lower_bound": float(lower[i]),
            "value": float(x[i]),
            "upper_bound": float(upper[i]),
        }
        for i in range(prog.num_vars())
    ]


def _row_violation(value, lower_bound, upper_bound):
    return float(max(lower_bound - value, 0.0, value - upper_bound))


def _constraint_rows(prog, x, run_keys):
    """Value-only constraint table (no Jacobians; double Eval).

    Joint-limit BoundingBoxConstraints are recorded here AND separately as
    variable bounds (variables.csv). To avoid double-counting joint-limit
    slack, ``max_constraint_violation`` is taken over the GENERAL
    (non-bounding-box) constraints only; bound satisfaction is reflected in
    the variable rows.
    """
    rows = []
    max_violation = 0.0
    for binding in prog.GetAllConstraints():
        evaluator = binding.evaluator()
        is_bound = isinstance(evaluator, BoundingBoxConstraint)
        idx = prog.FindDecisionVariableIndices(binding.variables())
        values = np.atleast_1d(evaluator.Eval(x[idx])).astype(float)
        lower = np.atleast_1d(evaluator.lower_bound()).astype(float)
        upper = np.atleast_1d(evaluator.upper_bound()).astype(float)
        label = _binding_label(binding)
        for row_index, (lower_bound, value, upper_bound) in enumerate(
            zip(lower, values, upper, strict=True)
        ):
            violation = _row_violation(value, lower_bound, upper_bound)
            if not is_bound:
                max_violation = max(max_violation, violation)
            rows.append(
                {
                    **run_keys,
                    "constraint_label": label,
                    "row_index": row_index,
                    "lower_bound": float(lower_bound),
                    "value": float(value),
                    "upper_bound": float(upper_bound),
                    "violation": violation,
                }
            )
    return rows, max_violation


def _solver_options(solver_name):
    options = SolverOptions()
    if solver_name == "NLopt":
        options.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 2000)
        options.SetOption(NloptSolver.id(), NloptSolver.MaxTimeName(), 60.0)
    return options


def _inspect_solver(solver_id, prog, q0, run_keys):
    solver = MakeSolver(solver_id)
    solver_name = solver.solver_id().name()
    row_keys = {**run_keys, "solver_name": solver_name}

    logger.debug(
        "solving solver=%s seed=%s orientation_tol=%s",
        solver_name,
        run_keys["seed"],
        run_keys["orientation_tol"],
    )
    try:
        result = solver.Solve(prog, q0, _solver_options(solver_name))
        x = np.asarray(result.GetSolution(prog.decision_variables()), dtype=float)

        if not np.all(np.isfinite(x)):
            logger.warning(
                "non-finite solution treated as failure solver=%s seed=%s tol=%s",
                solver_name, run_keys["seed"], run_keys["orientation_tol"],
            )
            return {                       # stops crash if q not finite
                "summary": [
                    {
                        **row_keys,
                        "success": False,
                        "solution_result": str(result.get_solution_result()),
                        "optimal_cost": np.nan,
                        "evaluated_total_cost": np.nan,
                        "max_constraint_violation": np.nan,
                        "error_type": "NonFiniteSolution",
                        "error_message": f"{int((~np.isfinite(x)).sum())} non-finite entries",
                    }
                ],
                "costs": [],
                "variables": [],
                "constraints": [],
            }

        # only reached when x is finite:
        cost_rows, evaluated_total_cost = _evaluate_cost_rows(prog, x, row_keys)
        constraint_rows, max_violation = _constraint_rows(prog, x, row_keys)
        variable_rows = _decision_variable_rows(prog, x, _collect_variable_bounds(prog), row_keys)
    except Exception as exc:  # noqa: BLE001 - diagnostic: record and continue
        logger.exception(
            "solver raised solver=%s seed=%s orientation_tol=%s",
            solver_name,
            run_keys["seed"],
            run_keys["orientation_tol"],
        )
        return {
            "summary": [
                {
                    **row_keys,
                    "success": False,
                    "solution_result": None,
                    "optimal_cost": np.nan,
                    "evaluated_total_cost": np.nan,
                    "max_constraint_violation": np.nan,
                    # error_type/message are populated ONLY for a raised
                    # exception; a solver that *returns* failure has
                    # success=False with error_type=None.
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            ],
            "costs": [],
            "variables": [],
            "constraints": [],
        }

    x = np.asarray(result.GetSolution(prog.decision_variables()), dtype=float)
    cost_rows, evaluated_total_cost = _evaluate_cost_rows(prog, x, row_keys)
    constraint_rows, max_violation = _constraint_rows(prog, x, row_keys)
    variable_rows = _decision_variable_rows(
        prog, x, _collect_variable_bounds(prog), row_keys
    )

    return {
        "summary": [
            {
                **row_keys,
                "success": result.is_success(),
                "solution_result": str(result.get_solution_result()),
                "optimal_cost": float(result.get_optimal_cost()),
                "evaluated_total_cost": evaluated_total_cost,
                "max_constraint_violation": max_violation,
                "error_type": None,
                "error_message": None,
            }
        ],
        "costs": cost_rows,
        "variables": variable_rows,
        "constraints": constraint_rows,
    }


def _write_csvs(out_dir, summary_rows, cost_rows, variable_rows, constraint_rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        out_dir / "summary.csv", index=False
    )
    pd.DataFrame(cost_rows, columns=COST_COLUMNS).to_csv(
        out_dir / "costs.csv", index=False
    )
    pd.DataFrame(variable_rows, columns=VARIABLE_COLUMNS).to_csv(
        out_dir / "variables.csv", index=False
    )
    pd.DataFrame(constraint_rows, columns=CONSTRAINT_COLUMNS).to_csv(
        out_dir / "constraints.csv", index=False
    )


def main():
    configure_logging(
        level=_LOG_LEVEL,
        log_file=_LOG_FILE,
        mode=_LOG_MODE,
        quiet=_LOG_QUIET,
    )

    from ik_benchmark.plant import build_plant
    from ik_benchmark.problem import build_ik_prog
    from ik_benchmark.scenarios import generate_scenarios

    summary_rows = []
    cost_rows = []
    variable_rows = []
    constraint_rows = []

    logger.info("writing primal inspection results to %s", OUT_DIR)
    for seed in SEEDS:
        for orientation_tol in ORIENTATION_TOLS:
            plant, plant_context, gripper_frame, q0 = build_plant(_TIME_STEP, _BOX)
            scenario = generate_scenarios(
                plant,
                plant_context,
                gripper_frame,
                q0,
                n=1,
                base_seed=seed,
                method_mix={"joint_space": 1.0, "task_space": 0.0},
                task_space_params=_TASK_SPACE,
            )[0]
            prog, _ = build_ik_prog(
                plant,
                plant_context,
                gripper_frame,
                scenario,
                orientation_tol=orientation_tol,
                use_minimum_distance=True,
                min_distance_bound=0.001,
                influence_distance_offset=0.1,
            )
            solver_ids = GetAvailableSolvers(GetProgramType(prog))
            logger.info(
                "inspecting seed=%s orientation_tol=%s solvers=%s",
                seed,
                orientation_tol,
                [solver_id.name() for solver_id in solver_ids],
            )
            for solver_id in solver_ids:
                rows = _inspect_solver(
                    solver_id,
                    prog,
                    scenario.q0,
                    {"seed": seed, "orientation_tol": orientation_tol},
                )
                summary_rows.extend(rows["summary"])
                cost_rows.extend(rows["costs"])
                variable_rows.extend(rows["variables"])
                constraint_rows.extend(rows["constraints"])

    _write_csvs(OUT_DIR, summary_rows, cost_rows, variable_rows, constraint_rows)
    logger.info("wrote primal inspection CSVs under %s", OUT_DIR)


if __name__ == "__main__":
    main()