"""Diagnostic: how each solver tests optimality on the IK problem.

***not part of the benchmark.


reported optimality measure 
    IPOPT: Dual infeasibility 
    SNOPT: Max Dual infeas

terminal multiplier arrays from solver_details
    SNOPT: Fmul (general-constraint mults), xmul (variable/bound mults)
    IPOPT: lambda_val (general), z_L / z_U (bound)

per-binding duals via GetDualSolution

solver-blind full Lagrangian residual
r = sum_costs grad_f - sum_constraints J_stored^T y
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
from pydrake.solvers import (
    CommonSolverOption,
    GetAvailableSolvers,
    GetProgramType,
    MakeSolver,
    NloptSolver,
    SolverOptions,
)
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient

from ik_benchmark.plant import build_plant
from ik_benchmark.scenarios import generate_scenarios
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.logging import configure_logging

OUT_DIR = Path("results/optimality_inspect")
logger = logging.getLogger("ik_benchmark.scripts.inspect_solver_opt_tols")

_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "max_tilt_rad": 0.5,
}

# lines to pull from print files
_PRINTFILE_MARKERS = {
    "IPOPT": ["Dual infeasibility", "Constraint violation",
              "Variable bound violation", "Complementarity", "Overall NLP error"],
    "SNOPT": ["Max Primal infeas", "Max Dual infeas", "INFO", "EXIT"],
}

SEEDS = [0, 42]
ORIENTATION_TOLS = [0.0, 0.0001]


def _binding_label(binding) -> str:
    """readable name: description if set, else evaluator type + variable indices."""
    ev = binding.evaluator()
    desc = ev.get_description() 
    base = desc if desc else type(ev).__name__
    return base


def stationarity_residual(prog, result):
    """common residual  r = sum_costs grad_f - sum_constraints J_stored^T y.

    grad_f and J from each evaluator's autodiff (stored sense, matching its bounds)
    
    y from GetDualSolution. 
    
    Returns (r, per_binding_duals, missing)."""
    x = result.GetSolution(prog.decision_variables())
    r = np.zeros(prog.num_vars())

    for b in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(b.variables())
        val = b.evaluator().Eval(InitializeAutoDiff(x[idx]))
        r[idx] += ExtractGradient(val)[0]         

    duals = {}
    missing = []
    for b in prog.GetAllConstraints():
        label = _binding_label(b)
        try:
            y = np.atleast_1d(result.GetDualSolution(b))
        except Exception as e:
            missing.append(label)
            continue
        duals[label] = y
        idx = prog.FindDecisionVariableIndices(b.variables())
        val = b.evaluator().Eval(InitializeAutoDiff(x[idx])) # plant is double so uses same method for calculating J
        J = ExtractGradient(val) # (m, k)     
          
        r[idx] -= J.T @ y 
    return r, duals, missing


def _dump_solver_details(name, details):
    """print the raw multiplier arrays from each solver."""
    def show(attr):
        if hasattr(details, attr):
            v = np.atleast_1d(getattr(details, attr))
            logger.info(
                "    %-11s: %s",
                attr,
                np.array2string(np.asarray(v, float), precision=4, max_line_width=120),
            )
    if name == "SNOPT":
        for a in ("Fmul", "xmul", "F", "info"):
            show(a)
    elif name == "IPOPT":
        for a in ("lambda_val", "z_L", "z_U", "g", "status"):
            show(a)


def _parse_printfile(name, print_file: Path):
    """pull the solver's own optimality lines out of its print file."""
    if not print_file.exists():
        logger.info("    (no print file)")
        return
    text = print_file.read_text(errors="replace")
    markers = _PRINTFILE_MARKERS.get(name, [])
    for line in text.splitlines():
        if any(m in line for m in markers):
            logger.info("    | %s", line.strip())
    # SNOPT also prints a per-variable reduced-gradient listing
    if name == "SNOPT":
        grab = False
        for line in text.splitlines():
            if "Reduced Gradnt" in line:
                logger.info("    | %s", line.strip())
                grab = True
                continue
            if grab:
                if line.strip() == "" :
                    grab = False
                elif (" LL " in line) or (" UL " in line):
                    logger.info("    | %s", line.strip())


def _inspect_one(solver_id, prog, q0, seed, tol):
    solver = MakeSolver(solver_id)
    name = solver.solver_id().name()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_file = OUT_DIR / f"{name}_optprint_{seed}_or{tol}.txt"

    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintFileName, str(print_file))
    if name == "IPOPT":
        # default Drake print_level with a file is 4; bump for the full KKT terminal block
        opts.SetOption("IPOPT", "print_level", 5)
    if name == "NLopt":
        logger.info("skipping NLopt; no optimality print-file details available")
        return

    result = solver.Solve(prog, q0, opts)

    logger.info("--- %s (%s, or theta_bound: %s) ---", name, seed, tol)
    logger.info("  success            : %s", result.is_success())
    logger.info("  solution_result    : %s", result.get_solution_result())
    logger.info("  optimal_cost       : %.6e", result.get_optimal_cost())

    logger.info("  [1] solver's own optimality lines:")
    _parse_printfile(name, print_file)

    logger.info("  [2] raw multiplier arrays:")
    _dump_solver_details(name, result.get_solver_details())

    r, duals, missing = stationarity_residual(prog, result)

    logger.info("  [3] per-binding duals:")
    if duals:
        for label, y in duals.items():
            logger.info(
                "    %-42s: %s",
                label,
                np.array2string(y, precision=5, max_line_width=120),
            )
    if missing:
        logger.info("    (no dual available for: %s)", sorted(set(missing)))

    logger.info("  [4] common stationarity residual:")
    if missing:
        logger.info("    N/A - some/all duals unavailable")
    else:
        logger.info("    ||grad f - sum J^T y||_inf = %.3e", np.max(np.abs(r)))
        # split: how much of the residual sits on variables that are at a joint limit
        logger.info("    ||.||_2                    = %.3e", np.linalg.norm(r))
    logger.info("  print file: %s", print_file)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect solver optimality tests on the IK problem."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Minimum ik_benchmark log level.",
    )
    parser.add_argument(
        "--log-file",
        default=OUT_DIR / "inspect_solver_opt_tols.log",
        help="Path for logs.",
    )
    parser.add_argument(
        "--log-mode",
        default="w",
        choices=["w", "a"],
        help="File log mode.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console logs. File logs are still written when --log-file is set.",
    )
    args = parser.parse_args()
    configure_logging(
        level=args.log_level,
        log_file=args.log_file,
        mode=args.log_mode,
        quiet=args.quiet,
    )

    for seed in SEEDS: 
        for tol in ORIENTATION_TOLS: 
            plant, plant_context, gripper_frame, q0 = build_plant(_TIME_STEP, _BOX)

            scenario = generate_scenarios(
                plant, plant_context, gripper_frame, q0,
                n=1, base_seed=seed,
                method_mix={"joint_space": 1.0, "task_space": 0.0},
                task_space_params=_TASK_SPACE,
            )[0]

            prog, q = build_ik_prog(
                plant, plant_context, gripper_frame, scenario,
                orientation_tol=tol,
                use_minimum_distance=True,
                min_distance_bound=0.001,
                influence_distance_offset=0.1,
            )

            program_type = GetProgramType(prog)
            solver_ids = GetAvailableSolvers(program_type)
            logger.info("Program type: %s", program_type)
            logger.info(
                "Capable + available solvers: %s\n",
                [sid.name() for sid in solver_ids],
            )

            for solver_id in solver_ids:
                _inspect_one(solver_id, prog, scenario.q0, seed, tol)

            logger.info("Dumps written under %s/.", OUT_DIR)


if __name__ == "__main__":
    main()
