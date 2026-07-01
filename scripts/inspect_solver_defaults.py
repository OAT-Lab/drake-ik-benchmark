"""Diagnostic: find which solver parameters engage on the IK problem, and their defaults.

***not part of the benchmark.

for finding the parameters/defaults used by each solver
"""

from __future__ import annotations

import argparse
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

from pydrake.solvers import (
    CommonSolverOption,
    GetAvailableSolvers,
    GetProgramType,
    MakeSolver,
    SolverOptions,
)

from ik_benchmark.plant import build_plant
from ik_benchmark.scenarios import generate_scenarios
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.logging import configure_logging
from pydrake.solvers import NloptSolver

OUT_DIR = Path("results/solver_defaults")
logger = logging.getLogger("ik_benchmark.scripts.inspect_solver_defaults")

# Minimal inline settings for the throwaway diagnostic (not the full config).
_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "max_tilt_rad": 0.5,
}

# potential ones
_NLOPT_ALGOS = ["GN_ISRES", "LN_COBYLA", "LD_SLSQP"] # only using SLSQP

def _probe_nlopt_algorithms(prog, q0, print_dir):
    for algo in _NLOPT_ALGOS:
        opts = SolverOptions()
        opts.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(), algo)
        opts.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 2000)  # never hang
        opts.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 100000)
        opts.SetOption(NloptSolver.id(), NloptSolver.MaxTimeName(), 60.0)  # seconds
        solver = NloptSolver()
        try:
            result = solver.Solve(prog, q0, opts)
            ok = result.is_success()
            # solver-blind feasibility check matters here: success flag is unreliable
            logger.info(
                "  %-10s: ran, is_success=%s, result=%s",
                algo,
                ok,
                result.get_solution_result(),
            )
        except Exception as e:
            logger.info("  %-10s: FAILED -> %s: %s", algo, type(e).__name__, e)

def _verbose_options(solver_name, print_file):
    """SolverOptions that maximize each solver's self-reporting of its settings."""
    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintToConsole, 1)
    opts.SetOption(CommonSolverOption.kPrintFileName, str(print_file))
    if solver_name == "IPOPT":
        opts.SetOption("IPOPT", "print_level", 12)
        opts.SetOption("IPOPT", "print_user_options", "yes")
        opts.SetOption("IPOPT", "print_advanced_options", "yes")
        opts.SetOption("IPOPT", "print_options_documentation", "yes")
    # NLopt doesn't work
    return opts


def _inspect_one(solver_id, prog, q0):
    solver = MakeSolver(solver_id)
    name = solver.solver_id().name()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_file = OUT_DIR / f"{name}_print.txt"

    opts = _verbose_options(name, print_file)

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = solver.Solve(prog, q0, opts)
    console_dump = buf.getvalue()
    (OUT_DIR / f"{name}_console.txt").write_text(console_dump)

    logger.info("--- %s ---", name)
    logger.info("  success     : %s", result.is_success())
    logger.info("  solver used : %s", result.get_solver_id().name())
    logger.info(
        "  console dump: %s (%s chars)",
        OUT_DIR / f"{name}_console.txt",
        len(console_dump),
    )
    if name == "NLopt": 
        _probe_nlopt_algorithms(prog, q0, "results/NLopt_probe.txt")
    if print_file.exists():
        logger.info("  print file  : %s (%s bytes)", print_file, print_file.stat().st_size)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect solver parameters/defaults used on the IK problem."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Minimum ik_benchmark log level.",
    )
    parser.add_argument(
        "--log-file",
        default=OUT_DIR / "inspect_solver_defaults.log",
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

    plant, plant_context, gripper_frame, q0 = build_plant(_TIME_STEP, _BOX)

    scenario = generate_scenarios(
        plant, plant_context, gripper_frame, q0,
        n=1, base_seed=0,
        method_mix={"joint_space": 1.0, "task_space": 0.0},
        task_space_params=_TASK_SPACE,
    )[0]

    prog, q = build_ik_prog(
        plant, plant_context, gripper_frame, scenario,
        orientation_tol=0.0,
        use_minimum_distance=True,
        min_distance_bound=0.001,
        influence_distance_offset=0.1,
    )

    program_type = GetProgramType(prog)
    solver_ids = GetAvailableSolvers(program_type)
    logger.info("Program type: %s", program_type)
    logger.info("Capable + available solvers: %s\n", [sid.name() for sid in solver_ids])

    for solver_id in solver_ids:
        _inspect_one(solver_id, prog, scenario.q0)

    logger.info("Dumps written under %s/.", OUT_DIR)


if __name__ == "__main__":
    main()
