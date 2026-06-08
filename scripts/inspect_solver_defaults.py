"""Diagnostic: find which solver parameters engage on the IK problem, and their defaults.

***not part of the benchmark.

for finding the parameters/defaults used by each solver

Run it once per machine; save the output into the repo as the documented basis
for the benchmark's "consistent defaults" claim.

Authoritative default references (for interpreting the dumps):
  SNOPT : SNOPT User's Guide, section 7.7  (web.stanford.edu/group/SOL/guides/sndoc7.pdf)
  IPOPT : IPOPT Options Reference          (coin-or.github.io/Ipopt/OPTIONS.html)
  NLOPT : NLopt stopping-criteria section  (nlopt.readthedocs.io)
"""

from __future__ import annotations

import io
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
from pydrake.solvers import NloptSolver

OUT_DIR = Path("results/solver_defaults")

# Minimal inline settings for the throwaway diagnostic (not the full config).
_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "cone_half_angle_rad": 0.5,
}

# Algorithms one might naively consider; only SLSQP handles nonlinear equality constraints.
_NLOPT_ALGOS = ["LD_SLSQP", "LD_MMA", "LD_LBFGS", "LN_COBYLA", "LD_CCSAQ"]

def _probe_nlopt_algorithms(prog, q0, print_dir):
    for algo in _NLOPT_ALGOS:
        opts = SolverOptions()
        opts.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(), algo)
        opts.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 2000)  # never hang
        solver = NloptSolver()
        try:
            result = solver.Solve(prog, q0, opts)
            ok = result.is_success()
            # solver-blind feasibility check matters here: success flag is unreliable
            print(f"  {algo:10s}: ran, is_success={ok}, "
                  f"result={result.get_solution_result()}")
        except Exception as e:
            print(f"  {algo:10s}: FAILED -> {type(e).__name__}: {e}")

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
    # SNOPT writes its parameter summary to the print file (kPrintFileName).
    # NLopt does not self-report; defaults inferred from docs + nlopt_solver.cc.
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

    print(f"--- {name} ---")
    print(f"  success     : {result.is_success()}")
    print(f"  solver used : {result.get_solver_id().name()}")
    print(f"  console dump: {OUT_DIR / f'{name}_console.txt'} ({len(console_dump)} chars)")
    if name == "NLopt": 
        _probe_nlopt_algorithms(prog, q0, "results/NLopt_probe.txt")
    if print_file.exists():
        print(f"  print file  : {print_file} ({print_file.stat().st_size} bytes)")
    print()


def main():
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
    print(f"Program type: {program_type}")
    print(f"Capable + available solvers: {[sid.name() for sid in solver_ids]}\n")

    for solver_id in solver_ids:
        _inspect_one(solver_id, prog, scenario.q0)

    print(f"Dumps written under {OUT_DIR}/.")


if __name__ == "__main__":
    main()