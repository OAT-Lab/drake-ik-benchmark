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
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgram,  
    SolverOptions,
    GetAvailableSolvers,
    MakeSolver,
)
from pydrake.solvers import ProgramType 

from ik_benchmark.plant import build_plant
from ik_benchmark.scenarios import generate_scenarios
from ik_benchmark.problem import build_ik_prog


OUT_DIR = Path("results/solver_defaults")


def capable_available_solver_ids(prog):
    """Solver IDs that can represent `prog` AND are installed in this Drake build.

    GetAvailableSolvers(ProgramType) returns the installed solvers for a program
    type; we then keep only those whose capabilities satisfy this exact prog.
    """
    ids = []
    for solver_id in GetAvailableSolvers(ProgramType.kNonlinearProgramming):
        solver = MakeSolver(solver_id)
        if solver.available() and solver.enabled() \
                and solver.AreProgramAttributesSatisfied(prog):
            ids.append(solver_id)
    return ids


def verbose_options(solver_name, print_file):
    """Build a SolverOptions that maximizes each solver's self-reporting.

    kPrintToConsole / kPrintFileName are Drake's solver-agnostic switches.
    IPOPT additionally needs print_level cranked to dump its option list.
    """
    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintToConsole, 1)
    opts.SetOption(CommonSolverOption.kPrintFileName, str(print_file))

    # Solver-specific verbosity that makes the library echo its options.
    if solver_name == "IPOPT":
        # print_level 12 is maximum; this makes IPOPT echo the options in effect.
        opts.SetOption("IPOPT", "print_level", 12)
        opts.SetOption("IPOPT", "print_user_options", "yes")
    # SNOPT writes its optional-parameter summary to the print file automatically
    # at default verbosity; kPrintFileName above captures it.
    # NLopt does not self-report; defaults are inferred from what is/ isn't set.
    return opts


def inspect_one(solver_id, prog, q, q0):
    """Solve once with full verbosity; capture the solver's option dump."""
    solver = MakeSolver(solver_id)
    name = solver.solver_id().name()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_file = OUT_DIR / f"{name}_print.txt"

    opts = verbose_options(name, print_file)

    # Capture anything the solver writes to stdout (IPOPT prints there).
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = solver.Solve(prog, q0, opts)

    console_dump = buf.getvalue()
    (OUT_DIR / f"{name}_console.txt").write_text(console_dump)

    print(f"--- {name} ---")
    print(f"  success      : {result.is_success()}")
    print(f"  solver used  : {result.get_solver_id().name()}")
    print(f"  console dump : {OUT_DIR / f'{name}_console.txt'} "
          f"({len(console_dump)} chars)")
    if print_file.exists():
        print(f"  print file   : {print_file} ({print_file.stat().st_size} bytes)")
    print()


def main():
    plant, plant_context, gripper_frame = build_plant()

    # One representative, reachable scenario (joint-space so it's well-posed).
    scenario = generate_scenarios(
        plant, n=1, base_seed=0,
        method_mix={"joint_space": 1.0, "task_space": 0.0},
        task_space_params=None,
    )[0]

    prog, q = build_ik_prog(
        plant, plant_context, gripper_frame, scenario, orientation_tol=0.0
    )

    solver_ids = capable_available_solver_ids(prog)
    print(f"Capable + available solvers: {[sid.name() for sid in solver_ids]}\n")

    for solver_id in solver_ids:
        inspect_one(solver_id, prog, q, scenario.q0)

    print(f"Dumps written under {OUT_DIR}/. Read each solver's print/console file "
          f"to see the parameters it engaged and their values.")


if __name__ == "__main__":
    sys.exit(main())