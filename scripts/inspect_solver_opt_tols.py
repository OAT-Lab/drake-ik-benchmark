"""Diagnostic: how each solver tests optimality on the IK problem.

***not part of the benchmark.


reported optimality measure 
    IPOPT: Dual infeasibility 
    SNOPT: Max Dual infeas

terminal multiplier arrays from solver_details
    SNOPT: Fmul (general-constraint mults), xmul (variable/bound mults = reduced grads)
    IPOPT: lambda_val (general), z_L / z_U (bound)

per-binding duals via GetDualSolution

solver-blind full Lagrangian residual
r = sum_costs grad_f - sum_constraints J_stored^T y
"""

from __future__ import annotations

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

OUT_DIR = Path("results/optimality_inspect")

_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "cone_half_angle_rad": 0.5,
}

# lines to pull from print files
_PRINTFILE_MARKERS = {
    "IPOPT": ["Dual infeasibility", "Constraint violation",
              "Variable bound violation", "Complementarity", "Overall NLP error"],
    "SNOPT": ["Max Primal infeas", "Max Dual infeas", "INFO", "EXIT"],
}


def _binding_label(binding) -> str:
    """readable name: description if set, else evaluator type + variable indices."""
    ev = binding.evaluator()
    desc = ev.get_description() 
    base = desc if desc else type(ev).__name__
    return base


def stationarity_residual(prog, result):
    """common residual  r = sum_costs grad_f - sum_constraints J_stored^T y.

    grad_f and J from each evaluator's autodiff (stored sense, matching its bounds)
    
    y from GetDualSolution (shadow price). Returns (r, per_binding_duals, missing)."""
    x = result.GetSolution(prog.decision_variables())
    r = np.zeros(prog.num_vars())

    for b in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(b.variables())
        val = b.evaluator().Eval(InitializeAutoDiff(x[idx]))
        r[idx] += ExtractGradient(val)[0]          # (1,k) -> (k,)

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
        val = b.evaluator().Eval(InitializeAutoDiff(x[idx]))
        J = ExtractGradient(val)                   # (m,k) stored-sense Jacobian
        r[idx] -= J.T @ y
    return r, duals, missing


def _dump_solver_details(name, details):
    """Print the raw multiplier arrays each solver exposes."""
    def show(attr):
        if hasattr(details, attr):
            v = np.atleast_1d(getattr(details, attr))
            print(f"    {attr:11s}: {np.array2string(np.asarray(v, float), precision=4, max_line_width=120)}")
    if name == "SNOPT":
        for a in ("Fmul", "xmul", "F", "info"):
            show(a)
    elif name == "IPOPT":
        for a in ("lambda_val", "z_L", "z_U", "g", "status"):
            show(a)


def _parse_printfile(name, print_file: Path):
    """Pull the solver's own optimality lines out of its print file."""
    if not print_file.exists():
        print("    (no print file)")
        return
    text = print_file.read_text(errors="replace")
    markers = _PRINTFILE_MARKERS.get(name, [])
    for line in text.splitlines():
        if any(m in line for m in markers):
            print(f"    | {line.strip()}")
    # SNOPT also prints a per-variable reduced-gradient listing
    if name == "SNOPT":
        grab = False
        for line in text.splitlines():
            if "Reduced Gradnt" in line:
                print(f"    | {line.strip()}")
                grab = True
                continue
            if grab:
                if line.strip() == "" :
                    grab = False
                elif (" LL " in line) or (" UL " in line):
                    print(f"    | {line.strip()}")


def _inspect_one(solver_id, prog, q0):
    solver = MakeSolver(solver_id)
    name = solver.solver_id().name()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_file = OUT_DIR / f"{name}_optprint.txt"

    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintFileName, str(print_file))
    if name == "IPOPT":
        # default Drake print_level with a file is 4; bump for the full KKT terminal block
        opts.SetOption("IPOPT", "print_level", 5)
    if name == "NLopt":
        # SLSQP only 
        opts.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(), "LD_SLSQP")
        opts.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 100000)

    result = solver.Solve(prog, q0, opts)

    print(f"--- {name} ---")
    print(f"  success            : {result.is_success()}")
    print(f"  solution_result    : {result.get_solution_result()}")
    print(f"  optimal_cost       : {result.get_optimal_cost():.6e}")

    print("  [1] solver's own optimality lines (from print file):")
    _parse_printfile(name, print_file)

    print("  [2] raw terminal multiplier arrays (solver_details):")
    _dump_solver_details(name, result.get_solver_details())

    r, duals, missing = stationarity_residual(prog, result)

    print("  [3] per-binding duals (GetDualSolution, shadow-price convention):")
    if duals:
        for label, y in duals.items():
            print(f"    {label:42s}: {np.array2string(y, precision=5, max_line_width=120)}")
    if missing:
        print(f"    (no dual available for: {sorted(set(missing))})")

    print("  [4] common full-Lagrangian stationarity residual:")
    if missing:
        print("    N/A -- some/all duals unavailable (expected for NLopt); cannot reconstruct.")
    else:
        print(f"    ||grad f - sum J^T y||_inf = {np.max(np.abs(r)):.3e}")
        # split: how much of the residual sits on variables that are AT a joint limit
        print(f"    ||.||_2                    = {np.linalg.norm(r):.3e}")
    print(f"  print file: {print_file}")
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