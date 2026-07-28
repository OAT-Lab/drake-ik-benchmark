"""validate the SLSQP dual patch on clean IK solves.
"""

import numpy as np

from pydrake.solvers import (
    BoundingBoxConstraint,
    IpoptSolver,
    NloptSolver,
    SnoptSolver,
    SolverOptions,
)

from ik_benchmark.plant import build_plant
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.scenarios import generate_scenarios

from ik_benchmark.measure import _stationarity

TIME_STEP = 0.001
BOX = {"size": [0.2, 0.2, 0.2], "position": [0.5, 0.0, 0.3]}
FEASIBILITY_TOL = 1e-6
MAX_SCENARIOS = 40  


def label(binding):
    return type(binding.evaluator()).__name__


def solver_options(ctol=None, xtol=None):
    opts = SolverOptions()
    opts.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(), "LD_SLSQP")
    opts.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), 100000)
    if ctol is not None:
        opts.SetOption(NloptSolver.id(), NloptSolver.ConstraintToleranceName(), ctol)
    if xtol is not None:
        opts.SetOption(NloptSolver.id(), NloptSolver.XRelativeToleranceName(), xtol)
    return opts


def collect_duals(prog, result):
    """One GetAllConstraints() call, duals aligned by position."""
    constraints = list(prog.GetAllConstraints())
    duals = []
    for b in constraints:
        try:
            duals.append(np.atleast_1d(result.GetDualSolution(b)))
        except Exception:
            duals.append(None)
    return constraints, duals


def report(name, prog, result, constraints, duals):
    x = result.GetSolution(prog.decision_variables())

    print(f"\n  {name}")
    print(f"    success={result.is_success()}  "
          f"result={result.get_solution_result()}  "
          f"finite_q={np.all(np.isfinite(x))}")
    
    if name == "NLopt":
        print(f"    nlopt status: {result.get_solver_details().status}")

    if not np.all(np.isfinite(x)):
        print("    non-finite q -- nothing further to check")
        return

    for b, y in zip(constraints, duals):
        rows = b.evaluator().num_constraints()
        if y is None:
            print(f"    {label(b):<32} rows={rows:<3} dual=MISSING")
        else:
            flag = "" if len(y) == rows else "   <-- LENGTH MISMATCH"
            print(f"    {label(b):<32} rows={rows:<3} dual={len(y)}{flag}")

    keep = [not isinstance(b.evaluator(), BoundingBoxConstraint)
            for b in constraints]
    c_sub = [b for b, k in zip(constraints, keep) if k]
    d_sub = [d for d, k in zip(duals, keep) if k]

    full = _stationarity(prog, constraints, duals, x)
    no_bbox = _stationarity(prog, c_sub, d_sub, x)
    print(f"    residual_full    = {full}")
    print(f"    residual_no_bbox = {no_bbox}")


def main():
    plant, plant_context, gripper_frame, q0 = build_plant(TIME_STEP, BOX)

    scenarios = generate_scenarios(
        plant, plant_context, gripper_frame, q0,
        n=MAX_SCENARIOS, base_seed=0,
        method_mix={"joint_space": 1.0, "task_space": 0.0},
        task_space_params={"position_min": [0.3, -0.3, 0.1],
                           "position_max": [0.7, 0.3, 0.6],
                           "max_tilt_rad": 0.3},
    )

    solvers = {"SNOPT": SnoptSolver(),
               "Ipopt": IpoptSolver(),
               "NLopt": NloptSolver()}
    opts = solver_options()

    for orientation_tol in (0.0, 1e-3):
        print("\n" + "=" * 70)
        print(f"orientation_tol = {orientation_tol}")
        print("=" * 70)

        for scenario in scenarios:
            prog, q = build_ik_prog(
                plant, plant_context, gripper_frame, scenario,
                orientation_tol=orientation_tol,
                use_minimum_distance=True,
                min_distance_bound=0.001,
                influence_distance_offset=0.1,
            )
            probe = solvers["NLopt"].Solve(prog, scenario.q0, opts)
            if np.all(np.isfinite(probe.GetSolution(q))):
                break
        else:
            print("  SLSQP returned non-finite q on every scenario tried")
            continue

        print(f"  scenario id={scenario.id} seed={scenario.seed}")

        for ctol in (None, 1e-4, 1e-6, 1e-9, 1e-12):
            for xtol in (None, 1e-6, 1e-8, 1e-10):
                print(f"\n  --- ctol={ctol} xtol={xtol}", flush=True)
                r = solvers["NLopt"].Solve(prog, scenario.q0,
                                           solver_options(ctol, xtol))
                print(f"      finite={np.all(np.isfinite(r.GetSolution(q)))} "
                      f"status={r.get_solver_details().status}", flush=True)
                
        for name, solver in solvers.items():
            if not solver.available(): 
                continue
            result = solver.Solve(prog, scenario.q0, opts)
            constraints, duals = collect_duals(prog, result)
            report(name, prog, result, constraints, duals)


if __name__ == "__main__":
    main()
