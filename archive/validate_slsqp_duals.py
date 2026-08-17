"""validate the SLSQP dual patch on clean IK solves.
"""

import logging
from datetime import datetime
from pathlib import Path

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
RESULTS_DIR = Path(__file__).resolve().parents[1] / "tests" / "results"

logger = logging.getLogger("validate_slsqp_duals")


def configure_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"slsqp_duals_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    return log_path


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

def find_joint_limit_bbox(prog):
    for b in prog.bounding_box_constraints():
        if b.evaluator().get_description() == "Joint limits":
            return b
    # fallback: the bbox spanning all decision vars
    for b in prog.bounding_box_constraints():
        if b.evaluator().num_constraints() == prog.num_vars():
            return b
    return None

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

    logger.info("\n  %s", name)
    logger.info("    success=%s  "
                "result=%s  "
                "finite_q=%s",
                result.is_success(),
                result.get_solution_result(),
                np.all(np.isfinite(x)))
    
    if name == "NLopt":
        logger.info("    nlopt status: %s", result.get_solver_details().status)

    if not np.all(np.isfinite(x)):
        logger.info("    non-finite q -- nothing further to check")
        return

    for b, y in zip(constraints, duals):
        rows = b.evaluator().num_constraints()
        if y is None:
            logger.info("    %-32s rows=%-3s dual=MISSING", label(b), rows)
        else:
            flag = "" if len(y) == rows else "   <-- LENGTH MISMATCH"
            logger.info("    %-32s rows=%-3s dual=%s%s", label(b), rows, len(y), flag)

    keep = [not isinstance(b.evaluator(), BoundingBoxConstraint)
            for b in constraints]
    c_sub = [b for b, k in zip(constraints, keep) if k]
    d_sub = [d for d, k in zip(duals, keep) if k]

    pos_i = next((i for i, b in enumerate(c_sub) if label(b) == "PositionConstraint"), None)
    if pos_i is not None and d_sub[pos_i] is not None:
        d_flip = [None if d is None else d.copy() for d in d_sub]
        d_flip[pos_i][1] *= -1
        logger.info("    residual_flip1   = %s", _stationarity(prog, c_sub, d_flip, x))
        
    logger.info("    q     = %s", x)
    for b, y in zip(constraints, duals):
        if y is not None:
            logger.info("    %-32s y=%s", label(b), y)

    neg = [None if d is None else -d for d in d_sub]
    logger.info("    residual_neg     = %s", _stationarity(prog, c_sub, neg, x))

    full = _stationarity(prog, constraints, duals, x)
    no_bbox = _stationarity(prog, c_sub, d_sub, x)
    logger.info("    residual_full    = %s", full)
    logger.info("    residual_no_bbox = %s", no_bbox)

    


def main():
    log_path = configure_logging()
    logger.info("logging to %s", log_path)

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
    opts = solver_options(ctol=1e-12)

    for orientation_tol in (0.0, 1e-3):
        logger.info("\n%s", "=" * 70)
        logger.info("orientation_tol = %s", orientation_tol)
        logger.info("%s", "=" * 70)

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
            logger.info("  SLSQP returned non-finite q on every scenario tried")
            continue

        logger.info("  scenario id=%s seed=%s", scenario.id, scenario.seed)
        # active-bound sign check
        # --- active-bound sign check -----------------------------------
        base = solvers["NLopt"].Solve(prog, scenario.q0, opts)
        q_star = base.GetSolution(q)
        bbox = find_joint_limit_bbox(prog)
        if np.all(np.isfinite(q_star)) and bbox is not None:
            j = 0
            lb0 = bbox.evaluator().lower_bound().copy()
            ub0 = bbox.evaluator().upper_bound().copy()

            lb_new = lb0.copy()
            lb_new[j] = q_star[j] + 5e-3          # pin lower bound above desired
            bbox.evaluator().set_bounds(lb_new, ub0)

            logger.info("\n  === ACTIVE-BOUND CHECK: joint %d lower pinned to %.6f "
                        "(wants %.6f) ===", j, lb_new[j], q_star[j])

            results_by_solver = {}
            for name in ("Ipopt", "NLopt"):
                seed = scenario.q0
                if name == "NLopt" and "Ipopt" in results_by_solver:
                    iq = results_by_solver["Ipopt"].GetSolution(q)
                    if np.all(np.isfinite(iq)):
                        seed = iq          # warm-start NLopt from IPOPT's solution
                r = solvers[name].Solve(prog, seed, opts)
                results_by_solver[name] = r
                cons, duals = collect_duals(prog, r)
                qv = r.GetSolution(q)
                bidx = next(i for i, b in enumerate(cons)
                            if label(b) == "BoundingBoxConstraint")
                bd = duals[bidx]
                at_bound = (np.all(np.isfinite(qv)) and
                            abs(qv[j] - lb_new[j]) < 1e-4)
                logger.info("    %-6s success=%s q[%d]=%.6f at_bound=%s bbox_dual[%d]=%s",
                            name, r.is_success(), j,
                            qv[j] if np.all(np.isfinite(qv)) else float("nan"),
                            at_bound, j,
                            "MISSING" if bd is None else f"{bd[j]:+.5f}")

            bbox.evaluator().set_bounds(lb0, ub0)  # restore
        # ---------------------------------------------------------------

        for ctol in (None, 1e-4, 1e-6, 1e-9, 1e-12):
            for xtol in (None, 1e-6, 1e-8, 1e-10):
                logger.info("\n  --- ctol=%s xtol=%s", ctol, xtol)
                r = solvers["NLopt"].Solve(prog, scenario.q0,
                                           solver_options(ctol, xtol))
                logger.info("      finite=%s status=%s",
                            np.all(np.isfinite(r.GetSolution(q))),
                            r.get_solver_details().status)
                
        for name, solver in solvers.items():
            if not solver.available(): 
                continue
            result = solver.Solve(prog, scenario.q0, opts)
            constraints, duals = collect_duals(prog, result)
            report(name, prog, result, constraints, duals)


if __name__ == "__main__":
    main()
