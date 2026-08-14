"""
The single execution code path.

run_one: every (scenario + solver) pair is executed here and
nowhere else, so no solver or scenario can be treated differently. 

- sets initial guess
- applies the shared pinned options
- discards warm-up runs and times only the solve
- hands result to the solver-blind feasibility measurement.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ik_benchmark.results import SolveRecord, SolveTables, build_solve_tables
from ik_benchmark.solvers import (
    build_shared_options,
    native_optimality_options,
    parse_native_optimality,
    quiet_options,
)

logger = logging.getLogger(__name__)


def _timed_solve(solver, prog, q0, opts_timed, n_warmup, n_repeats):
    """Wall clock of Solve() alone. Returns (times, last result)."""
    for _ in range(n_warmup):
        solver.Solve(prog, q0, opts_timed)

    times = np.empty(n_repeats)
    result = None
    for i in range(n_repeats):
        t0 = time.perf_counter()
        result = solver.Solve(prog, q0, opts_timed)
        times[i] = time.perf_counter() - t0

    return times, result


def _same_solution(qa, qb, atol=1e-12):
    """Did the timed and recorded passes land on the same point? NaN-aware."""
    qa, qb = np.asarray(qa), np.asarray(qb)
    if qa.shape != qb.shape:
        return False
    both_nan = ~np.isfinite(qa) & ~np.isfinite(qb)
    return bool(np.all(both_nan | np.isclose(qa, qb, rtol=0, atol=atol)))


def run_one(
    prog,
    solver,
    scenario,
    *,
    solve_id,
    scenario_id,
    seed,
    orientation_tol,
    cfg,
    feasibility_threshold,
    n_warmup,
    n_repeats,
):
    """Solve one prog with one solver under controlled conditions.

    Returns
        SolveTables
    """
    solver_name = solver.solver_id().name()
    sid = solver.solver_id()

    logger.debug("solve %s", solve_id)
    try:
        # recorded pass: print file on
        opts_record, print_path = native_optimality_options(
            solver_name, build_shared_options(sid, solver_name, cfg))
        result = solver.Solve(prog, scenario.q0, opts_record)
        native_opt = parse_native_optimality(solver_name, print_path)

        # timed pass: printing off
        opts_timed = quiet_options(
            solver_name, build_shared_options(sid, solver_name, cfg))
        times, timed_result = _timed_solve(
            solver, prog, scenario.q0, opts_timed, n_warmup, n_repeats)

        matches = _same_solution(timed_result.get_x_val(),
                                 result.get_x_val())
        if not matches:
            logger.warning("timed != recorded solve %s", solve_id)

    except Exception as exc:  
        logger.exception("solver raised %s", solve_id)
        return SolveTables(solve=SolveRecord.failure(
            solve_id=solve_id,
            scenario_id=scenario_id,
            seed=seed,
            orientation_tol=orientation_tol,
            solver_name=solver_name,
            generation_method=scenario.generation_method,
            error_type=type(exc).__name__,
            error_message=str(exc),
        ))

    return build_solve_tables(
        prog, result,
        solve_id=solve_id, scenario_id=scenario_id, seed=seed,
        orientation_tol=orientation_tol, solver_name=solver_name,
        generation_method=scenario.generation_method,
        feasibility_threshold=feasibility_threshold,
        native_optimality=native_opt,
        times=times,
        timing_matches_record=matches,
    )
