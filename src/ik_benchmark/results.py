"""
records for the solver-comparison study.

flatten one solver result into tables keyed by solve_id:
  solves      - one row per (scenario, solver)
  joints      - one row per (solve, joint)
  constraints - one row per (solve, constraint row)
  timings     - one row per (solve, timed repeat)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
from pydrake.solvers import BoundingBoxConstraint

from ik_benchmark.measure import (
    _binding_label,
    _joint_bounds_and_duals,
    _row_violation,
    _stationarity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveRecord:
    solve_id: str
    scenario_id: int
    seed: int
    orientation_tol: float
    solver_name: str
    generation_method: str | None   # "joint_space" | "task_space"
    success: bool
    solution_result: str | None
    optimal_cost: float          # solver's own
    evaluated_cost: float        # solver-blind, evaluated at q
    feasible: bool               # solver-blind: max_violation < threshold
    max_violation: float         # over non bound constraints
    common_stationarity_inf: float | None
    common_stationarity_2: float | None
    native_optimality: float | None     # from solver
    q: np.ndarray
    time_median: float = np.nan         # headline; distribution in timings table
    n_repeats: int = 0
    timing_matches_record: bool = False
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def failure(cls, *, solve_id, scenario_id, seed, orientation_tol, solver_name,
                generation_method, error_type, error_message,
                solution_result=None, native_optimality=None, q=None,
                time_median=np.nan, n_repeats=0, timing_matches_record=False):
        """Row for a solve that produced no usable numbers."""
        return cls(
            solve_id=solve_id,
            scenario_id=scenario_id,
            seed=seed,
            orientation_tol=orientation_tol,
            solver_name=solver_name,
            generation_method=generation_method,
            success=False,
            solution_result=solution_result,
            optimal_cost=np.nan,
            evaluated_cost=np.nan,
            feasible=False,
            max_violation=np.nan,
            common_stationarity_inf=None,
            common_stationarity_2=None,
            native_optimality=native_optimality,
            q=np.full(0, np.nan) if q is None else q,
            time_median=time_median,
            n_repeats=n_repeats,
            timing_matches_record=timing_matches_record,
            error_type=error_type,
            error_message=error_message,
        )


@dataclass(frozen=True)
class JointRecord:
    solve_id: str
    joint_index: int
    joint_name: str
    q_value: float
    lower_bound: float
    upper_bound: float
    at_limit: bool
    dual: float | None
    stationarity_contribution: float | None  # per-joint entry of grad f - sum J^T y


@dataclass(frozen=True)
class ConstraintRecord:
    solve_id: str
    constraint_label: str
    row_index: int
    value: float
    lower_bound: float
    upper_bound: float
    violation: float
    dual: float | None


@dataclass(frozen=True)
class TimingRecord:
    solve_id: str
    repeat_index: int   # order run; warmups excluded
    seconds: float


@dataclass(frozen=True)
class SolveTables:
    solve: SolveRecord
    joints: list[JointRecord] = field(default_factory=list)
    constraints: list[ConstraintRecord] = field(default_factory=list)
    timings: list[TimingRecord] = field(default_factory=list)


def _timing_records(solve_id, times):
    return [TimingRecord(solve_id=solve_id, repeat_index=i, seconds=float(t))
            for i, t in enumerate(times)]


def build_solve_tables(
    prog,
    result,
    *,
    solve_id,
    scenario_id,
    seed,
    orientation_tol,
    solver_name,
    generation_method=None,
    feasibility_threshold,
    native_optimality=None,
    times=None,
    timing_matches_record=False,
    q_at_limit_tol=1e-9,
) -> SolveTables:
    """Flatten one solved prog into the tables."""
    times = np.asarray([] if times is None else times, dtype=float)
    time_median = float(np.median(times)) if times.size else np.nan
    timing_records = _timing_records(solve_id, times)

    x = np.asarray(result.GetSolution(prog.decision_variables()), dtype=float)
    constraints = list(prog.GetAllConstraints())            # pydrake
    duals = []
    for b in constraints:
        try:
            duals.append(np.atleast_1d(result.GetDualSolution(b)))  # pydrake
        except Exception:
            duals.append(None)

    if not np.all(np.isfinite(x)):
        logger.warning(
            "non-finite solution solver=%s scenario=%s seed=%s tol=%s",
            solver_name, scenario_id, seed, orientation_tol,
        )
        rec = SolveRecord.failure(
            solve_id=solve_id,
            scenario_id=scenario_id,
            seed=seed,
            orientation_tol=orientation_tol,
            solver_name=solver_name,
            generation_method=generation_method,
            solution_result=str(result.get_solution_result()),
            native_optimality=native_optimality,
            q=x,
            time_median=time_median,
            n_repeats=int(times.size),
            timing_matches_record=timing_matches_record,
            error_type="NonFiniteSolution",
            error_message=f"{int((~np.isfinite(x)).sum())} non-finite entries",
        )
        return SolveTables(solve=rec, timings=timing_records)   # time-to-NaN is still data

    # solver-blind cost
    evaluated_cost = 0.0
    for b in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(b.variables())
        evaluated_cost += float(np.atleast_1d(b.evaluator().Eval(x[idx]))[0])

    # stationarity (shares the single constraints/duals list, by position)
    stationarity_vec = _stationarity(prog, constraints, duals, x)
    if stationarity_vec is None:
        stationarity_inf = stationarity_2 = None
    else:
        stationarity_inf = float(np.max(np.abs(stationarity_vec)))
        stationarity_2 = float(np.linalg.norm(stationarity_vec))

    # constraint rows (general = non-bound); bound rows handled in joints table
    constraint_records = []
    max_violation = 0.0
    for b, y in zip(constraints, duals, strict=True):
        ev = b.evaluator()
        if isinstance(ev, BoundingBoxConstraint):
            continue
        idx = prog.FindDecisionVariableIndices(b.variables())
        vals = np.atleast_1d(ev.Eval(x[idx])).astype(float)
        lo = np.atleast_1d(ev.lower_bound()).astype(float)
        up = np.atleast_1d(ev.upper_bound()).astype(float)
        label = _binding_label(b)
        n_rows = len(vals)
        if y is not None and len(y) != n_rows:
            logger.warning(
                "dual/value length mismatch solve=%s constraint=%s "
                "n_vals=%d n_dual=%d - dropping dual for this constraint",
                solve_id, label, n_rows, len(y),
            )
            y = None
        for row, (lb, val, ub) in enumerate(zip(lo, vals, up, strict=True)):
            viol = _row_violation(val, lb, ub)
            max_violation = max(max_violation, viol)
            constraint_records.append(ConstraintRecord(
                solve_id=solve_id, constraint_label=label, row_index=row,
                value=float(val), lower_bound=float(lb), upper_bound=float(ub),
                violation=viol,
                dual=(float(y[row]) if y is not None else None),
            ))

    # joints table: q, joint-limit bounds, bound dual, per-joint stationarity contribution
    lower, upper, bound_dual = _joint_bounds_and_duals(prog, constraints, duals)
    variables = prog.decision_variables()
    joint_records = []
    for i in range(prog.num_vars()):
        at_limit = bool(
            abs(x[i] - lower[i]) <= q_at_limit_tol
            or abs(x[i] - upper[i]) <= q_at_limit_tol
        )
        joint_records.append(JointRecord(
            solve_id=solve_id, joint_index=i, joint_name=variables[i].get_name(),
            q_value=float(x[i]), lower_bound=float(lower[i]), upper_bound=float(upper[i]),
            at_limit=at_limit,
            dual=(float(bound_dual[i]) if bound_dual is not None
                  and np.isfinite(bound_dual[i]) else None),
            stationarity_contribution=(float(stationarity_vec[i])
                                   if stationarity_vec is not None else None),
        ))

    feasible = max_violation < feasibility_threshold
    rec = SolveRecord(
        solve_id=solve_id, scenario_id=scenario_id, seed=seed,
        orientation_tol=orientation_tol, solver_name=solver_name,
        generation_method=generation_method,
        success=bool(result.is_success()),
        solution_result=str(result.get_solution_result()),
        optimal_cost=float(result.get_optimal_cost()),
        evaluated_cost=evaluated_cost, feasible=feasible,
        max_violation=max_violation,
        common_stationarity_inf=stationarity_inf, common_stationarity_2=stationarity_2,
        native_optimality=native_optimality,
        q=x,
        time_median=time_median,
        n_repeats=int(times.size),
        timing_matches_record=timing_matches_record,
    )
    return SolveTables(solve=rec, joints=joint_records,
                       constraints=constraint_records, timings=timing_records)


def solve_row(solve_record):
    """SolveRecord -> flat dict for CSV, dropping the q array (q lives in joints table)."""
    d = asdict(solve_record)
    d.pop("q", None)
    return d
