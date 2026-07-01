"""records for the solver-comparison study.

flatten one solver result into tables keyed by solve_id:
  solves      - one row per (scenario, solver)
  joints      - one row per (solve, joint)
  constraints - one row per (solve, constraint row)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient
from pydrake.solvers import BoundingBoxConstraint

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
    common_residual_inf: float | None
    common_residual_2: float | None
    native_optimality: float | None     # from solver
    q: np.ndarray
    error_type: str | None = None
    error_message: str | None = None


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
    residual_contribution: float | None  # per-joint entry of grad f - sum J^T y


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
class SolveTables:
    solve: SolveRecord
    joints: list[JointRecord] = field(default_factory=list)
    constraints: list[ConstraintRecord] = field(default_factory=list)


def _binding_label(binding) -> str:
    ev = binding.evaluator()
    desc = ev.get_description()
    return desc if desc else type(ev).__name__


def _row_violation(value, lower, upper) -> float:
    return float(max(lower - value, 0.0, value - upper))


def _stationarity(prog, constraints, duals, x):
    """Common full-Lagrangian residual  grad f - sum_i J_i^T y_i  at x.

    `constraints` / `duals` are the SINGLE shared list + position-aligned duals
    built once in build_solve_tables. duals[i] is None where a dual was
    unavailable (e.g. NLopt provides none). Iterated together by position, so
    each J_i is paired with its own y_i -- no id() keying, no re-iteration.

    Returns the length-num_vars residual vector, or None if any constraint has
    no dual (residual is undefined without a full multiplier set).
    """
    r = np.zeros(prog.num_vars())

    # cost gradient part: grad f
    for b in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(b.variables())
        r[idx] += ExtractGradient(b.evaluator().Eval(InitializeAutoDiff(x[idx])))[0]

    # constraint-force part: - sum_i J_i^T y_i, paired by position
    for b, y in zip(constraints, duals, strict=True):
        if y is None:
            return None      # missing a dual -> stationarity residual undefined
        idx = prog.FindDecisionVariableIndices(b.variables())
        J = ExtractGradient(b.evaluator().Eval(InitializeAutoDiff(x[idx])))
        r[idx] -= J.T @ y

    return r


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
    q_at_limit_tol=1e-9,
) -> SolveTables:
    """Flatten one solved prog into the tables."""
    x = np.asarray(result.GetSolution(prog.decision_variables()), dtype=float)
    constraints = list(prog.GetAllConstraints())            # pydrake
    duals = []
    for b in constraints:
        try:
            duals.append(np.atleast_1d(result.GetDualSolution(b)))  # pydrake
        except Exception:
            duals.append(None)   # e.g. NLopt: no duals

    if not np.all(np.isfinite(x)):
        logger.warning(
            "non-finite solution solver=%s scenario=%s seed=%s tol=%s",
            solver_name, scenario_id, seed, orientation_tol,
        )
        rec = SolveRecord(
            solve_id=solve_id, scenario_id=scenario_id, seed=seed,
            orientation_tol=orientation_tol, solver_name=solver_name,
            generation_method=generation_method,
            success=False, solution_result=str(result.get_solution_result()),
            optimal_cost=np.nan, evaluated_cost=np.nan, feasible=False,
            max_violation=np.nan, common_residual_inf=None, common_residual_2=None,
            native_optimality=native_optimality,
            q=x,
            error_type="NonFiniteSolution",
            error_message=f"{int((~np.isfinite(x)).sum())} non-finite entries",
        )
        return SolveTables(solve=rec)

    # solver-blind cost
    evaluated_cost = 0.0
    for b in prog.GetAllCosts():
        idx = prog.FindDecisionVariableIndices(b.variables())
        evaluated_cost += float(np.atleast_1d(b.evaluator().Eval(x[idx]))[0])

    # stationarity residual (shares the single constraints/duals list, by position)
    residual_vec = _stationarity(prog, constraints, duals, x)
    if residual_vec is None:
        res_inf = res_2 = None
    else:
        res_inf = float(np.max(np.abs(residual_vec)))
        res_2 = float(np.linalg.norm(residual_vec))

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

    # joints table: q, joint-limit bounds, bound dual, per-joint residual contribution
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
            residual_contribution=(float(residual_vec[i])
                                   if residual_vec is not None else None),
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
        common_residual_inf=res_inf, common_residual_2=res_2,
        native_optimality=native_optimality,
        q=x,
    )
    return SolveTables(solve=rec, joints=joint_records, constraints=constraint_records)


def _joint_bounds_and_duals(prog, constraints, duals):
    """Intersect all BoundingBoxConstraints into per-variable and
    gather each variable's bound dual into an array.
    """
    n = prog.num_vars()
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    bound_dual = np.full(n, np.nan)
    any_dual = False
    for b, y in zip(constraints, duals, strict=True):
        ev = b.evaluator()
        if not isinstance(ev, BoundingBoxConstraint):
            continue
        idx = prog.FindDecisionVariableIndices(b.variables())
        lo = np.atleast_1d(ev.lower_bound()).astype(float)
        up = np.atleast_1d(ev.upper_bound()).astype(float)
        lower[idx] = np.maximum(lower[idx], lo)
        upper[idx] = np.minimum(upper[idx], up)
        if y is not None:
            any_dual = True
            for k, gi in enumerate(idx):
                bound_dual[gi] = y[k]
    return lower, upper, (bound_dual if any_dual else None)