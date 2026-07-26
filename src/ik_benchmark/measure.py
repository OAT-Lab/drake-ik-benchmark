"""
Solver-independent measurements for solved MathematicalPrograms.
"""

from __future__ import annotations

import numpy as np
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient
from pydrake.solvers import BoundingBoxConstraint


def _binding_label(binding) -> str:
    ev = binding.evaluator()
    desc = ev.get_description() # name of evaluator
    return desc if desc else type(ev).__name__  


def _row_violation(value, lower, upper) -> float:
    return float(max(lower - value, 0.0, value - upper))


def _stationarity(prog, constraints, duals, x):
    """grad f - sum_i J_i^T y_i  at x

    Returns the length-num_vars residual vector, or None if any constraint has no dual 
    """
    r = np.zeros(prog.num_vars())

    # grad f
    for b in prog.GetAllCosts(): 
        idx = prog.FindDecisionVariableIndices(b.variables())   # vector of decision var indices
        r[idx] += ExtractGradient(b.evaluator().Eval(InitializeAutoDiff(x[idx])))[0]

    # constraint-force part: - sum_i J_i^T y_i, paired by position
    for b, y in zip(constraints, duals, strict=True):
        if y is None:
            return None      # missing a dual -> stationarity residual undefined
        idx = prog.FindDecisionVariableIndices(b.variables())
        J = ExtractGradient(b.evaluator().Eval(InitializeAutoDiff(x[idx])))
        r[idx] -= J.T @ y

    return r


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
