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


def run_one(prog, q, solver, q0, options, n_repeats, n_warmup, threshold):
    """Solve one prog with one solver under controlled conditions.

    Returns
    -------
    Result
    """
    raise NotImplementedError