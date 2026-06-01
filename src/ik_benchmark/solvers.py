"""
Solver enumeration and option pinning.

Returns potential solvers: 
    - available (present and licensed in this Drake build) 
    - capable (able to represent the prog's non-convex nonlinear constraints). 

The set is enumerated from Drake rather than hardcoded. Also builds
one shared SolverOptions for same tolerances.
"""

from __future__ import annotations


def get_candidate_solvers(prog):
    """Return solver instances that are both available and capable of `prog`.

    Returns
    -------
    list[SolverInterface]
    """
    raise NotImplementedError


def build_shared_options(tolerances):
    """Construct one SolverOptions applying identical tolerances to all solvers.

    Returns
    -------
    SolverOptions
    """
    raise NotImplementedError