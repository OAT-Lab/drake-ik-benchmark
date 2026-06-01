"""
Solver-independent feasibility measurement.

Judges a returned solution by re-evaluating the prog's own 
constraints against their bounds.

Keeps the feasibility metric solver-blind so tolerance 
differences cannot leak into it.
"""

from __future__ import annotations


def measure_feasibility(prog, q_solution, threshold):
    """Re-evaluate every constraint at q_solution and report violation.

    Returns
    -------
    dict
        {"max_violation": float, "is_feasible": bool, "per_constraint": {...}}
    """
    raise NotImplementedError