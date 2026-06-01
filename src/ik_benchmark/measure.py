"""
Solver-independent feasibility measurement.

Judges a returned solution by re-evaluating the prog's own constraints against
their bounds, never by trusting a solver's success flag (whose tolerance differs
between solvers). Keeps the feasibility metric solver-blind so tolerance
differences cannot leak into it.
"""