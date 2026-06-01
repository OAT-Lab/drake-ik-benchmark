"""
Solver enumeration and option pinning.

Returns the solvers that pass both gates: available (present and licensed in this
Drake build) and capable (able to represent the prog's non-convex nonlinear
constraints). The set is enumerated from Drake rather than hardcoded. Also builds
one shared SolverOptions applying identical tolerances to every solver.
"""