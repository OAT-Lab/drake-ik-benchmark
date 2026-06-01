"""
The single execution code path.

run_one is the chokepoint: every (scenario x solver) pair is executed here and
nowhere else, so no solver or scenario can be treated differently. Sets the
initial guess explicitly, applies the shared pinned options, discards warm-up
runs and times only the solve, then hands the result to the solver-blind
feasibility measurement.
"""