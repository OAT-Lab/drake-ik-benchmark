"""
Scenario generation.

A Scenario is data, not code: it describes an IK problem (target pose, fixed
initial guess, metadata) but is not the Drake program itself. Provides the
Scenario dataclass and seeded generators for both joint-space sampling
(forward-kinematics to a reachable target) and task-space sampling (a pose drawn
directly in the workspace, possibly unreachable). All randomness flows through a
single seed for reproducibility.
"""