"""
ik_benchmark: 
    a controlled benchmark of Drake's NLP solvers on iiwa7 + WSG inverse kinematics.

Every (scenario x solver) pair flows through a single code path (runner.run_one)
so that all solvers are evaluated under identical conditions.
"""

__version__ = "0.0.1"