"""
Ensures reproducibility

 Asserts:
    - The same seed produces identical scenarios
    - Scenarios are immutable once generated
    - The capable+available solver set is stable across scenarios. 
"""


from __future__ import annotations


def test_same_seed_identical_scenarios():
    """Same base_seed yields identical scenarios."""
    raise NotImplementedError


def test_scenarios_are_immutable():
    """Scenario instances are frozen and cannot be mutated."""
    raise NotImplementedError


def test_solver_set_stable_across_scenarios():
    """The capable+available solver set is the same for every scenario's prog."""
    raise NotImplementedError