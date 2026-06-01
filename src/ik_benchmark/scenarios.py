"""
Scenario generation.

The Scenario describes an IK problem
    (target pose, fixed initial guess, metadata)
Provides the Scenario dataclass and seeded generators for both 
joint-space sampling and task-space sampling. 

All randomness flows through one seed for reproducibility.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A fully-specified IK problem instance. Frozen so it cannot be mutated."""

    id: int
    target_position: np.ndarray          # (3,) [x, y, z] in world
    target_orientation: np.ndarray       # (3, 3) rotation matrix R_WG
    q0: np.ndarray                        # (nq,) fixed initial guess, shared across solvers
    generation_method: str               # "joint_space" | "task_space"
    seed: int                            # per-scenario seed that produced it


def generate_scenarios(plant, n, base_seed, method_mix, task_space_params):
    """Produce a deterministic list of Scenario objects.

    Same base_seed always yields the identical list.

    Returns
    -------
    list[Scenario]
    """
    raise NotImplementedError