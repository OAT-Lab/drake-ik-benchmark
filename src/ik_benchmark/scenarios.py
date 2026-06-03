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
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common.eigen_geometry import AngleAxis


_NOMINAL_DOWNWARD = RotationMatrix.MakeXRotation(np.pi)


@dataclass(frozen=True) # not mutable
class Scenario:
    id: int
    target_position: np.ndarray          # (3,) world [x, y, z]
    target_orientation: np.ndarray       # (3, 3) R_WG
    q0: np.ndarray                        # (nq,) nominal posture: cost ref + initial guess
    generation_method: str               # "joint_space" | "task_space"
    seed: tuple


def _assert_finite_limits(lower, upper):
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError(
            "Joint-space sampling requires finite position limits. Infinite lower: "
            f"{np.where(~np.isfinite(lower))[0].tolist()}; upper: "
            f"{np.where(~np.isfinite(upper))[0].tolist()}."
        )


def _gripper_pose(plant, plant_context, gripper_frame, q):
    plant.SetPositions(plant_context, q)
    return plant.EvalBodyPoseInWorld(plant_context, gripper_frame.body())


def _sample_joint_space(plant, plant_context, gripper_frame, lower, upper, rng):
    q = rng.uniform(lower, upper)
    X_WG = _gripper_pose(plant, plant_context, gripper_frame, q)
    return X_WG.translation(), X_WG.rotation().matrix()


def _sample_cone_orientation(rng, cone_half_angle_rad):
    """random orientation: nominal downward grasp tilted within cone

    cone_half_angle_rad = 0 gives exact nominal orientation
    """
    axis = rng.normal(size=3) # random unit axis
    norm = np.linalg.norm(axis)
    if norm < 1e-12:            # fallback
        axis, norm = np.array([1.0, 0.0, 0.0]), 1.0
    axis = axis / norm
    angle = rng.uniform(0.0, cone_half_angle_rad)
    tilt = RotationMatrix(AngleAxis(angle=angle, axis=axis))
    return (tilt @ _NOMINAL_DOWNWARD).matrix()


def _sample_task_space(rng, position_min, position_max, cone_half_angle_rad):
    position = rng.uniform(position_min, position_max)
    orientation = _sample_cone_orientation(rng, cone_half_angle_rad)
    return position, orientation


def generate_scenarios(
    plant, plant_context, gripper_frame, q0,
    n, base_seed, method_mix, task_space_params,
):
    """produce list of Scenario objects

    Parameters: 
    plant, plant_context, gripper_frame
        From build_plant(); used for joint limits and forward kinematics.
    q0 : np.ndarray
        The nominal/comfortable posture, attached to every scenario.
    n : int
        Total number of scenarios.
    base_seed : int
        Master seed; each scenario's own seed derives from it deterministically.
    method_mix : dict
        e.g. {"joint_space": 0.5, "task_space": 0.5}; fractions must sum to 1.0.
    task_space_params : dict
        {"position_min": [x, y, z], "position_max": [x, y, z],
         "cone_half_angle_rad": float}.

    Returns: 
    list[Scenario]
        Same base_seed always yields the identical list.
    """
    if abs(sum(method_mix.values()) - 1.0) > 1e-9:
        raise ValueError(f"method_mix must sum to 1.0, got {sum(method_mix.values())}")

    lower = plant.GetPositionLowerLimits()
    upper = plant.GetPositionUpperLimits()
    _assert_finite_limits(lower, upper)

    n_joint = int(round(n * method_mix.get("joint_space", 0.0)))
    methods = ["joint_space"] * n_joint + ["task_space"] * (n - n_joint)

    pos_min = np.asarray(task_space_params["position_min"], dtype=float)
    pos_max = np.asarray(task_space_params["position_max"], dtype=float)
    cone = float(task_space_params["cone_half_angle_rad"])

    seed_seq = np.random.SeedSequence(base_seed) # generates n random seeds from base seed
    children = seed_seq.spawn(n)

    scenarios = []
    for i, method in enumerate(methods):
        rng = np.random.default_rng(children[i])
        if method == "joint_space":
            position, orientation = _sample_joint_space(
                plant, plant_context, gripper_frame, lower, upper, rng
            )
        else:
            position, orientation = _sample_task_space(rng, pos_min, pos_max, cone)

        scenarios.append(Scenario(
            id=i,
            target_position=position,
            target_orientation=orientation,
            q0=np.asarray(q0, dtype=float).copy(),
            generation_method=method,
            seed=(base_seed, i),
        ))

    plant.SetPositions(plant_context, q0) # set plant back to nominal pos
    return scenarios