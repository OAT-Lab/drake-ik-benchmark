"""
Build a Drake MathematicalProgram from a Scenario.

Encodes a Scenario into Drake's InverseKinematics
(position constraint, orientation constraint, mindistlowerbound, quadratic error cost). 

- Separate from scenario generation so the abstract problem description and its
Drake encoding can change independently.
"""


from __future__ import annotations

import numpy as np
from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics


def build_ik_prog(
    plant, plant_context, gripper_frame, scenario,
    orientation_tol,
    use_minimum_distance,
    min_distance_bound,
    influence_distance_offset
):
    """Construct the IK MathematicalProgram for one scenario.

    Parameters:
    plant, plant_context, gripper_frame
        From build_plant().
    scenario : Scenario
        Supplies target_position, target_orientation, and q0.
    orientation_tol : float
        Angular tolerance (radians) on the orientation constraint.
    use_minimum_distance : bool
        Whether to add the collision-avoidance constraint. Requires the plant to
        have collision geometry registered with a SceneGraph.
    min_distance_bound : float
        d_min: minimum allowed signed distance between geometry pairs.
    influence_distance_offset : float
        Influence distance offset; pairs farther than this are ignored. Must be
        > min_distance_bound.

    Returns:
    (prog, q)
        prog : MathematicalProgram
        q : decision-variable vector (for setting the initial guess)
    """
    if use_minimum_distance: 
        if min_distance_bound >= influence_distance_offset: 
            raise ValueError(
                f"minimum distance lower bound constraint requires min_distance_bound < influence_distance_offset." 
                f"\ngot min_dist_bound: {min_distance_bound}, influence_distance_offset: {influence_distance_offset}")
    
    ik = InverseKinematics(plant, plant_context)
    q = ik.q()

    target_position = np.asarray(scenario.target_position, dtype=float)
    R_WG = RotationMatrix(np.asarray(scenario.target_orientation, dtype=float))

    ik.AddPositionConstraint(
        gripper_frame, [0.0, 0.0, 0.0], plant.world_frame(),
        target_position, target_position,
    )

    ik.AddOrientationConstraint(
        gripper_frame, RotationMatrix(),
        plant.world_frame(), R_WG,
        orientation_tol,
    )

    if use_minimum_distance:
        ik.AddMinimumDistanceLowerBoundConstraint(
            min_distance_bound, influence_distance_offset,
        )

    prog = ik.get_mutable_prog()
    prog.AddQuadraticErrorCost(
        np.identity(len(q)), np.asarray(scenario.q0, dtype=float), q,
    )

    return prog, q