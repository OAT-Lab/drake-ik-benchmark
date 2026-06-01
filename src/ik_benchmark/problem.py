"""
Build a Drake MathematicalProgram from a Scenario.

Encodes a Scenario into Drake's InverseKinematics
(position constraint, orientation constraint, mindistlowerbound, quadratic error cost). 

- Separate from scenario generation so the abstract problem description and its
Drake encoding can change independently.
"""

from __future__ import annotations


def build_ik_prog(plant, plant_context, gripper_frame, scenario, orientation_tol):
    """Construct the IK MathematicalProgram for one scenario.

    Returns
    -------
    (prog, q)
        prog : MathematicalProgram
        q : decision-variable vector (for setting the initial guess)
    """
    raise NotImplementedError