"""
Build the iiwa7 + WSG MultibodyPlant.

The plant is constructed once and reused across every scenario. Rebuilding per
scenario would be slow and a place for inconsistency to creep in. Only the prog
(constraints) and q0 change per scenario, not the plant.
"""

from __future__ import annotations


def build_plant():
    """Construct, weld, and finalize the iiwa7 + WSG plant.

    Returns
    -------
    (plant, plant_context, gripper_frame)
    """
    raise NotImplementedError