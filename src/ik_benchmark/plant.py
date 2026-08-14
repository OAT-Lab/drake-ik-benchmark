"""
Build the iiwa7 + WSG MultibodyPlant.

The plant is constructed once and reused across every scenario. 
"""

from __future__ import annotations

import numpy as np
from pydrake.geometry import Box
from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from manipulation.scenarios import AddIiwa, AddShape, AddWsg


def build_plant(time_step, box_params):
    """
    build, populate, finalize the iiwa7 + WSG + box plant.

    Args:
        time_step: float
            MultibodyPlant time step.
        box_params: dict
            {"size": [x, y, z], "position": [x, y, z]} obstacle dims + 3d pos

    Returns:
        (plant, plant_context, gripper_frame, q0)
            plant: MultibodyPlant[float], finalized
            plant_context: plant's sub-context from default diagram context
            gripper_frame: the WSG body frame (frameB for IK constraints)
            q0: np.ndarray, the nominal joint pos read from the default context
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    iiwa = AddIiwa(plant, "with_box_collision")
    wsg = AddWsg(plant, iiwa, welded=True)

    size = np.asarray(box_params["size"], dtype=float)
    position = np.asarray(box_params["position"], dtype=float)
    box = AddShape(plant, Box(size[0], size[1], size[2]), "box")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("box", box),
        RigidTransform(position),
    )

    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body", wsg)

    return plant, plant_context, gripper_frame, q0