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
from pydrake.geometry import Box, Mesh, Role
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction

from manipulation.scenarios import AddIiwa, AddShape, AddWsg

def _add_mesh_collision(plant, scene_graph, model):
    inspector = scene_graph.model_inspector()
    for body_index in plant.GetBodyIndices(model):
        body = plant.get_body(body_index)
        frame_id = plant.GetBodyFrameIdOrThrow(body_index)
        for geometry_id in inspector.GetGeometries(frame_id, Role.kIllustration):
            shape = inspector.GetShape(geometry_id)
            if isinstance(shape, Mesh):
                plant.RegisterCollisionGeometry(
                    body, inspector.GetPoseInFrame(geometry_id), shape,
                    f"{body.name()}_mesh_collision", CoulombFriction(),
                )

def build_plant(time_step, box_params, collision_model="with_box_collision"):
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

    if collision_model == "mesh":
        iiwa = AddIiwa(plant, "no_collision")
        _add_mesh_collision(plant, scene_graph, iiwa)
    else:
        iiwa = AddIiwa(plant, collision_model)
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