import numpy as np
from ik_benchmark.plant import build_plant

plant, ctx, gripper_frame, q0 = build_plant(
    0.001, {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]})

# zero configuration (arm straight up)
plant.SetPositions(ctx, np.zeros(plant.num_positions()))
p = plant.EvalBodyPoseInWorld(ctx, gripper_frame.body()).translation()
print("gripper z at q=0:", p[2])          # = 1.22 + WSG offset
print("gripper xy at q=0:", p[0], p[1]) 