import numpy as np
from ik_benchmark.plant import build_plant

plant, ctx, gripper_frame, q0 = build_plant(0.001, {"size":[0.1,0.1,1.0],"position":[0.25,0.0,0.5]})

lower = plant.GetPositionLowerLimits()
upper = plant.GetPositionUpperLimits()

# base position of the arm (world origin of the gripper chain's root)
# sample the reachable cloud
rng = np.random.default_rng(0)
N = 20000
pts = np.empty((N,3))
for i in range(N):
    q = rng.uniform(lower, upper)
    plant.SetPositions(ctx, q)
    pts[i] = plant.EvalBodyPoseInWorld(ctx, gripper_frame.body()).translation()

print("reach cloud stats (gripper position over random q):")
print("  x:", pts[:,0].min(), pts[:,2].max())
for ax,name in enumerate("xyz"):
    lo, hi = np.percentile(pts[:,ax], [2.5, 97.5])
    print(f"  {name}: full [{pts[:,ax].min():.3f}, {pts[:,ax].max():.3f}]  "
          f"5-95% [{lo:.3f}, {hi:.3f}]")
print("  max radius from origin:", np.linalg.norm(pts, axis=1).max())
print("  base (q=q0 gripper pos):", end=" ")
plant.SetPositions(ctx, q0); print(plant.EvalBodyPoseInWorld(ctx, gripper_frame.body()).translation())