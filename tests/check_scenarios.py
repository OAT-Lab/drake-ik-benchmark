# save as check_scenarios.py, run: python check_scenarios.py
import numpy as np
from ik_benchmark.plant import build_plant
from ik_benchmark.scenarios import generate_scenarios

_TIME_STEP = 0.001
_BOX = {"size": [0.1, 0.1, 1.0], "position": [0.25, 0.0, 0.5]}
_TASK_SPACE = {
    "position_min": [0.3, -0.3, 0.2],
    "position_max": [0.6, 0.3, 0.6],
    "max_tilt_rad": 0.5,   # <-- use whatever you renamed it to
}

for seed in [0, 42]:
    plant, plant_context, gripper_frame, q0 = build_plant(_TIME_STEP, _BOX)
    for method in ("joint_space", "task_space"):
        mix = {"joint_space": 0.0, "task_space": 0.0}
        mix[method] = 1.0
        s = generate_scenarios(
            plant, plant_context, gripper_frame, q0,
            n=1, base_seed=seed, method_mix=mix,
            task_space_params=_TASK_SPACE,
        )[0]
        pos_ok = np.all(np.isfinite(s.target_position))
        rot_ok = np.all(np.isfinite(s.target_orientation))
        q0_ok = np.all(np.isfinite(s.q0))
        print(f"seed={seed} {method:11s}: "
              f"target_pos_finite={pos_ok} target_rot_finite={rot_ok} q0_finite={q0_ok}")