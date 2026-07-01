# scratch_duals.py — put in studies/solver_values/, run from there:  python scratch_duals.py
import numpy as np
import yaml
from pathlib import Path

from pydrake.solvers import GetAvailableSolvers, GetProgramType, MakeSolver, IpoptSolver
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient

from ik_benchmark.plant import build_plant
from ik_benchmark.problem import build_ik_prog
from ik_benchmark.scenarios import generate_scenarios

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))

plant, plant_context, gripper_frame, q0 = build_plant(cfg["time_step"], cfg["box"])
scenario = generate_scenarios(
    plant, plant_context, gripper_frame, q0,
    n=1, base_seed=cfg["seeds"][0],
    method_mix=cfg["method_mix"],
    task_space_params=cfg["task_space"],
)[0]

prog, q = build_ik_prog(
    plant, plant_context, gripper_frame, scenario,
    orientation_tol=cfg["orientation_tols"][0],
    use_minimum_distance=cfg["minimum_distance"]["use_minimum_distance"],
    min_distance_bound=cfg["minimum_distance"]["min_distance_bound"],
    influence_distance_offset=cfg["minimum_distance"]["influence_distance_offset"],
)

result = IpoptSolver().Solve(prog, scenario.q0, None)
x = np.asarray(result.GetSolution(prog.decision_variables()), dtype=float)

print("success:", result.is_success())
print(f"{'constraint':40s} {'vals':>4} {'J_shape':>12} {'dual_len':>8} {'vars':>4}")
for b in prog.GetAllConstraints():
    ev = b.evaluator()
    idx = prog.FindDecisionVariableIndices(b.variables())
    J = ExtractGradient(ev.Eval(InitializeAutoDiff(x[idx])))
    try:
        y = np.atleast_1d(result.GetDualSolution(b))
        yl = str(len(y))
    except Exception as e:
        yl = f"ERR:{type(e).__name__}"
    print(f"{type(ev).__name__:40s} {ev.num_constraints():>4} "
          f"{str(J.shape):>12} {yl:>8} {len(idx):>4}")