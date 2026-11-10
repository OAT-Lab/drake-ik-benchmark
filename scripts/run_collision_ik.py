"""
Full IK problems on the mesh-collision iiwa with one obstacle (a cube or a high-vertex convex)
placed at a fixed gap from the arm, solved once per collision backend (FCL, Coal, Polyak, Nesterov).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import Box, Convex, Mesh, Role
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import SpatialInertia
from pydrake.solvers import IpoptSolver
from pydrake.systems.framework import DiagramBuilder

from manipulation.scenarios import AddIiwa, AddWsg

METHODS = {
    "fcl": "off",
    "coal_default": "default",
    "coal_polyak": "polyak",
    "coal_nesterov": "nesterov",
}
REFERENCE_MODE = "default"
OBSTACLES = ("cube", "sphere_hull")

CONFIG_PATH = Path("configs/default.yaml")
OUT_ROOT = Path("results/ik_collision")
SEED = 42
N_SCENARIOS = 140
ORIENTATION_TOL = 1.0e-3
N_REPEATS = 5

GAP = 1.0e-3
GAP_TOL = 1.0e-12
START_OFFSET = 0.4
MAX_SLIDES = 50
CUBE_SIZE = 0.1
SPHERE_RADIUS = 0.05
SPHERE_N_LAT = 32


def sphere_hull_points(n_lat, radius):
    points = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    for i in range(1, n_lat + 1):
        theta = np.pi * i / (n_lat + 1)
        for j in range(2 * n_lat):
            phi = np.pi * j / n_lat
            points.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
    return np.array(points).T


def obstacle_shape(obstacle):
    if obstacle == "cube":
        return Box(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    return Convex(sphere_hull_points(SPHERE_N_LAT, SPHERE_RADIUS), "sphere_hull")


def add_mesh_collision(plant, scene_graph, model):
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


def build_scene(time_step, obstacle, obstacle_pose=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    iiwa = AddIiwa(plant, "no_collision")
    wsg = AddWsg(plant, iiwa, welded=True)
    add_mesh_collision(plant, scene_graph, iiwa)

    body = plant.AddRigidBody(
        obstacle, plant.AddModelInstance(obstacle),
        SpatialInertia.SolidBoxWithMass(1.0, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
    )
    plant.RegisterCollisionGeometry(
        body, RigidTransform(), obstacle_shape(obstacle), obstacle, CoulombFriction())
    if obstacle_pose is not None:
        plant.WeldFrames(plant.world_frame(), body.body_frame(), obstacle_pose)

    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    gripper_frame = plant.GetFrameByName("body", wsg)
    q0 = plant.GetPositions(plant_context, iiwa)
    return plant, plant_context, gripper_frame, q0, body


def closest_arm_pair(plant, plant_context, body):
    gid = plant.GetCollisionGeometriesForBody(body)[0]
    query = plant.get_geometry_query_input_port().Eval(plant_context)
    pairs = [p for p in query.ComputeSignedDistancePairwiseClosestPoints(10.0)
             if gid in (p.id_A, p.id_B)]
    return min(pairs, key=lambda p: p.distance), gid


def place_obstacle(plant, plant_context, q0, body, rng):
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa7"), q0)
    arm_links = [plant.GetBodyByName(f"iiwa_link_{i}") for i in range(8)]
    anchor = plant.EvalBodyPoseInWorld(
        plant_context, arm_links[rng.integers(len(arm_links))]).translation()
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    v = rng.normal(size=4)
    v /= np.linalg.norm(v)
    X = RigidTransform(RotationMatrix(Quaternion(v)), anchor + START_OFFSET * u)

    for _ in range(MAX_SLIDES):
        plant.SetFreeBodyPose(plant_context, body, X)
        pair, gid = closest_arm_pair(plant, plant_context, body)
        err = pair.distance - GAP
        if abs(err) < GAP_TOL:
            return X
        n_toward_arm = pair.nhat_BA_W if pair.id_B == gid else -pair.nhat_BA_W
        X = RigidTransform(X.rotation(), X.translation() + err * n_toward_arm)
    raise RuntimeError(f"{body.name()} did not reach gap {GAP}: {pair.distance}")


def run_place(out_dir, cfg):
    for k, obstacle in enumerate(OBSTACLES):
        plant, plant_context, _, q0, body = build_scene(cfg["time_step"], obstacle)
        rng = np.random.default_rng([SEED, k])
        poses = [place_obstacle(plant, plant_context, q0, body, rng).GetAsMatrix4()
                 for _ in range(N_SCENARIOS)]
        np.save(out_dir / f"placements_{obstacle}.npy", np.array(poses))


def max_violation(prog, x):
    worst = 0.0
    for b in prog.GetAllConstraints():
        ev = b.evaluator()
        v = np.atleast_1d(prog.EvalBinding(b, x))
        lo = np.atleast_1d(ev.lower_bound())
        up = np.atleast_1d(ev.upper_bound())
        worst = max(worst, float(np.max(np.maximum(lo - v, v - up))))
    return max(worst, 0.0)


def min_distance(plant, plant_context, q, max_distance):
    plant.SetPositions(plant_context, q)
    query = plant.get_geometry_query_input_port().Eval(plant_context)
    pairs = query.ComputeSignedDistancePairwiseClosestPoints(max_distance)
    return min((p.distance for p in pairs), default=np.inf)


def timed_solve(solver, prog, q0):
    result = solver.Solve(prog, q0, None)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        solver.Solve(prog, q0, None)
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def run_solve(method, out_dir, cfg):
    import pydrake

    from ik_benchmark.scenarios import generate_scenarios

    md = cfg["minimum_distance"]
    bound = md["min_distance_bound"]
    influence = md["influence_distance_offset"]
    feas_threshold = cfg["compare"]["feasibility_threshold"]
    solver = IpoptSolver()

    for obstacle in OBSTACLES:
        poses = np.load(out_dir / f"placements_{obstacle}.npy")
        plant, plant_context, gripper_frame, q0, _ = build_scene(
            cfg["time_step"], obstacle, RigidTransform(poses[0]))
        scenarios = generate_scenarios(
            plant, plant_context, gripper_frame, q0,
            n=N_SCENARIOS, base_seed=SEED,
            method_mix={"joint_space": 1.0, "task_space": 0.0},
            task_space_params=cfg["task_space"],
        )
        rows = [solve_scenario(method, obstacle, RigidTransform(poses[i]), scenario,
                               solver, cfg, bound, influence, feas_threshold, pydrake.__file__)
                for i, scenario in enumerate(scenarios)]
        pd.DataFrame(rows).to_csv(out_dir / f"{method}_{obstacle}.csv", index=False)


def solve_scenario(method, obstacle, pose, scenario, solver, cfg,
                   bound, influence, feas_threshold, pydrake_path):
    from ik_benchmark.problem import build_ik_prog

    plant, plant_context, gripper_frame, q0, body = build_scene(
        cfg["time_step"], obstacle, pose)
    plant.SetPositions(plant_context, q0)
    gap_at_q0 = closest_arm_pair(plant, plant_context, body)[0].distance

    prog, _ = build_ik_prog(
        plant, plant_context, gripper_frame, scenario,
        orientation_tol=ORIENTATION_TOL,
        use_minimum_distance=True,
        min_distance_bound=bound,
        influence_distance_offset=influence,
    )
    result, t = timed_solve(solver, prog, scenario.q0)
    q = result.GetSolution()
    finite = bool(np.all(np.isfinite(q)))
    violation = max_violation(prog, q) if finite else np.nan
    print(f"{method} {obstacle} {scenario.id}: {t * 1e3:.1f} ms", flush=True)

    return {
        "method": method,
        "obstacle": obstacle,
        "scenario_id": scenario.id,
        "success": result.is_success(),
        "feasible": finite and violation < feas_threshold,
        "max_violation": violation,
        "solve_time_s": t,
        "cost": result.get_optimal_cost(),
        "min_distance": min_distance(plant, plant_context, q, influence) if finite else np.nan,
        "gap_at_q0": gap_at_q0,
        **{f"q{j}": v for j, v in enumerate(q)},
        "pydrake": pydrake_path,
        "coal_mode": os.environ.get("DRAKE_COAL_MODE", ""),
    }


def run_child(args, mode):
    env = {**os.environ, "DRAKE_COAL_MODE": mode}
    subprocess.run([sys.executable, __file__, *args], env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["place", "solve"])
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    if args.stage == "place":
        run_place(args.out, cfg)
        return
    if args.stage == "solve":
        run_solve(args.method, args.out, cfg)
        return

    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_child(["--stage", "place", "--out", str(out_dir)], REFERENCE_MODE)
    for method, mode in METHODS.items():
        run_child(["--stage", "solve", "--method", method, "--out", str(out_dir)], mode)

    df = pd.concat(pd.read_csv(out_dir / f"{m}_{o}.csv") for m in METHODS for o in OBSTACLES)
    df.to_csv(out_dir / "all.csv", index=False)
    print(df.groupby(["obstacle", "method"]).agg(
        feasible=("feasible", "sum"),
        median_ms=("solve_time_s", lambda s: 1e3 * s.median()),
    ))
    paired = df.pivot_table(index=["obstacle", "scenario_id"], columns="method",
                            values="feasible", aggfunc="first").all(axis=1)
    print("feasible under all four methods:")
    print(paired.groupby(level="obstacle").sum())


if __name__ == "__main__":
    main()