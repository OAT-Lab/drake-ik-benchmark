"""
The thin CLI conductor. 
- Contains no benchmark logic: 
    - load config
    - build plant
    - generate scenarios
    - loop over each (scenario + solver) pair through the single runner path
    - collect results
    - save
Orchestration only — all real logic lives in the src package.
"""


from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the Drake IK solver benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()

    # cfg = load_config(args.config)
    # plant, plant_context, gripper_frame = build_plant()
    # scenarios = generate_scenarios(plant, cfg.n, cfg.base_seed, cfg.method_mix, cfg.task_space)
    # results = []
    # for scenario in scenarios:
    #     prog, q = build_ik_prog(plant, plant_context, gripper_frame, scenario, cfg.orientation_tol)
    #     solvers = get_candidate_solvers(prog)
    #     options = build_shared_options(cfg.tolerances)
    #     for solver in solvers:
    #         results.append(run_one(prog, q, solver, scenario.q0, options,
    #                                cfg.n_repeats, cfg.n_warmup, cfg.feasibility_threshold))
    # df = aggregate(results)
    # save(df, args.out, metadata=collect_metadata(cfg))
    raise NotImplementedError


if __name__ == "__main__":
    main()