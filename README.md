# drake-ik-benchmark

A controlled benchmark of Drake's nonlinear program (NLP) solvers on the inverse kinematics problem for the KUKA iiwa7 arm with the Schunk WSG gripper (9 DOF: 7 arm joints + 2 finger joints).

## What this measures

For each solver capable of the IK problem, across many generated scenarios:

* **Feasibility** of the returned solution, and the extent of infeasibility when it isn't feasible

* **Computation time**, averaged over repeated solves with warm-up runs discarded

* **Returned objective value**

## Design principles

The benchmark is built so that every (scenario + solver) pair runs under identical conditions:

* **Single code path.** Every run flows through one function (runner.run_one), so no solver or scenario can be treated differently. Uniformity is structural, not a matter of discipline.

* **Solvers are matched.** The candidate solver set is enumerated from Drake by capability and availability rather than hardcoded; shared solver tolerances are pinned from one source so all solvers run with the same effective settings.

* **Solver-blind feasibility.** Feasibility of a returned solution is judged by re-evaluating the program's own constraints against their bounds.

* **Fixed initial guess per scenario.** Each scenario has one initial guess, held identical across all solvers, so differences in outcome are attributable to the solver and not the start.

* **Reproducibility.** All randomness derives from a single seed; a run is fully described by its config file plus the pinned Drake version.

## Scenario generation

Two methods, mixed:

* Joint-space sampling — sample a random configuration, forward-kinematics to a guaranteed-reachable target pose.

* Task-space sampling — sample a pose directly in the workspace, which may be unreachable and exercises each solver's infeasibility handling.

Position, orientation, and minimum distance lower bound constraints are used.

## Solver scope

The realistic candidate set is Drake's NLP solvers (SNOPT, IPOPT, the NLopt family). 

Convex-only solvers cannot represent the non-convex kinematic constraints and are excluded automatically by the capability filter rather than by hand.

## Repository layout

```
drake_ik_benchmark/
├── configs/default.yaml      all tunable knobs: seed, scenario count, tolerances, timing
├── src/ik_benchmark/
│   ├── plant.py              build the iiwa7 + WSG plant (once)
│   ├── scenarios.py          Scenario dataclass + seeded generators
│   ├── problem.py            build an IK MathematicalProgram from a Scenario
│   ├── solvers.py            enumerate capable+available solvers; pin shared options
│   ├── measure.py            solver-blind feasibility measurement
│   ├── runner.py             the single execution code path
│   └── results.py            Result record + aggregation/saving
├── scripts/run_benchmark.py  thin CLI conductor (no logic)
├── results/                  output: Parquet table + run metadata
└── tests/                    reproducibility / control-invariant tests
```

## Limitations

* Timings are valid relative to each other on one machine and configuration; they are not absolute claims about solver speed in general.

* SNOPT, IPOPT, and NLopt are local methods on a non-convex problem; results are conditioned on the fixed initial guess and do not establish global superiority.

* Tolerances are pinned for fairness, so the benchmark measures normalized behavior rather than out-of-the-box defaults.