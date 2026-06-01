"""
Result records and aggregation.

- defines the Result record (one row per scenario + solver run)
- turns a list of them into a dataframe
- saves output as Parquet alongside a metadata sidecar
    - (resolved config, Drake version, git commit, host, timestamp) 
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    scenario_id: int
    generation_method: str
    solver_name: str
    success_flag: bool                # solver's own verdict — recorded, not authoritative
    is_feasible: bool                 # from solver-blind measurement
    max_violation: float
    objective_value: float
    solve_time_mean_s: float
    solve_time_std_s: float


def aggregate(results):
    """Turn a list[Result] into a pandas DataFrame (one row per result)."""
    raise NotImplementedError


def save(df, out_dir, metadata):
    """Write df to Parquet and metadata to a sidecar JSON in out_dir."""
    raise NotImplementedError
