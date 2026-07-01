"""solver comparison

reads raw record tables and makes comp for each pair

  q_diff_inf, q_diff_2          how far apart the returned joint vectors are
  cost_diff_abs, cost_diff_rel  difference in solver-blind objective
  common_residual_inf_diff      difference in stationarity of each returned point
  both_feasible                 both solves feasible (solver-blind)
  same_q                        q_diff_inf < q_same_tol

Output: pairwise.csv (one row per scenario x solver-pair)
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ik_benchmark.log_config import configure_logging

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _reconstruct_q(joints_df):
    """Map solve_id -> q vector

    Returns {solve_id: q} with joints in ascending index order
    """
    wide = (
        joints_df.pivot(index="solve_id", columns="joint_index", values="q_value")
        .sort_index(axis=1)
    )
    return {sid: wide.loc[sid].to_numpy() for sid in wide.index}


def _q_diffs(qa, qb):
    """(inf-norm, 2-norm) of qa - qb, or (nan, nan) if either is missing/non-finite."""
    if qa is None or qb is None:
        return np.nan, np.nan
    if not (np.all(np.isfinite(qa)) and np.all(np.isfinite(qb))):
        return np.nan, np.nan
    d = qa - qb
    return float(np.max(np.abs(d))), float(np.linalg.norm(d))


def _residual_diff(ra, rb):
    """|ra - rb|, or nan if either residual is missing"""
    if pd.isna(ra) or pd.isna(rb):
        return np.nan
    return abs(float(ra) - float(rb))


def build_pairwise(solves_df, joints_df, q_same_tol):
    """One row per (scenario, unordered solver-pair) with diff/agreement columns."""
    qvec = _reconstruct_q(joints_df)
    rows = []

    for scenario_id, grp in solves_df.groupby("scenario_id"):
        recs = grp.to_dict("records")
        for a, b in combinations(recs, 2):
            # deterministic pair order so (SNOPT, IPOPT) and (IPOPT, SNOPT)
            # don't both appear
            if a["solver_name"] > b["solver_name"]:
                a, b = b, a

            qa = qvec.get(a["solve_id"])
            qb = qvec.get(b["solve_id"])
            q_inf, q_2 = _q_diffs(qa, qb)

            cost_abs = abs(a["evaluated_cost"] - b["evaluated_cost"])
            denom = max(1.0, abs(a["evaluated_cost"]), abs(b["evaluated_cost"]))

            rows.append({
                "scenario_id": scenario_id,
                "seed": a["seed"],
                "orientation_tol": a["orientation_tol"],
                "solver_a": a["solver_name"],
                "solver_b": b["solver_name"],
                "q_diff_inf": q_inf,
                "q_diff_2": q_2,
                "cost_diff_abs": cost_abs,
                "cost_diff_rel": cost_abs / denom,
                "common_residual_inf_diff": _residual_diff(
                    a["common_residual_inf"], b["common_residual_inf"]
                ),
                "both_feasible": bool(a["feasible"] and b["feasible"]),
                "same_q": bool(np.isfinite(q_inf) and q_inf < q_same_tol),
            })

    return pd.DataFrame(rows)


def main():
    cfg = _load_config(CONFIG_PATH)
    log = cfg["logging"]
    configure_logging(
        level=log["level"],
        log_file=RESULTS_DIR / log["file"],
        mode="a",  # append: compare runs after run_study, same log
        quiet=log["quiet"],
    )

    solves = pd.read_csv(RESULTS_DIR / "solves.csv")
    joints = pd.read_csv(RESULTS_DIR / "joints.csv")
    logger.info("loaded %d solves, %d joint rows", len(solves), len(joints))

    pairwise = build_pairwise(solves, joints, cfg["compare"]["q_same_tol"])
    pairwise.to_csv(RESULTS_DIR / "pairwise.csv", index=False)
    logger.info("wrote %d pairwise rows to %s", len(pairwise), RESULTS_DIR / "pairwise.csv")


if __name__ == "__main__":
    main()