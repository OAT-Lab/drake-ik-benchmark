"""
Solver enumeration and option pinning.

Returns potential solvers: 
    - available (present and licensed in this Drake build) 
    - capable (able to represent the prog's non-convex nonlinear constraints). 

The set is enumerated from Drake rather than hardcoded. 
Also builds one shared SolverOptions for same tolerances.
"""

from __future__ import annotations

import os
import re
import tempfile

from pydrake.solvers import (
    CommonSolverOption,
    GetAvailableSolvers,
    GetProgramType,
    MakeSolver,
    NloptSolver,
    SolverOptions,
)

# regex for getting from print files
_SNOPT_DUAL_RE = re.compile(r"Max Dual infeas\s+\d+\s+([0-9.eE+\-]+)")
_IPOPT_DUAL_RE = re.compile(
    r"Dual infeasibility\.+:\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)"
)


def get_candidate_solvers(prog):
    """Return solver instances that are available and capable

    Returns:
        list[SolverInterface]: list of solvers
    """
    solver_ids = GetAvailableSolvers(GetProgramType(prog))
    return [MakeSolver(solver_id) for solver_id in solver_ids]


def build_shared_options(solver_id, solver_name, cfg):
    """Pinned shared tolerances + NLopt caps. Fresh object per call."""
    opts = SolverOptions()
    tol = cfg["tolerances"]     # from config.yaml
    feas, optm, iters = tol["feasibility_tol"], tol["optimality_tol"], tol["major_iter_limit"]

    if solver_name == "SNOPT":
        opts.SetOption(solver_id, "Major feasibility tolerance", feas)
        opts.SetOption(solver_id, "Major optimality tolerance", optm)
        opts.SetOption(solver_id, "Major iterations limit", iters)
    elif solver_name in ("Ipopt", "IPOPT"):
        opts.SetOption(solver_id, "constr_viol_tol", feas)
        opts.SetOption(solver_id, "tol", optm)
        opts.SetOption(solver_id, "max_iter", iters)
        opts.SetOption(solver_id, "acceptable_iter", 0)     # no acceptable soln, only desired
    elif solver_name == "NLopt":
        n = cfg["nlopt"]
        opts.SetOption(solver_id, NloptSolver.AlgorithmName(), "LD_SLSQP")      # default for nlopt but safety net
        opts.SetOption(solver_id, NloptSolver.ConstraintToleranceName(), feas)
        opts.SetOption(solver_id, NloptSolver.XRelativeToleranceName(), 1e-8)
        opts.SetOption(solver_id, NloptSolver.MaxEvalName(), n["max_eval"])
        opts.SetOption(solver_id, NloptSolver.MaxTimeName(), n["max_time_sec"])
    return opts


def quiet_options(solver_name, base_options=None):
    """Return SolverOptions with every print path disabled.

    Used by the timed pass. Print files take longer.
    """
    opts = base_options or SolverOptions()

    if solver_name in ("Ipopt", "IPOPT"):
        sid = _ipopt_id()
        opts.SetOption(sid, "print_level", 0)  # console off (default is 5!)
        opts.SetOption(sid, "sb", "yes")       # suppress the startup banner
    return opts


def native_optimality_options(solver_name, base_options=None):
    """Return (SolverOptions, print_file_path) that make the solver write its
    native optimality line to a file.
    """
    if solver_name == "NLopt":
        return (base_options or SolverOptions()), None

    opts = base_options or SolverOptions()
    fd, path = tempfile.mkstemp(suffix=".out")
    os.close(fd)

    if solver_name == "SNOPT":
        opts.SetOption(CommonSolverOption.kPrintFileName, path)
    elif solver_name in ("Ipopt", "IPOPT"):
        sid = _ipopt_id()
        opts.SetOption(sid, "output_file", path)     # ipopt-native file
        opts.SetOption(sid, "file_print_level", 5)
        opts.SetOption(sid, "print_level", 0)        # console off
        opts.SetOption(sid, "sb", "yes")             # suppress the startup banner
    return opts, path


def parse_native_optimality(solver_name, print_file_path):
    """Parse the native optimality number from a solved print file.

    Returns a float or none
    """
    if print_file_path is None or not os.path.exists(print_file_path):
        return None
    text = _read_and_cleanup(print_file_path)

    if solver_name == "SNOPT":
        m = _SNOPT_DUAL_RE.search(text)
        return float(m.group(1)) if m else None
    if solver_name in ("Ipopt", "IPOPT"):
        m = _IPOPT_DUAL_RE.search(text)
        return float(m.group(2)) if m else None  # group(2) = unscaled
    return None


def _read_and_cleanup(path):
    with open(path, errors="replace") as f:
        text = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return text


def _ipopt_id():
    from pydrake.solvers import IpoptSolver
    return IpoptSolver().solver_id()
