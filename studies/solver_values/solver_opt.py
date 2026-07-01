"""native optimality extraction from solver print files.

For SNOPT/IPOPT: make the solver write its optimality line to a temp file and
parse it back. 
"""

import os
import re
import tempfile

from pydrake.solvers import CommonSolverOption, SolverOptions

# regex for getting from print files
_SNOPT_DUAL_RE = re.compile(r"Max Dual infeas\s+\d+\s+([0-9.eE+\-]+)")
_IPOPT_DUAL_RE = re.compile(
    r"Dual infeasibility\.+:\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)"
)


def native_optimality_options(solver_name, base_options=None):
    """Return (SolverOptions, print_file_path) that make the solver write its
    native optimality line to a file WITHOUT flooding the console.
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