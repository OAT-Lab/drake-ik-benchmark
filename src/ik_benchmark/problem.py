"""
Build a Drake MathematicalProgram from a Scenario.

The single place that encodes a Scenario into Drake's InverseKinematics
(position constraint, orientation constraint, mindistlowerbound, quadratic error cost). Kept
separate from scenario generation so the abstract problem description and its
Drake encoding can change independently.
"""