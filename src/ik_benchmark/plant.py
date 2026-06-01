"""
Build the iiwa7 + WSG MultibodyPlant.

The plant is constructed once and reused across every scenario; rebuilding per
scenario would be slow and a place for inconsistency to creep in. Only the prog
(constraints) and q0 change per scenario, not the plant.
"""