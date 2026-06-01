"""
Result records and aggregation.

Defines the Result record (one row per scenario x solver run) and turns a list of
them into a tidy DataFrame. Saves output as Parquet alongside a metadata sidecar
(resolved config, Drake version, git commit, host, timestamp) so any results file
is traceable to the conditions that produced it.
"""