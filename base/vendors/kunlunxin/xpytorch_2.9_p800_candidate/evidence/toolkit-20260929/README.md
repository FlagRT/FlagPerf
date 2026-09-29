# P800 Toolkit engineering evidence — 2026-09-29

Twelve native Toolkit cases are functionally verified. This is shared-card exploratory evidence;
it does not promote the candidate or qualify the Base two/eight-card gates. Physical card 1 is excluded.

`support-matrix.json` maps each case to actual runs. `verification.json` records raw metric checks,
monitoring, immutable measurement evidence, byte-identical report regeneration and SHA-256 audits.
`repeat-groups.json` retains every repeat, including unstable groups. `qualification-record.json`
declares the exact engineering scope and hashes the delivered sources.

`complete-integration-run/` contains a full compact run with working relative evidence/report links.
`run-records/` contains original-byte metadata snapshots of all runs, including historical failures.
Its copied indexes describe the FULL remote runs, not the subset mirrored in that directory;
do not validate a full-run index against a metadata subset. See `run-index.json` for absolute
remote paths. The bundle root index describes only this bundle and includes nested run indexes.

The 04:21 P2P exploratory run used an early metadata counter reporting one logical work unit
for a bidirectional sample, although the implementation executed two actual peer copies.
The counter was corrected to two native submissions and a new five-repeat group was run.
The earlier raw artifact is retained without rewriting its measurements or status.

Official fc_effciency/perf_regression still need the matching XBLAS unittest. Vendor threshold
diagnosis is not-supported. Pinned 512 MiB async copies use explicit 1 MiB native chunks.
INT8 uses native fc_fusion with FP32 output. These scopes are not equivalent hardware peak claims.
