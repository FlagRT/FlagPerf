# Day 9 exploratory communication evidence

Date: 2026-09-28

This directory records the first post-Day-8 exploratory communication runs. The selected pair was physical cards 4 and 6. Card 4 became occupied between inventory and execution; the run therefore used the explicit `P800_ALLOW_FOREIGN_HANDLES=1` authorization and is exploratory evidence only. No candidate validation status was changed.

## Results

| Run | Status | Result |
|---|---|---|
| `mpi-2-exploratory-4-6-wrapper` | partial | execution, correctness, measurement, cleanup and lease release passed; monitor partial; rank algbw 9.9227/9.6929 GB/s |
| `p2p-2-exploratory-4-6-wrapper` | partial | execution, correctness, measurement, cleanup and lease release passed; monitor partial; one-way bandwidth 6.3367/5.6730 GB/s |
| `mpi-8-exploratory-0-7-wrapper` | failed | host preflight rejected open handles; no container or collective qualification ran |

The two-card records demonstrate that the expanded two-rank path completes on the selected pair. They do not close the formal idle-card gate because the pair was shared and monitoring was partial. The eight-card record preserves the current fail-closed behavior when a selected device has unverified handles.

## Reproduction notes

The host shell used a temporary sudo wrapper because the account requires a password for each noninteractive sudo invocation. The wrapper and password were not copied into this evidence directory. The runtime image remained candidate and `validated:false`.

The post-Day-8 static dry-run sweep covered all 10 registered Kunlunxin/P800 Base cases. After fixing a provider import-scope defect in `validate_benchmark`, all 10 dry-runs passed, including the explicitly authorized high-risk capacity case. This is a planning/configuration gate only and does not imply execution qualification.
