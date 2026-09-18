# Portable mapping verification — 2026-09-18

Tested executable code: `f632019c` (clean tree at both successful launches).
32 offline tests passed. Host/card indices and device minors are no longer
restricted to eight cards or one-digit suffixes. UUID joins tolerate reordered
framework enumeration and reject missing/extra/duplicate identities.

## Hardware results

| Attempt | Result |
|---|---|
| `p800-mapping-20260918-cards6-2-attempt01` | Passed two-device mapping and independent 2×2 FP32 on both devices |
| `p800-mapping-20260918-card2-regression01` | Rejected open handles in preflight; no container started |
| `p800-mapping-20260918-card2-regression02` | Passed all six API stages plus native route control |

The requested physical order `[6, 2]` resolved to host nodes
`[/dev/xpu5, /dev/xpu3]` and framework logical IDs `[1, 0]`. Both tiny matrix
results were `[[19,22],[43,50]]`. Thus request order is preserved independently
of framework enumeration. Each future rank must bind the logical ID returned
for its requested position; local rank is not automatically the logical ID.

Single-card regression again returned maximum absolute FP32 error
`6.6186313503191485e-06` with native launch counts 4 (control) and 8 (four matmuls).
Both successful containers exited zero and were removed. Raw selected-card
preflight/postflight records show zero memory usage and zero utilization.
The first rejected single-card attempt remains in the raw results directory.

Portable artifacts are in `multi/` and `single/`; raw logs are under
`base/result/` with the attempt names above. These checks cover identity and
independent device computation, not simultaneous multi-process ranks, peer copy,
collectives or communication performance. Those remain later case tests.

## User decisions

The user accepts the current driver/M1 version combination for development,
confirms recipients can obtain the same image, and confirms observing execution
on card 2. These three items no longer block Day 1 completion. No vendor
compatibility statement, registry pull or named kernel trace is fabricated.
The candidate flag concerns the still-unimplemented formal Base benchmark
scope, not an outstanding request for these confirmations.

Mapping checks remain mandatory on each recipient's machine. Historical UUIDs,
card indices and PCI addresses in evidence are observations, not configuration
defaults. The image lock and vendor protocol are intentionally explicit.
