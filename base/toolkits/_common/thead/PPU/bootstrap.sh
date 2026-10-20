#!/usr/bin/env bash
set -euo pipefail
# The image ENTRYPOINT (/opt/t-head/entrypoint.sh) sources the SDK environment
# and then execs a shell, which swallows stdin and pollutes stdout with its
# banner. Sourcing envsetup here keeps every evidence_runner child pipe
# machine-parseable while providing the same SDK environment.
set +u
source /usr/local/PPU_SDK/envsetup.sh
set -u
# Evidence files stay group/world writable so the non-privileged host executor
# can enrich manifest and metrics after the container exits.
umask 0000
exec python3 /workspace/FlagPerf/base/toolkits/_common/thead/PPU/evidence_runner.py \
    --context /run/flagperf/host-context.json \
    --output /workspace/FlagPerf/results/toolkit-evidence
