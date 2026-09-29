#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/FlagPerf/base
PROFILE="$ROOT/vendors/kunlunxin/xpytorch_2.9_p800_candidate"
PREFIX=$(python3 -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_prefix"])' "$PROFILE/stack.lock.yaml")
unset CUDA_VISIBLE_DEVICES XPU_VISIBLE_DEVICES XPU_EVENT_KL3_ENABLE
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 USE_FLAGGEMS=0
exec "$PREFIX/bin/python" -S "$ROOT/toolkits/_common/kunlunxin/P800/evidence_runner.py" \
    --context /run/flagperf/host-context.json \
    --output /workspace/FlagPerf/results/toolkit-evidence
