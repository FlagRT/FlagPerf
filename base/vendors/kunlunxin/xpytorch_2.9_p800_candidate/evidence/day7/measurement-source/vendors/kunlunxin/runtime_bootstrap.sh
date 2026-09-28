#!/usr/bin/env bash
# P800 bounded identity bootstrap; PR0's exec-based entrypoint remains unchanged.
set -euo pipefail
VENDOR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_PREFIX=$(python3 -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_prefix"])' "$VENDOR_ROOT/xpytorch_2.9_p800_candidate/stack.lock.yaml")
test -x "$RUNTIME_PREFIX/bin/python"
test -x "$RUNTIME_PREFIX/bin/torchrun"
source "$(dirname -- "$(dirname -- "$RUNTIME_PREFIX")")/bin/activate" "$RUNTIME_PREFIX"
unset XPU_EVENT_KL3_ENABLE XPU_VISIBLE_DEVICES
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 USE_FLAGGEMS=0
exec "$RUNTIME_PREFIX/bin/python" -S "$VENDOR_ROOT/container_probe.py" "$@"
