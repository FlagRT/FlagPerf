#!/usr/bin/env bash
# Copyright (c) 2026, FlagPerf contributors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
set -euo pipefail
PROFILE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNTIME_PREFIX=$(python3 -S -c 'import json,sys; print(json.load(open(sys.argv[1]))["conda_prefix"])' "$PROFILE_ROOT/stack.lock.yaml")
test -x "$RUNTIME_PREFIX/bin/python"
test -x "$RUNTIME_PREFIX/bin/torchrun"
source "$(dirname -- "$(dirname -- "$RUNTIME_PREFIX")")/bin/activate" "$RUNTIME_PREFIX"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
# -S bypasses executable .pth/sitecustomize in the no-device audit.
if [[ "${1:-}" == "--static" ]]; then
    exec "$RUNTIME_PREFIX/bin/python" -S "$PROFILE_ROOT/verify_runtime.py" "$@"
fi
unset XPU_EVENT_KL3_ENABLE XPU_VISIBLE_DEVICES
exec "$RUNTIME_PREFIX/bin/python" -S "$PROFILE_ROOT/hardware_probe.py" "$@"
