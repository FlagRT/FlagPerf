#!/usr/bin/env bash
# PPU bounded identity bootstrap; PR0's exec-based entrypoint remains unchanged.
# The image bakes the CUDA-compat environment (CUDA_PATH/LD_LIBRARY_PATH) into
# its config and envsetup.sh only re-exports it, so sourcing is idempotent and
# keeps the probe identical to the day-1/probe-B invocation. CUDA_VISIBLE_DEVICES
# is re-exported after sourcing so no vendor script can clobber the selection.
set -euo pipefail
VENDOR_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /usr/local/PPU_SDK/envsetup.sh >/dev/null 2>&1
unset XPU_VISIBLE_DEVICES XPU_EVENT_KL3_ENABLE
export CUDA_VISIBLE_DEVICES="$FLAGPERF_CVD"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -S "$VENDOR_ROOT/container_probe.py" "$@"
