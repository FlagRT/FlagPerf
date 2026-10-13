#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
exec python3 "$BASE/run.py" toolkit run --config "$BASE/configs/kunlunxin_p800_xpytorch29.yaml" --case "computation-INT8" "$@"
