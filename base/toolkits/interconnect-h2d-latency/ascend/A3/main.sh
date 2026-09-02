# Copyright 2026 FlagOS Contributors
set -euo pipefail
[[ ! -f /usr/local/Ascend/toolbox/set_env.sh ]] || source /usr/local/Ascend/toolbox/set_env.sh
[[ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]] || source /usr/local/Ascend/ascend-toolkit/set_env.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/../../../_common/ascend/A3/evidence_runner.py" \
    --cases "interconnect-h2d-latency" "$@"
