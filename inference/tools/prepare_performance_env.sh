#!/usr/bin/env bash
# Isolated environment inside an existing CUDA-capable container.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PATH=${1:-"$ROOT/runtime_assets/a100-performance-venv"}
if [[ -e "$ENV_PATH" ]]; then
  echo "Refusing to overwrite: $ENV_PATH" >&2
  exit 2
fi
python3 -c 'import sys; sys.version_info[:2] == (3, 12) or sys.exit("A100 environment preparation requires Python 3.12")'
python3 -m venv "$ENV_PATH"
"$ENV_PATH/bin/python" -m pip --isolated install --no-cache-dir -r "$ROOT/tools/requirements-performance.lock.txt" --extra-index-url https://download.pytorch.org/whl/cu126
"$ENV_PATH/bin/python" -m pip check
"$ENV_PATH/bin/python" -m pip freeze > "$ENV_PATH/packages.lock.txt"
echo "Use $ENV_PATH/bin/python to run run_inference.py performance-baseline"
