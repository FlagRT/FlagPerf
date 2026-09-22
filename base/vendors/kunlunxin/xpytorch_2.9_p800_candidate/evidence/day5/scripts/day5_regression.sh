#!/bin/bash
# Day 5 offline regression: targeted tests -> Base full -> PR0 -> Toolkit
# -> 15-case static dry-run matrix -> selector/runtime gates -> whitespace.
set -u
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf || exit 1
VENV=base/result/p800-pr1-20260920-162724/cpu-test-venv/bin/python
OUT=base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day5/final-regression
mkdir -p "$OUT"
FAILED=0

run() {
  name="$1"; shift
  echo "=== $name ==="
  "$@" >"$OUT/$name.stdout.log" 2>"$OUT/$name.stderr.log"
  code=$?
  python3 - "$OUT/$name.command.json" "$code" "$name" <<'PY'
import json, sys
out, code, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
json.dump({'name': name, 'exit_code': code}, open(out, 'w'), indent=1)
PY
  echo "$name exit=$code"
  [ $code -ne 0 ] && FAILED=1
  return 0
}

# Targeted new-case tests first
run targeted-computation "$VENV" -B -m unittest base.tests.test_kunlunxin_computation_cases -v
run targeted-transfer "$VENV" -B -m unittest base.tests.test_kunlunxin_transfer_benchmarks -v
run targeted-fp32 "$VENV" -B -m unittest base.tests.test_kunlunxin_fp32 -v
run targeted-preflight "$VENV" -B -m unittest base.tests.test_p800_preflight -v

# Full suites
run base "$VENV" -B -m unittest discover -s base/tests -p 'test_*.py' -v
run pr0 python3 -S -B -m unittest discover -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py' -v
run toolkit python3 -B -m unittest discover -s base/toolkits/_common/ascend/A3/tests -v

# Static dry-run planning matrix (Ascend config; no hardware)
for case in computation-FP32 computation-FP16 computation-BF16 computation-INT8 \
            computation-FP64 computation-FP8 computation-TF32 \
            interconnect-h2d interconnect-d2h \
            interconnect-P2P_intraserver interconnect-MPI_intraserver \
            interconnect-P2P_interserver interconnect-MPI_interserver \
            main_memory-bandwidth main_memory-capacity; do
  run "dry-${case//_/-}" python3 -B base/run.py benchmark run --case "$case" --dry-run \
      --config base/configs/ascend910_cann9_local.yaml --device-ids 14 --nproc-per-node 1
done

# Selector and runtime gates
run dry-selector-npu-ids-on python3 -B base/run.py benchmark run --case computation-FP32 --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --npu-ids 3 --nproc-per-node 1
run dry-selector-physical-device-ids-on python3 -B base/run.py benchmark run --case computation-FP32:P800 --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --physical-device-ids 6 --nproc-per-node 1
run dry-selector-device-ids-on python3 -B base/run.py benchmark run --case computation-FP32:P800 --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --device-ids 2 --nproc-per-node 1
run dry-capacity-denied python3 -B base/run.py benchmark run --case main_memory-capacity:P800 --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --physical-device-ids 6 --nproc-per-node 1
run dry-p800-on python3 -B base/run.py benchmark preflight --config base/configs/kunlunxin_p800_xpytorch29.yaml \
    --physical-device-ids 6 --dry-run
run dry-p800-off python3 -B base/run.py benchmark preflight --config base/configs/kunlunxin_p800_xpytorch29.yaml --dry-run

# Shell syntax and whitespace hygiene
run bootstrap bash -n base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh
run whitespace bash -c 'git diff --check'

echo "REGRESSION-FINISHED failed=$FAILED"
