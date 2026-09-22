#!/bin/bash
# Day 5 regression rerun: corrected command shapes for the cases whose first
# attempt used wrong invocation (script error, not product regression), plus
# explicit expected-rejection assertions for documented fail-closed gates.
set -u
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf || exit 1
VENV=base/result/p800-pr1-20260920-162724/cpu-test-venv/bin/python
OUT=base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day5/final-regression
mkdir -p "$OUT"
FAILED=0

# normal expectation: exit 0
run() {
  name="$1"; shift
  "$@" >"$OUT/run2-$name.stdout.log" 2>"$OUT/run2-$name.stderr.log"
  code=$?
  echo "run2-$name exit=$code"
  [ $code -ne 0 ] && { echo "  UNEXPECTED NONZERO"; FAILED=1; }
  return 0
}

# expected rejection: exit 2 AND stderr contains the marker
run_expect_reject() {
  name="$1"; marker="$2"; shift 2
  "$@" >"$OUT/run2-$name.stdout.log" 2>"$OUT/run2-$name.stderr.log"
  code=$?
  if [ $code -eq 2 ] && grep -qF "$marker" "$OUT/run2-$name.stderr.log"; then
    echo "run2-$name exit=2 expected-reject OK"
  else
    echo "run2-$name FAILED (exit=$code, marker '$marker' missing)"
    FAILED=1
  fi
  return 0
}

# 1. targeted computation tests via discover (bare-name sibling imports)
run targeted-computation "$VENV" -B -m unittest discover -s base/tests -p 'test_kunlunxin_computation_cases.py' -v

# 2. P2P/MPI intraserver Ascend dry-run with the dedicated flagcx profile (Day 4 shape)
run dry-interconnect-P2P-intraserver python3 -B base/run.py benchmark run --case interconnect-P2P_intraserver \
    --dry-run --config base/configs/ascend910_cann9_p2p.yaml --device-ids 14,15 --nproc-per-node 2
run dry-interconnect-MPI-intraserver python3 -B base/run.py benchmark run --case interconnect-MPI_intraserver \
    --dry-run --config base/configs/ascend910_cann9_p2p.yaml --device-ids 14,15 --nproc-per-node 2

# 3. Day-4-shape Ascend selector matrix (baseline comparability, expect exit 0)
run dry-selector-ascend-npu-ids-on python3 -B base/run.py benchmark run --case computation-FP32 --npu-ids 7 --monitor on --dry-run
run dry-selector-ascend-npu-ids-off python3 -B base/run.py benchmark run --case computation-FP32 --npu-ids 7 --monitor off --dry-run
run dry-selector-ascend-physical-on python3 -B base/run.py benchmark run --case computation-FP32 --physical-device-ids 6,2 --monitor on --dry-run
run dry-selector-ascend-device-ids-on python3 -B base/run.py benchmark run --case computation-FP32 --device-ids 14 --monitor on --dry-run
run dry-selector-ascend-device-ids-off python3 -B base/run.py benchmark run --case computation-FP32 --device-ids 14 --monitor off --dry-run

# 4. P800 explicit selection with required timeout (expect exit 0)
run dry-selector-p800-physical-on python3 -B base/run.py benchmark run --case computation-FP32:P800 --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --physical-device-ids 5 --nproc-per-node 1 --timeout 300

# 5. Documented fail-closed gates (expected rejection with marker)
run_expect_reject dry-main-memory-capacity 'high risk' python3 -B base/run.py benchmark run --case main_memory-capacity \
    --dry-run --config base/configs/ascend910_cann9_local.yaml --device-ids 14 --nproc-per-node 1
run_expect_reject dry-capacity-denied 'high risk' python3 -B base/run.py benchmark run --case main_memory-capacity --dry-run \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --device-ids 2 --nproc-per-node 1
run_expect_reject dry-p800-selector-npu-ids 'legacy aliases are undefined' python3 -B base/run.py benchmark run \
    --case computation-FP32:P800 --dry-run --config base/configs/kunlunxin_p800_xpytorch29.yaml --npu-ids 3 --nproc-per-node 1
run_expect_reject dry-p800-selector-device-ids 'legacy aliases are undefined' python3 -B base/run.py benchmark run \
    --case computation-FP32:P800 --dry-run --config base/configs/kunlunxin_p800_xpytorch29.yaml --device-ids 2 --nproc-per-node 1
run_expect_reject dry-p800-preflight-no-selection 'one of the arguments' python3 -B base/run.py benchmark preflight \
    --config base/configs/kunlunxin_p800_xpytorch29.yaml --dry-run

echo "RERUN-FINISHED failed=$FAILED"
