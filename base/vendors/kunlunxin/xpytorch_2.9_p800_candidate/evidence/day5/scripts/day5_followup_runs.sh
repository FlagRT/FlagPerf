#!/bin/bash
# Day 5 follow-up device runs after transfer qualification:
#   1. one monitored FP32 regression round with the frozen day-4 config
#   2. controlled-failure cleanup recheck for a computation case (FP16 cpu-wait)
#   3. controlled-failure cleanup recheck for a transfer case (h2d cpu-wait)
# Invoked inside tmux session "day5fu" so sudo tty tickets stay refreshable.
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf || exit 1
sudo -v || { echo SUDO-FAIL; exit 1; }
echo SUDO-OK
python3 base/result/p800-pr4-pr5-20260922-0935/run_day5.py \
  measured measured-FP32-a01 5 computation-FP32 \
  && echo RUN-DONE-measured-FP32 || echo RUN-FAIL-measured-FP32
python3 base/result/p800-pr4-pr5-20260922-0935/run_day5.py \
  timeout timeout-FP16-a01 5 computation-FP16 \
  && echo RUN-DONE-timeout-FP16 || echo RUN-EXPECTED-FAIL-timeout-FP16
python3 base/result/p800-pr4-pr5-20260922-0935/run_day5.py \
  timeout timeout-h2d-a01 5 interconnect-h2d \
  && echo RUN-DONE-timeout-h2d || echo RUN-EXPECTED-FAIL-timeout-h2d
echo ALL-FOLLOWUP-DONE
