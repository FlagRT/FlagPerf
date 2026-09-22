#!/bin/bash
# Day 5 transfer qualification: 8 groups x 5 repeats on card 5.
# Invoked inside tmux session "day5" so sudo tty tickets stay refreshable.
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf || exit 1
sudo -v || { echo SUDO-FAIL; exit 1; }
echo SUDO-OK
for direction in h2d d2h; do
  for variant in pageable-blocking pageable-nonblocking pinned-blocking pinned-nonblocking; do
    for i in 01 02 03 04 05; do
      python3 base/result/p800-pr4-pr5-20260922-0935/run_day5.py \
        qualification "qualification-${direction}-${variant}-q${i}" 5 \
        "interconnect-${direction}" \
        "base/benchmarks/interconnect-${direction}/kunlunxin/P800/case_config.measured-${variant}.yaml" \
        || { echo "GROUP-FAIL-${direction}-${variant}"; break; }
    done
    echo "GROUP-DONE-${direction}-${variant}"
  done
done
echo ALL-QUALIFICATION-DONE
