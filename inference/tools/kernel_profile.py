# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
"""Operator (device-kernel) profile from a sealed profiler trace.

Aggregates cat='kernel' events by name: call count, total duration, share of
device time, and flags collective kernels (pccl/nccl/xccl prefixes). Prints a
readable top table and writes a JSON next to it.
Usage: kernel_profile.py <trace.json> <out.json>
"""
import json
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def is_comm(name):
    lowered = name.lower()
    return any(t in lowered for t in ('pccl', 'nccl', 'xccl', 'bkcl', 'hccl'))


def main():
    trace_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    data = json.loads(trace_path.read_text())
    events = data['traceEvents'] if isinstance(data, dict) else data
    agg = defaultdict(list)
    for event in events:
        if event.get('cat') == 'kernel' and event.get('dur') is not None:
            duration_ns = int(Decimal(str(event['dur'])) * 1000)
            agg[str(event.get('name'))].append(duration_ns)
    total_calls = sum(len(v) for v in agg.values())
    total_ns = sum(sum(v) for v in agg.values())
    rows = []
    for name, durs in agg.items():
        rows.append(dict(name=name, calls=len(durs), total_ns=round(sum(durs), 1),
                         median_ns=round(statistics.median(durs), 1),
                         share=round(sum(durs) / total_ns, 4) if total_ns else 0,
                         comm=is_comm(name)))
    rows.sort(key=lambda r: r['total_ns'], reverse=True)
    comm_ns = sum(r['total_ns'] for r in rows if r['comm'])
    report = dict(kernels_total=total_calls, kernel_time_ns=round(total_ns, 1),
                  distinct_names=len(rows), comm_kernel_time_ns=round(comm_ns, 1),
                  comm_share=round(comm_ns / total_ns, 4) if total_ns else 0,
                  top=rows[:40])
    out_path.write_text(json.dumps(report, indent=1) + '\n')
    print('kernels=%d distinct=%d kernel_time_ms=%.1f comm_share=%.3f'
          % (total_calls, len(rows), total_ns / 1e6, report['comm_share']))
    print('top kernels by total time:')
    for r in rows[:12]:
        print('  %8.1fms  %6d calls  %s' % (r['total_ns'] / 1e6, r['calls'], r['name'][:90]))


if __name__ == '__main__':
    main()
