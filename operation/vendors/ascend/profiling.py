# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CANN msprof subprocess capture. Raw traces stay separate from clean timing."""
import csv
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from runtime.evidence import read, write, sha
from runtime.progress import publish_profile

GROUPS = ('ArithmeticUtilization', 'PipeUtilization', 'Memory', 'MemoryL0', 'MemoryUB', 'ResourceConflictRatio')
MARKER = 'operation_target'


def target_window(adapter, root, task, call):
    import mstx
    for _ in range(task['case_config']['WARMUP']):
        out = call()
        del out
    adapter.synchronize()
    for index in range(5):
        handle = mstx.range_start(f'{MARKER}_{index}')
        try:
            out = call()
            adapter.synchronize()
            del out
        finally:
            mstx.range_end(handle)
    write(root / 'profile-window.json', {'calls': 5, 'marker': MARKER,
          'boundary': 'isolated one-device process; synchronized MSTX ranges after warmup; original measurement invocation'})


def union_duration(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def analyze_events(events):
    """Only explicit device categories/process metadata qualify, never op names."""
    origins = [Decimal(str(e['ts'])) for e in events if e.get('ph') == 'X' and str(e.get('name', '')).startswith(MARKER + '_')]
    if not origins or any(not x.is_finite() for x in origins):
        return {'status': 'partial', 'reason': 'missing or invalid target timestamps'}
    origin = min(origins)
    def timestamp(event):
        return float(Decimal(str(event['ts'])) - origin)
    processes = {e.get('pid'): str(e.get('args', {}).get('name', '')).lower()
                 for e in events if e.get('ph') == 'M' and e.get('name') == 'process_name'}
    ranges = []
    for e in events:
        if e.get('ph') == 'X' and str(e.get('name', '')).startswith(MARKER + '_'):
            start, duration = timestamp(e), float(e['dur'])
            if math.isfinite(start) and math.isfinite(duration) and duration > 0:
                ranges.append((start, start + duration))
    if len(ranges) != 5:
        return {'status': 'partial', 'reason': 'requires five complete synchronized target MSTX ranges', 'kernel_count': None}
    sources = {str(e.get('id')): e for e in events if e.get('cat') == 'HostToDevice' and e.get('ph') == 's'}
    finishes = {}
    for e in events:
        if e.get('cat') == 'HostToDevice' and e.get('ph') == 'f':
            finishes.setdefault((e.get('pid'), e.get('tid'), timestamp(e)), []).append(e)
    kernels = []
    for e in events:
        if e.get('ph') != 'X': continue
        category = str(e.get('cat', '')).lower()
        process = processes.get(e.get('pid'), '')
        task_type = str(e.get('args', {}).get('Task Type', '')).upper()
        hardware = category in ('kernel', 'aicore', 'ai_core', 'aivector', 'ai_vector', 'aicpu', 'ai_cpu') or ('ascend hardware' in process and task_type in ('AI_CORE', 'AI_VECTOR_CORE', 'AI_CPU', 'AICPU', 'MIX_AIC', 'MIX_AIV'))
        if not hardware: continue
        start, duration = timestamp(e), float(e.get('dur', 0))
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0: continue
        flows = finishes.get((e.get('pid'), e.get('tid'), start), [])
        hosts = [sources[str(f['id'])] for f in flows if str(f.get('id')) in sources]
        matches = {i for host in hosts for i, (lo, hi) in enumerate(ranges) if lo <= timestamp(host) <= hi}
        if len(matches) != 1: continue
        kernels.append({'name': e.get('name'), 'ts_us': start, 'duration_us': duration,
                        'pid': e.get('pid'), 'tid': e.get('tid'), 'call': next(iter(matches)), 'host_launch_us': [timestamp(h) for h in hosts], 'args': e.get('args', {})})
    if not kernels or {k['call'] for k in kernels} != set(range(5)):
        return {'status': 'partial', 'reason': 'no device events attributable to every target call', 'kernel_count': None}
    calls = []
    for index in range(5):
        kk = [k for k in kernels if k['call'] == index]
        intervals = [(k['ts_us'], k['ts_us'] + k['duration_us']) for k in kk]
        calls.append({'kernel_count': len(kk), 'kernel_sum_us': sum(k['duration_us'] for k in kk),
                      'device_union_us': union_duration(intervals),
                      'device_span_us': max(y for x,y in intervals)-min(x for x,y in intervals)})
    return {'status': 'measured', 'kernel_count': len(kernels), 'calls': calls, 'kernels': kernels, 'clock_origin_us': str(origin), 'timestamp_scope': 'microseconds relative to clock_origin_us',
            'boundary': 'CANN HostToDevice flows correlate host launches inside MSTX ranges to device events; no cross-clock containment assumption; cumulative/union/span are separate'}


def parse_capture(folder):
    candidates = []
    errors = []
    for path in sorted(folder.rglob('msprof_[0-9]*.json')):
        try:
            raw = read(path)
            events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
            if isinstance(events, list):
                result = analyze_events(events)
                result['trace'] = str(path.relative_to(folder))
                candidates.append(result)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            errors.append(f'{path.name}: {exc}')
    measured = [c for c in candidates if c['status'] == 'measured']
    if len(measured) == 1: return measured[0]
    return {'status': 'partial', 'reason': 'missing or ambiguous target device trace', 'candidates': candidates, 'errors': errors}


def counter_rows(folder, timeline):
    # Preserve exported metric units/headers. Match device task and stream IDs only.
    records = []
    ids = set()
    for k in timeline.get('kernels', []):
        args = {str(key).lower().replace(' ', '_'): str(v) for key,v in k.get('args', {}).items()}
        stream = args.get('physic_stream_id', args.get('stream_id'))
        if 'task_id' in args and stream is not None:
            ids.add((args['task_id'], stream))
    for path in sorted(folder.rglob('*.csv')):
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                keys = {key.lower().replace(' ', '_'): key for key in row if key}
                if 'task_id' not in keys or 'stream_id' not in keys: continue
                key = (row[keys['task_id']], row[keys['stream_id']])
                if key in ids:
                    records.append({'file': str(path.relative_to(folder)), 'values': row})
    return records



def hardware_metrics(records, group):
    patterns = {
        'ArithmeticUtilization': ('_fp16_ratio', '_fp32_ratio', '_int8_ratio', '_fops'),
        'PipeUtilization': ('_mac_ratio', '_vec_ratio', '_mte1_ratio', '_mte2_ratio'),
        'Memory': ('_main_mem_', '_l1_', '_l2_'),
        'MemoryL0': ('_l0a_', '_l0b_', '_l0c_'),
        'MemoryUB': ('_ub_read_bw_', '_ub_write_bw_'),
        'ResourceConflictRatio': ('_cflt_ratio',),
    }
    found = []
    for record in records:
        values = {}
        for key, value in record['values'].items():
            if not any(token in key.lower() for token in patterns[group]): continue
            try: numeric = float(value)
            except (ValueError, TypeError): continue
            if math.isfinite(numeric): values[key] = numeric
        if values: found.append({**record, 'metrics': values})
    return found

def collect(root, task):
    tool = shutil.which('msprof')
    result = {'protocol': 'ascend-profile-v1', 'mode': task['profiling_mode'], 'status': 'partial',
              'groups': [], 'boundary': 'independent replays, not clean benchmark timings; groups are different executions'}
    if not tool:
        result['reason'] = 'msprof not found'
        write(root / 'profiling.json', result)
        return
    result['tool'] = {'path': tool, 'sha256': sha(tool)}
    result['collector_sha256'] = sha(Path(__file__))
    started = time.monotonic()
    requested = ('timeline',) + (GROUPS if task['profiling_mode'] == 'full' else ())
    for index, group in enumerate(requested, 1):
        group_started = time.monotonic()
        def progress(state):
            publish_profile(root, group, index, len(requested), state, time.monotonic() - group_started)
        progress('preparing')
        folder = root / 'profiling' / group
        folder.mkdir(parents=True, exist_ok=True)
        # Dedicated root keeps each replay identity, window and errors independent.
        for name in ('inputs.pt', 'reference.pt', 'correctness.json', 'task.json'):
            if (root / name).exists(): shutil.copyfile(root / name, folder / name)
        command = [tool, '--output=' + str(folder / 'raw'), '--runtime-api=on', '--task-time=on',
                   '--msproftx=on', '--aicpu=on', '--ai-core=' + ('off' if group == 'timeline' else 'on')]
        if group != 'timeline': command += ['--aic-metrics=' + group]
        command += [sys.executable, str(Path(__file__).resolve().parents[2] / 'runtime/worker.py'),
                    '--root', str(folder), '--phase', 'profile-target']
        write(folder / 'command.json', {'argv': command})
        entry = {'group': group, 'status': 'partial', 'directory': str(folder.relative_to(root))}
        with (folder / 'capture.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + max(1, task.get('watchdog_s', 300) - 20 - (time.monotonic()-started))
                while True:
                    progress('capturing')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, 0)
                    try:
                        proc.wait(timeout=min(1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                entry['reason'] = 'capture watchdog exceeded'
                result['execution_error'] = True
                result['groups'].append(entry)
                progress('timeout')
                break
        progress('parsing')
        # CANN creates owner-only raw directories. Evidence must remain inspectable
        # by the host owner and included by the host manifest walker.
        for path in [folder, *folder.rglob('*')]:
            if not path.is_symlink(): path.chmod(0o755 if path.is_dir() else 0o644)
        entry['returncode'] = proc.returncode
        if proc.returncode or not (folder / 'profile-window.json').exists() or (folder / 'profile-target-error.json').exists():
            entry['reason'] = 'capture or target execution failed; inspect capture.log'
        else:
            parsed = parse_capture(folder)
            entry['timeline'] = parsed
            entry['status'] = parsed['status']
            if group != 'timeline':
                records = hardware_metrics(counter_rows(folder, parsed), group)
                write(folder / 'counter-rows.json', {'rows': records, 'units': 'original CSV column headers'})
                entry['counter_rows'] = len(records)
                # Task rows alone are not evidence of counter values.
                if not records:
                    entry.update(status='partial', reason='no target-correlated hardware metric values')
        if '[flagos cpu_fallback]' in (folder / 'capture.log').read_text(errors='replace'):
            entry.update(status='partial', reason='CPU fallback observed in capture process; target attribution is not qualified')
        identity_path = folder / 'profile-target-identity.json'
        if identity_path.exists() and read(identity_path).get('input_sha256') != sha(root / 'inputs.pt'):
            entry.update(status='partial', reason='profile replay input identity mismatch')
        result['groups'].append(entry)
        progress(entry['status'])
        write(root / 'profiling.json', result)
        # A target execution failure must not be repeated for every counter group.
        if (folder / 'profile-target-error.json').exists() or proc.returncode < 0:
            result['execution_error'] = True
            break
    for path in [root / 'profiling', *(root / 'profiling').rglob('*')]:
        if not path.is_symlink(): path.chmod(0o755 if path.is_dir() else 0o644)
    requested = ('timeline',) + (GROUPS if task['profiling_mode'] == 'full' else ())
    for group in requested:
        if not any(g['group'] == group for g in result['groups']):
            result['groups'].append({'group': group, 'status': 'skipped', 'reason': 'earlier target failure or capture watchdog; no further replay', 'directory': f'profiling/{group}'})
    expected = len(requested)
    result['status'] = 'measured' if len(result['groups']) == expected and all(g['status'] == 'measured' for g in result['groups']) else 'partial'
    result['elapsed_s'] = time.monotonic()-started
    write(root / 'profiling.json', result)
