"""PPU CUDA-compat timeline capture with conservative attribution."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import os

from runtime.evidence import read, write, sha
from .trace import analyze_events

ASIGHT_SECTIONS = (
    'SpeedOfLight',
    'MemoryWorkloadAnalysis',
    'Occupancy',
    'WarpStateStats',
    'LaunchStats',
)
ASIGHT_SECTION_NAMES = {
    'SpeedOfLight': 'PPU Speed Of Light Throughput',
    'MemoryWorkloadAnalysis': 'Memory Workload Analysis',
    'Occupancy': 'Occupancy',
    'WarpStateStats': 'Warp State Statistics',
    'LaunchStats': 'Launch Statistics',
}


def target_window(adapter, root, task, call):
    import torch
    for _ in range(task['case_config']['WARMUP']):
        out = call(); del out
    adapter.synchronize()
    if os.environ.get('FLAGPERF_PPU_COUNTER_TARGET') == '1':
        api = torch.cuda.cudart()
        started = api.cudaProfilerStart()
        if int(started) != 0:
            raise RuntimeError('PPU profiler-start API failed: ' + str(started))
        try:
            for _ in range(5):
                out = call(); adapter.synchronize(); del out
        finally:
            stopped = api.cudaProfilerStop()
        if int(stopped) != 0:
            raise RuntimeError('PPU profiler-stop API failed: ' + str(stopped))
        write(root / 'profile-window.json', {'calls': 5, 'device': adapter.device.index,
              'boundary': 'five original target invocations bracketed by native profiler-start/stop after warmup',
              'status': 'completed'})
        return
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        for index in range(5):
            with torch.profiler.record_function('operation_target_%d' % index):
                out = call(); adapter.synchronize(); del out
    prof.export_chrome_trace(str(root / 'trace.json'))
    (root / 'profile-window.json').write_text(json.dumps({
        'calls': 5, 'device': adapter.device.index, 'marker': 'operation_target_',
        'status': 'captured',
        'boundary': 'five synchronized CUDA-compat profiler windows; hardware attribution is not asserted'}, indent=2) + '\n')


def collect(root, task):
    folder = root / 'profiling' / 'timeline'; folder.mkdir(parents=True, exist_ok=False)
    for name in ('inputs.pt', 'reference.pt', 'correctness.json', 'task.json'):
        if (root / name).is_file(): shutil.copyfile(root / name, folder / name)
    command = [sys.executable, '-B', '-S', str(Path(__file__).with_name('bootstrap.py')),
               '--root', str(folder), '--phase', 'profile-target']
    log = folder / 'capture.log'
    began = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try: proc.wait(timeout=max(1, task.get('watchdog_s', 300) - 20))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=3)
    entry = {'group': 'timeline', 'directory': 'profiling/timeline', 'returncode': proc.returncode}
    if proc.returncode or not (folder / 'trace.json').is_file():
        entry.update(status='partial', reason='PPU profiler target replay failed; inspect capture.log')
    else:
        identity = read(folder / 'profile-target-identity.json')
        measured = read(root / 'measurement.json')
        fields = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256', 'registration', 'runtime_config', 'device', 'input_devices')
        mismatches = [k for k in fields if k not in identity or k not in measured or identity[k] != measured[k]]
        window = read(folder / 'profile-window.json')
        timeline = analyze_events(read(folder / 'trace.json').get('traceEvents', []), window['device'])
        if mismatches:
            entry.update(status='partial', reason='profile/measure identity mismatch', mismatches=mismatches)
        else:
            entry.update(status=timeline['status'], reason=timeline.get('reason'))
        entry['timeline'] = {**timeline, 'trace': 'trace.json'}
        entry['trace_sha256'] = __import__('hashlib').sha256((folder / 'trace.json').read_bytes()).hexdigest()
    result = {'protocol': 'thead-ppu-cuda-compat-profile-v1', 'mode': task['profiling_mode'],
              'status': 'measured' if entry.get('status') == 'measured' else 'partial', 'groups': [entry],
              'boundary': 'timeline evidence is diagnostic; it does not qualify performance or prove absence of fallback',
              'counter_groups': [{'group': n, 'status': 'not-supported',
                  'reason': 'no reviewed equivalence to this Ascend group; PPU-native metrics are recorded separately'}
                  for n in ('ArithmeticUtilization','PipeUtilization','Memory','MemoryL0','MemoryUB','ResourceConflictRatio')]}
    if proc.returncode:
        result['execution_error'] = True
    if task['profiling_mode'] == 'full':
        budget = task.get('watchdog_s', 300) - (time.monotonic() - began) - 20
        result['native_counters'] = collect_counters(root, task, timeout=max(1, budget))
        native = result['native_counters']
        result['groups'].append({**native, 'group': 'PPUAsight',
            'timeline': {'status': 'partial', 'reason': 'PPU native per-kernel counter report is not a per-call device timeline'},
            'reason': native.get('reason', 'PPU-native per-kernel metrics retained; per-call timeline attribution remains separate')})
        if result['native_counters'].get('execution_error'):
            result['execution_error'] = True
        result['status'] = 'partial'  # Native target-range counters do not establish per-call device timing.
    result['elapsed_s'] = time.monotonic() - began
    write(root / 'profiling.json', result)


def parse_counter_csv(text):
    """Retain PPU-native per-kernel metric names and units without remapping."""
    import csv
    import io
    import math
    lines = text.splitlines()
    required = {'ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'}
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        if required <= set(fields):
            rows = list(csv.DictReader(io.StringIO('\n'.join(lines[index:]))))
            records = []
            for row in rows:
                # Live acu output appends progress/report messages after CSV.
                # DictReader leaves absent columns as None for those lines.
                raw_value = row.get('Metric Value')
                if not isinstance(raw_value, str):
                    continue
                try:
                    value = float(raw_value.replace(',', ''))
                except (ValueError, TypeError, KeyError):
                    continue
                if not math.isfinite(value) or not all(row.get(k) for k in required - {'Metric Value'}):
                    continue
                records.append({k: row[k] for k in fields if row.get(k) is not None})
            return records
    return []


def collect_counters(root, task, timeout=None):
    folder = root / 'profiling' / 'native-counters'
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('inputs.pt', 'reference.pt', 'correctness.json', 'task.json'):
        if (root / name).is_file():
            shutil.copyfile(root / name, folder / name)
    executable = Path('/usr/local/PPU_SDK/asight/bin/acu')
    result = {'status': 'partial', 'directory': 'profiling/native-counters',
              'boundary': 'PPU-native per-kernel metrics within five original target invocations; not Ascend L0/UB equivalents or clean timing',
              # Asight accepts section Identifiers, which differ from two SDK
              # filenames: WarpStateStats and LaunchStats.
              'sections': list(ASIGHT_SECTIONS)}
    if not executable.is_file():
        return {**result, 'status': 'not-supported', 'reason': 'native Asight acu executable unavailable'}
    command = [str(executable), '--devices', str(task['worker_device']), '--profile-from-start', 'off',
               '--kill', 'no', '--check-exit-code', 'yes', '--page', 'details', '--csv',
               '--export', str(folder / 'target')]
    for section in result['sections']:
        command += ['--section', section]
    command += [sys.executable, '-B', '-S', str(Path(__file__).with_name('bootstrap.py')),
                '--root', str(folder), '--phase', 'profile-target']
    write(folder / 'command.json', {'argv': command})
    env = {**os.environ, 'FLAGPERF_PPU_COUNTER_TARGET': '1'}
    with (folder / 'capture.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout if timeout is not None else max(1, task.get('watchdog_s', 300) - 20))
        except subprocess.TimeoutExpired:
            result.update(execution_error=True, reason='native counter replay watchdog exceeded')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=3)
    result['returncode'] = proc.returncode
    if proc.returncode or (folder / 'profile-target-error.json').exists():
        return {**result, 'execution_error': True, 'reason': 'native counter target failed; original capture.log retained'}
    if not (folder / 'profile-target-identity.json').is_file() or not (folder / 'profile-window.json').is_file():
        return {**result, 'reason': 'native counter target completion/identity missing'}
    identity, measured = read(folder / 'profile-target-identity.json'), read(root / 'measurement.json')
    keys = ('vendor_runtime', 'packages', 'case_sha256', 'input_sha256', 'registration', 'runtime_config', 'device', 'input_devices')
    mismatches = [k for k in keys if k not in identity or k not in measured or identity[k] != measured[k]]
    if mismatches:
        return {**result, 'reason': 'counter/measure identity mismatch', 'mismatches': mismatches}
    records = parse_counter_csv((folder / 'capture.log').read_text(errors='replace'))
    reports = [p for p in folder.iterdir() if p.suffix in ('.acurep', '.acu-rep')]
    if not records or not reports:
        return {**result, 'reason': 'native metric records or native report missing; no fabricated zero metrics'}
    write(folder / 'metrics.json', records)
    rows = [{'values': {**row, 'Op Name': row['Kernel Name']},
             'metrics': {row['Metric Name'] + ' [' + row['Metric Unit'] + ']': float(row['Metric Value'].replace(',', ''))}}
            for row in records]
    write(folder / 'counter-rows.json', {'rows': rows, 'protocol': 'ppu-asight-native-metrics-v1'})
    sections = {}
    for row in records:
        sections.setdefault(row['ID'], set()).add(row.get('Section Name'))
    expected = set(ASIGHT_SECTION_NAMES.values())
    missing = {key: sorted(expected - present) for key, present in sections.items() if expected - present}
    return {**result, 'status': 'partial' if missing else 'measured', 'metric_records': len(records),
            'kernel_records': len(sections), 'missing_sections_by_kernel': missing,
            'reason': 'requested section metrics missing for one or more kernels' if missing else None,
            'native_reports': {p.name: sha(p) for p in reports}, 'metrics_sha256': sha(folder / 'metrics.json')}
