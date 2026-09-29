"""Selected-identity P800 telemetry; no daemon or global PID file."""
import hashlib
import json
from pathlib import Path
import threading
import time

from executors.lifecycle import HostCommands, save
from base.vendors.kunlunxin.preflight import machine_inventory, query_record


class UsageMonitor:
    def __init__(self, targets, *, commands=None, interval_s=1.0):
        if not targets or len({t['device_id'] for t in targets}) != len(targets) or interval_s <= 0:
            raise ValueError('explicit unique monitor targets and positive interval required')
        self.targets = list(targets)
        self.commands = commands or HostCommands()
        self.interval_s = interval_s
        self.origin_monotonic_s = time.monotonic()
        self.raw_records, self.samples = [], []
        self.stop_event = threading.Event()
        self.thread = None

    def collect_once(self):
        machine = self.commands.run(['xpu-smi', '-m'], timeout=3)
        self.raw_records.append(machine)
        for target in self.targets:
            query = self.commands.run(['xpu-smi', '-i', str(target['physical_id']), '-q'], timeout=3)
            self.raw_records.append(query)
            sample = {'device_id': target['device_id'], 'physical_id': target['physical_id'],
                      'started_offset_s': machine['started_monotonic_s'] - self.origin_monotonic_s,
                      'finished_offset_s': query['finished_monotonic_s'] - self.origin_monotonic_s,
                      'values': {}, 'valid': False}
            try:
                if machine['returncode'] or query['returncode']:
                    raise ValueError('telemetry command failed or timed out')
                rows = [r for r in machine_inventory(machine['stdout']) if r['physical_id'] == target['physical_id']]
                if len(rows) != 1 or rows[0]['pci_bdf'] != target['pci_bdf']:
                    raise ValueError('telemetry physical identity changed')
                values = query_record(query['stdout'], target['pci_bdf'])
                if values['uuid'] != target['uuid'] or values['total_memory_mib'] != rows[0]['total_memory_mib']:
                    raise ValueError('telemetry UUID or memory units changed')
                sample.update(valid=True, values={k: v for k, v in values.items() if k in (
                    'utilization_percent', 'used_memory_mib', 'total_memory_mib', 'free_memory_mib', 'temperature_c', 'power_w')})
            except (ValueError, KeyError, TypeError) as exc:
                sample['error'] = str(exc)
            self.samples.append(sample)

    def _loop(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.collect_once()
            except Exception as exc:
                self.raw_records.append({'error': str(exc), 'returncode': 127})
                for target in self.targets:
                    self.samples.append({'device_id': target['device_id'], 'valid': False, 'values': {},
                                         'started_offset_s': started - self.origin_monotonic_s,
                                         'finished_offset_s': time.monotonic() - self.origin_monotonic_s, 'error': str(exc)})
            self.stop_event.wait(max(0, self.interval_s - (time.monotonic() - started)))

    def start(self):
        if self.thread is not None:
            raise RuntimeError('monitor already started')
        self.thread = threading.Thread(target=self._loop, name='flagperf-p800-sampler', daemon=False)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(3 * (1 + len(self.targets)) + 2)
            if self.thread.is_alive():
                raise RuntimeError('sampler did not stop within bounded command budget')

    def finish(self, root, directory, windows, extensions, *, min_samples_per_target=10,
               primary_role='measurement', window_semantics='', target_resource=None):
        self.stop()
        directory.mkdir(parents=True, exist_ok=True)
        refs = {}
        for name, rows in [('samples.raw.jsonl', self.raw_records), ('samples.jsonl', self.samples)]:
            path = directory / name
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
            refs[name] = {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'bytes': path.stat().st_size}
        counts = {t['device_id']: 0 for t in self.targets}
        invalid = 0
        for sample in self.samples:
            matching = [w for w in windows if w.get('role') == primary_role and w.get('device_id') == sample['device_id']
                        and w['finished_offset_s'] > w['started_offset_s']
                        and sample['started_offset_s'] <= w['finished_offset_s']
                        and sample['finished_offset_s'] >= w['started_offset_s']]
            if matching:
                if sample['valid']:
                    counts[sample['device_id']] += 1
                else:
                    invalid += 1
        reasons = [f'{key}: {count}/{min_samples_per_target} valid samples' for key, count in counts.items() if count < min_samples_per_target]
        if invalid:
            reasons.append(f'{invalid} invalid samples in observation window')
        result = {'schema_version': 2, 'vendor': 'kunlunxin', 'collector': 'xpu-smi -m and selected -q',
                  'status': 'partial' if reasons else 'passed', 'reasons': reasons, 'targets': self.targets,
                  'clock_domain': 'same-host Linux monotonic', 'workload_windows': list(windows),
                  'window_semantics': window_semantics, 'target_resource': target_resource,
                  'primary_sample_counts_by_target': counts, 'sample_counts_by_target': counts,
                  'raw_samples': refs['samples.raw.jsonl'], 'parsed_samples': refs['samples.jsonl'],
                  'extensions': list(extensions), 'total_sample_commands': len(self.raw_records)}
        save(directory / 'summary.json', result)
        return result
