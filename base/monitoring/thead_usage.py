"""Selected-identity PPU telemetry from one full `ppu-smi -q` per round."""
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from executors.lifecycle import HostCommands, save

from base.vendors.thead.provider import canonical_uuid, normalize_bdf

# ppu-smi -q prints one `PPU <bdf>` section per card; a full query costs under
# one second for 16 cards (day1 probe D), cheaper and more consistent than
# per-target queries.
_SECTION = re.compile(r'^PPU (' + r'[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]' + r')[ \t]*$', re.M)


def parse_q_sections(raw):
    """Split one full -q dump into per-card sections keyed by canonical BDF."""
    marks = [(match.start(), match.end(), normalize_bdf(match[1])) for match in _SECTION.finditer(raw)]
    if not marks:
        raise ValueError('empty ppu-smi -q dump')
    sections = {}
    for index, (_, end, bdf) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(raw)
        if bdf in sections:
            raise ValueError('duplicate ppu-smi section: ' + bdf)
        sections[bdf] = raw[end:stop]
    return sections


def one(raw, pattern, label):
    values = re.findall(pattern, raw, re.M)
    if len(values) != 1:
        raise ValueError('missing or ambiguous query field: ' + label)
    return values[0]


def parse_card(raw, bdf):
    identity = {'pci_bdf': bdf}
    identity['uuid'] = canonical_uuid(one(raw, r'^\s*PPU UUID\s*:\s*GPU-([0-9a-fA-F-]+)\s*$', 'uuid'))
    identity['device_minor'] = int(one(raw, r'^\s*Minor Number\s*:\s*(\d+)\s*$', 'minor'))
    identity['host_device_node'] = '/dev/alixpu_ppu%d' % (identity['device_minor'] - 1)
    memory = one(raw, r"^    Memory Usage\s*\n((?:[ \t]+[^\n]*\n)+?)(?=    \S)", 'Memory Usage')
    values = {}
    for label in ('Total', 'Used', 'Free'):
        value = one(memory, r"^\s+%s\s*:\s*(\d+) MiB\s*$" % label, label)
        values[label.lower() + '_memory_mib'] = int(value)
    values['utilization_percent'] = int(one(raw, r"^\s+Ppu\s*:\s*(\d+) %\s*$", 'utilization'))
    if not (0 <= values['used_memory_mib'] <= values['total_memory_mib'] and
            0 <= values['free_memory_mib'] <= values['total_memory_mib'] and
            values['total_memory_mib'] > 0 and 0 <= values['utilization_percent'] <= 100):
        raise ValueError('invalid query telemetry range')
    process_sections = re.findall(r'^    Processes(?:[ \t]*:[ \t]*(None))?[ \t]*$', raw, re.M)
    if len(process_sections) != 1:
        raise ValueError('missing or ambiguous query field: processes')
    values['processes_empty'] = process_sections[0] == 'None'
    if not values['processes_empty'] and not re.search(r'^        Process ID[ \t]*:[ \t]*[0-9]+[ \t]*$', raw, re.M):
        raise ValueError('unrecognized nonempty process table')
    # Historical counters are observations, not newly detected errors. Preserve
    # them so acceptance cannot silently promote a degraded/shared run.
    ecc = [int(v) for v in re.findall(r'DRAM Uncorrectable\s*:\s*(\d+)', raw)]
    values['uncorrectable_ecc_counts'] = ecc
    values['health_status'] = 'degraded' if any(ecc) else 'observed-clean' if len(ecc) == 2 else 'unknown'
    for field, pattern in [('temperature_c', r"^\s+PPU Current Temp\s*:\s*([0-9.]+) C\s*$"),
                           ('power_w', r"^\s+Power Draw\s*:\s*([0-9.]+) W\s*$")]:
        found = re.findall(pattern, raw, re.M)
        if len(found) == 1:
            value = float(found[0])
            if math.isfinite(value) and value >= 0:
                values[field] = value
    return {**identity, **values}


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
        query = self.commands.run(['ppu-smi', '-q'], timeout=5)
        self.raw_records.append(query)
        started_offset_s = query['started_monotonic_s'] - self.origin_monotonic_s
        finished_offset_s = query['finished_monotonic_s'] - self.origin_monotonic_s
        sections = {}
        if not query['returncode']:
            try:
                sections = parse_q_sections(query['stdout'])
            except ValueError:
                sections = {}
        for target in self.targets:
            sample = {'device_id': target['device_id'], 'physical_id': target['physical_id'],
                      'started_offset_s': started_offset_s, 'finished_offset_s': finished_offset_s,
                      'values': {}, 'valid': False}
            try:
                if query['returncode']:
                    raise ValueError('telemetry command failed or timed out')
                bdf = normalize_bdf(target['pci_bdf'])
                card = parse_card(sections[bdf], bdf)
                if card['uuid'] != target['uuid'] or card['device_minor'] != target['physical_id'] + 1:
                    raise ValueError('telemetry identity changed')
                sample.update(valid=True, values={k: v for k, v in card.items() if k in (
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
        self.thread = threading.Thread(target=self._loop, name='flagperf-ppu-sampler', daemon=False)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(int(5 + self.interval_s) + 2)
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
        result = {'schema_version': 2, 'vendor': 'thead', 'collector': 'ppu-smi single -q full parse',
                  'status': 'partial' if reasons else 'passed', 'reasons': reasons, 'targets': self.targets,
                  'clock_domain': 'same-host Linux monotonic', 'workload_windows': list(windows),
                  'window_semantics': window_semantics, 'target_resource': target_resource,
                  'primary_sample_counts_by_target': counts, 'sample_counts_by_target': counts,
                  'raw_samples': refs['samples.raw.jsonl'], 'parsed_samples': refs['samples.jsonl'],
                  'extensions': list(extensions), 'total_sample_commands': len(self.raw_records)}
        save(directory / 'summary.json', result)
        return result
