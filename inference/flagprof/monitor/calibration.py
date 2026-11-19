"""Sampling floors and calibration receipts tied to the actual collection stack."""
import hashlib
import json
import math
from pathlib import Path
import time


def binding(vendor, topology, inventory):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(vendor=vendor, topology=topology, boot_id=inventory['boot_id'],
                source=inventory['source'], driver=inventory.get('driver_version'),
                sdk=inventory.get('library_version'), hardware=digest(inventory['entities']),
                collector=inventory['source_identity']['sha256'], metrics=digest(inventory['descriptors']))


def calibrate(root, vendor, topology, provider, inventory):
    key = binding(vendor, topology, inventory)
    inventory['calibration_binding'] = key
    base = math.ceil(inventory.get('minimum_interval_ms', 100))
    path = Path(root) / 'telemetry-calibration.json'
    for candidate in (path, Path(root) / 'telemetry-sampling.json'):
        if not candidate.is_file():
            continue
        try:
            previous = json.loads(candidate.read_text())
            floor = previous.get('minimum_interval_ms', base)
            valid = type(floor) in (float, int) and math.isfinite(floor) and floor > 0
        except (ValueError, OSError, AttributeError):
            valid = False
        if valid and previous.get('binding') == key:
            inventory.update(idle_baselines=previous.get('idle_baselines', {}),
                             calibration_status='matched')
            return max(base, math.ceil(floor))
        inventory['calibration_status'] = 'identity_mismatch'
    round_count = inventory.get('calibration_rounds', 0)
    if not round_count:
        return base
    # Called under the device lease in a supervised sampler process. Only
    # complete fast rounds count. A stalled SDK read triggers its watchdog.
    rounds = []
    for _ in range(round_count):
        start = time.monotonic()
        for call in provider.calls():
            if call.group == 'fast':
                call.read()
        rounds.append((time.monotonic() - start) * 1000)
    inventory['sampling_calibration'] = dict(rounds_ms=rounds, source='current provider')
    minimum = max(base, math.ceil(2 * max(rounds)))
    from flagprof.store.manifest import atomic_json
    atomic_json(Path(root) / 'telemetry-sampling.json', dict(binding=key, minimum_interval_ms=minimum,
                                                            rounds_ms=rounds))
    return minimum
