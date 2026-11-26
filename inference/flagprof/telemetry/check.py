# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Diagnose the device telemetry stack on this machine.

``flagprof telemetry check --vendor <v> [--devices all] [--json]``

Prints, in order:
- the resolved provider source and library path;
- the discovered entities (chip UUIDs) and driver/library versions;
- every metric the provider exposes, and every metric it could not read
  with the reason;
- on failure, every library path tried with its per-path error, plus
  actionable hints.

Runs under -S (no site-packages) like the monitor worker, so it reports
exactly what an agent monitor session would see.
"""
import json as _json
import sys


HINTS = {
    'thead': (
        'PPU: mount the driver library into the container '
        '(-v /usr/lib64/libnvidia-ml.so:/usr/lib64/libnvidia-ml.so:ro) or set '
        'FLAGPROF_TELEMETRY_LIBRARY to the libnvidia-ml.so path.'),
    'kunlunxin': (
        'P800: the NVML-compatible library ships inside the image Python '
        'packages (triton/backends/xpu/*/so/libxpunvidia-ml.so); run inside the '
        'vendor image, or set FLAGPROF_TELEMETRY_LIBRARY explicitly.'),
    'ascend': (
        'Ascend: DCMI (libascendcl.so) comes from the CANN runtime; the container '
        'must mount the host driver/CANN directories (launcher.md §9).'),
    'nvidia': (
        'NVIDIA: libnvidia-ml.so comes from the driver installation; mount it '
        'into the container or set FLAGPROF_TELEMETRY_LIBRARY.'),
}


def run(vendor, devices='all', as_json=False):
    report = dict(vendor=vendor, devices=devices, ok=False, source=None,
                  library=None, entities=[], metrics=[], unavailable={},
                  driver_version=None, hints=HINTS.get(vendor, ''))
    try:
        from flagprof.registry import Registry
        provider, _identity = Registry().create('telemetry.' + vendor, devices=devices)
        inventory = provider.inventory
        report.update(ok=True, source=inventory.get('source'),
                      library=inventory.get('library'),
                      driver_version=inventory.get('driver_version'),
                      entities=[dict(entity=e.get('entity'), name=e.get('name'),
                                     pci=e.get('pci')) for e in inventory.get('entities', [])],
                      metrics=[dict(name=m.name, unit=m.unit, kind=m.kind, api=m.source_api,
                                    roles=list(m.roles)) for m in getattr(provider, 'metrics', [])],
                      unavailable=dict(inventory.get('unavailable', {})))
    except Exception as error:  # keep the full message: it enumerates attempts
        report['error'] = type(error).__name__ + ': ' + str(error)
    if as_json:
        print(_json.dumps(report, ensure_ascii=False))
        return 0 if report['ok'] else 2
    if report['ok']:
        print('vendor:', report['vendor'], '| source:', report['source'])
        print('library:', report['library'])
        print('driver:', report['driver_version'])
        print('entities:', len(report['entities']))
        for entity in report['entities'][:16]:
            print('  ', entity['entity'], entity['name'] or '')
        print('metrics:', len(report['metrics']))
        for metric in report['metrics']:
            print('  ', metric['name'], '[' + str(metric['unit']) + ']', metric['kind'], metric['api'])
        if report['unavailable']:
            print('unavailable:')
            for name, reason in sorted(report['unavailable'].items()):
                print('  ', name, '->', reason)
        return 0
    print('vendor:', report['vendor'], '| telemetry unavailable', file=sys.stderr)
    print(report.get('error', 'unknown error'), file=sys.stderr)
    if report['hints']:
        print('hint:', report['hints'], file=sys.stderr)
    return 2
