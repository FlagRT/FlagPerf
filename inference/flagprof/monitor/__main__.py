"""Public host entry and supervised local sampler. Standard library only."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from flagprof._process import module_environment
from flagprof.registry import Registry
from flagprof.vendors import monitored_vendors, vendor_names
from flagprof.store import Writer, WriterConfig, create_run, finalize
from flagprof.store.manifest import atomic_json, proc_key
from flagprof.telemetry.contract import raw_clock
from flagprof.telemetry.host_proc import Provider as Host, host_pid
from flagprof.telemetry.sampler import Sampler
from .supervision import Heartbeat, reclaim
from .leases import Leases, device_keys


def sample(args):
    run = Path(args.run)
    inventory = dict(version=1, status='running', started_raw_ns=raw_clock(), ended_raw_ns=None,
        boot_id=proc_key()['boot_id'], entities=[], descriptors=[], source=args.source, errors=[])
    stop = False
    def terminate(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    sampler = writer = None
    leases = Leases()
    heartbeat = run/'heartbeat'
    heartbeat.touch()
    try:
        host_only = args.vendor not in monitored_vendors()
        registry = Registry()
        if host_only:
            host, identity = registry.create('telemetry.host')
            provider = host
            metrics, calls = host.metrics, host.calls()
            inventory.update(device_telemetry='unavailable', source='host-procfs')
        else:
            provider, identity = registry.create('telemetry.' + args.vendor, devices=args.devices, source=args.source)
            host = Host()
            inventory.update(provider.inventory)
            metrics, calls = provider.metrics + host.metrics, provider.calls() + host.calls()
        inventory.update(source_identity=identity['shipped_source_closure'],
                         descriptors=[metric.snapshot() for metric in metrics])
        leases.acquire(device_keys(dict(boot_id=inventory['boot_id'],vendor=args.vendor),inventory))
        leases.owner(proc_key())
        from .calibration import calibrate
        inventory['minimum_interval_ms'] = calibrate(args.root, args.vendor, 'local', provider, inventory)
        atomic_json(run/'inventory.json', inventory)
        writer = Writer(run, kind='counter', config=WriterConfig(shard_period_seconds=300))
        effective_ms = max(args.interval_ms, math.ceil(inventory.get('minimum_interval_ms',100)))
        if effective_ms > 5000:
            raise ValueError('source round time exceeds 5000 ms monitoring interval')
        inventory.update(requested_interval_ms=args.interval_ms,effective_interval_ms=effective_ms)
        atomic_json(run/'inventory.json', inventory)
        sampler = Sampler(calls, effective_ms,
                          timeout=inventory.get('call_timeout_s', 3))
        known = {}; next_scan = 0
        awaiting_pre_sample = {e['entity'] for e in inventory['entities'] if e['entity'].startswith('chip:')}
        first_sample = False
        def publish(observations,health):
            nonlocal first_sample
            for event in health:
                writer.emit(event)
            for point in observations:
                writer.emit(dict(t='counter', src='host' if point.entity.startswith(('host:', 'proc:')) else 'device',
                    dev=point.entity, ts=point.start_ns, dur=point.end_ns-point.start_ns,
                    values=point.values, miss=point.miss,
                    ext=dict(telemetry=dict(point.detail, source=point.source, sample_times=point.sample_times))))
                sample_metric = 'host.memory.available' if host_only else 'memory.used'
                if not first_sample and point.values.get(sample_metric) is not None:
                    awaiting_pre_sample.discard(point.entity)
                    if not awaiting_pre_sample:
                        atomic_json(run/'first-sample.json', dict(ts=point.start_ns, entities=[e['entity'] for e in inventory['entities'] if e['entity'].startswith('chip:')]))
                        first_sample = True
                for event in point.health:
                    writer.emit(event)
        while not stop:
            now = time.monotonic()
            if now >= next_scan:
                next_scan = now+sampler.slow
                heartbeat.touch()
                active = {}
                for target in (Path(args.root)/'targets').glob('*.json'):
                    try:
                        registration = json.loads(target.read_text())
                        key = registration['worker_proc_key']
                        ident = json.dumps(key, sort_keys=True)
                        if registration.get('window', [None, None])[1] is not None:
                            continue
                        if len(active) >= 256:
                            continue
                        pid = host_pid(key)
                        if pid is not None:
                            active[ident] = known.get(ident)
                            if ident not in known:
                                call = host.process(key, pid)
                                sampler.calls.append(call); active[ident] = call
                                inventory.setdefault('processes',[]).append(dict(worker_proc_key=key,host_pid=pid))
                                atomic_json(run/'inventory.json',inventory)
                    except (OSError, ValueError, KeyError):
                        continue
                for ident, call in known.items():
                    if ident not in active:
                        sampler.remove([call.key]); host.forget(call.entity)
                known = active
            observations, health = sampler.tick()
            publish(observations,health)
            time.sleep(min(.05, sampler.interval/4))
        for observations,health in sampler.finish():publish(observations,health)
        inventory['status'] = 'partial' if sampler.isolated else 'complete'
    except Exception as error:
        inventory.update(status='failed', errors=inventory['errors']+[type(error).__name__+': '+str(error)])
    finally:
        if sampler:
            sampler.close()
        if writer:
            writer.close()
        inventory['ended_raw_ns'] = raw_clock()
        atomic_json(run/'inventory.json', inventory)
        try:
            finalize(run)
        finally:
            try: leases.owner(None)
            finally: leases.close()
    return 0 if inventory['status'] in ('complete', 'partial') else 2


def supervise(args):
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    run = Path(args.run) if args.run else create_run(root/'monitor', identity=dict(kind='monitor',vendor=args.vendor))
    command = [sys.executable, '-B', '-m', 'flagprof.monitor', '--sample', '--root', str(root),
               '--run', str(run), '--vendor', args.vendor, '--devices', args.devices,
               '--interval-ms', str(args.interval_ms), '--source', args.source]
    child = subprocess.Popen(command, env=module_environment())
    stopping = False
    def terminate(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    heartbeat = Heartbeat(run/'heartbeat', max(10, 3*args.interval_ms/1000))
    timed_out = False
    while child.poll() is None and not stopping:
        if heartbeat.expired():
            timed_out = True
            break
        time.sleep(.1)
    reclaimed = reclaim(child)
    if timed_out or child.returncode not in (0, 2):
        path = run/'inventory.json'
        inventory = json.loads(path.read_text()) if path.exists() else dict(started_raw_ns=None, entities=[])
        inventory.update(status='failed', ended_raw_ns=raw_clock(),
                         watchdog=dict(timed_out=timed_out, **reclaimed))
        atomic_json(path, inventory)
    return child.returncode if reclaimed['confirmed'] else 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--vendor', choices=vendor_names(), required=True)
    parser.add_argument('--devices', default='all')
    parser.add_argument('--interval-ms', type=int, default=1000)
    parser.add_argument('--source', choices=('auto','dcmi','dsmi','npu-smi','nvml','nvidia-smi'), default='auto')
    parser.add_argument('--run', help=argparse.SUPPRESS)
    parser.add_argument('--sample', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 100 <= args.interval_ms <= 5000:
        parser.error('interval must be 100..5000 ms')
    return sample(args) if args.sample else supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
