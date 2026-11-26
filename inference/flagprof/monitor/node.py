"""Foreground node supervisor and isolated provider worker; no system service."""
import argparse
import fcntl
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
from flagprof.store.manifest import atomic_json, proc_key, process_alive
from flagprof.telemetry.contract import raw_clock
from flagprof.telemetry.host_proc import Provider as Host, host_pid
from flagprof.telemetry.sampler import Sampler
from .leases import Leases, device_keys
from .aggregation import Minutes
from .consumer import Consumer
from .retention import CapacityError, RollingWriter, recover_closed
from .session import configuration, edit, initialize, load, locked
from .subscriptions import Server
from .supervision import Heartbeat, reclaim


def sampling_interval(requested, inventory, previous=None, calibrated=100):
    minimum=max(math.ceil(inventory.get('minimum_interval_ms',100)),calibrated)
    # SMI measures a whole command round. Round its minimum up to a second so
    # startup timing jitter cannot silently change the rule configuration.
    quantum = inventory.get('interval_quantum_ms', 1)
    minimum = math.ceil(minimum / quantum) * quantum
    interval=max(requested,minimum)
    if previous:
        if minimum>previous:raise ValueError('source sampling floor increased: recalibrate in a new session')
        interval=max(interval,previous)
    return interval


def drain_notifications(root, writer, capacity):
    for path in sorted((root/'notifications').glob('*.json'))[:capacity]:
        payload = path.read_bytes()
        writer.emit(json.loads(payload))
        writer.writer.flush()
        # An authoritative wait receipt can replace a provisional end record
        # while its Store write is in flight. Acknowledge only what we flushed.
        with locked(root):
            if path.read_bytes() == payload:
                path.unlink()


def worker(root):
    root = Path(root); state = load(root); config = state['config']
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    writer = sampler = consumer = minutes = diagnostic_inputs = None
    try:
        provider, identity = Registry().create('telemetry.'+state['vendor'],devices=state['devices'],source=config['source'],processes=True)
        host = Host()
        inventory = dict(provider.inventory,boot_id=state['boot_id'],source_identity=identity['shipped_source_closure'],
                         descriptors=[m.snapshot() for m in provider.metrics+host.metrics])
        interval = sampling_interval(config['interval_ms'], inventory)
        inventory.update(effective_interval_ms=interval, requested_interval_ms=config['interval_ms'])
        atomic_json(root/'probe.json',dict(process=proc_key(),inventory=inventory))
        while not stopping and not (root/'worker-grant.json').exists():
            time.sleep(.02)
        if stopping: return 0
        grant = json.loads((root/'worker-grant.json').read_text())
        if grant['process'] != proc_key(): raise ValueError('worker lease grant identity')
        from .calibration import calibrate
        floor = calibrate(root, state['vendor'], 'agent', provider, inventory)
        interval = sampling_interval(config['interval_ms'], inventory, state.get('inventory',{}).get('effective_interval_ms'), floor)
        inventory.update(minimum_interval_ms=floor, effective_interval_ms=interval)
        sampler = Sampler(provider.calls()+host.calls(),interval,timeout=inventory.get('call_timeout_s',3),
                          queue_capacity=config['queue_capacity'])
        inventory['sampling_epoch_ns']=sampler.epoch_ns
        writer = RollingWriter(root,inventory)
        with edit(root) as state: state.update(status='running',inventory=inventory)
        consumer=Consumer(root);minutes=Minutes(root)
        if 'diagnosis' in config:
            from .diagnostic_inputs import Inputs
            diagnostic_inputs=Inputs(root,writer)
        from .retention import publish_control
        from .rules import VERSION
        if not (root/'rules.json').exists():
            publish_control(root,root/'rules.json',dict(schema='monitor-rule-config/1',rule_version=VERSION,
                config_version=consumer.engine.config_id,config=config,source_identity=inventory['source_identity']))
        known = {};known_pids={}; next_scan = next_tick = next_consume = 0
        def publish(points,health):
            for event in health: writer.emit(event)
            for point in points:
                event=dict(t='counter',src='host' if point.entity.startswith(('host:','proc:')) else 'device',
                    dev=point.entity,ts=point.start_ns,dur=point.end_ns-point.start_ns,values=point.values,miss=point.miss,
                    ext=dict(telemetry=dict(point.detail,source=point.source,sample_times=point.sample_times)))
                writer.emit(event);minutes.add(event)
                for event in point.health: writer.emit(event)
                if point.entity.startswith('proc:'):
                    for ident,call in known.items():
                        if call.entity!=point.entity and not point.entity.startswith(call.entity+'/'): continue
                        with locked(root):
                            path=root/'targets/active'/ident
                            if not path.exists(): path=root/'targets/history'/ident
                            if path.exists():
                                data=json.loads(path.read_text())
                                field='device_process_coverage' if '/' in point.entity else 'process_coverage'
                                value='observed' if any(v is not None for v in point.values.values()) else 'unavailable'
                                if data.get(field)!=value and data.get(field)!='observed':
                                    data[field]=value; atomic_json(path,data)
        while not stopping:
            now = time.monotonic()
            if now >= next_scan:
                next_scan = now+sampler.slow
                (root/'heartbeat').touch()
                active = {}; device_targets={};target_devices={};active_pids={}
                for path in sorted((root/'targets/active').glob('*.json'))[:config['max_targets']]:
                    try:
                        data = json.loads(path.read_text()); key = data['worker_proc_key']
                        pid = host_pid(key,known_pid=known_pids.get(path.name))
                        if pid is None:
                            # Disappearance is not an exit status. The launcher can
                            # subsequently replace this unknown with its wait receipt.
                            Server.handle(type('Local',(),{'root':root})(),dict(action='finish',reference=data,path=str(path)))
                            continue
                        device_targets[pid]=key
                        active_pids[path.name]=pid
                        target_devices[pid]=tuple(d['entity'] for d in data.get('devices',[]) if d.get('status')=='available')
                        if path.name not in known:
                            call = host.process(key,pid); sampler.calls.append(call)
                            active[path.name] = call
                        else: active[path.name] = known[path.name]
                    except FileNotFoundError: continue
                for ident,call in known.items():
                    if ident not in active:
                        sampler.remove([call.key]); host.forget(call.entity)
                known = active;known_pids=active_pids
                if hasattr(provider,'targets'):
                    provider.target_devices=target_devices;provider.targets=device_targets
                if load(root)['status']=='capacity': raise CapacityError('control_capacity')
                drain_notifications(root, writer, config['queue_capacity'])
            publish(*sampler.tick())
            if now>=next_tick:
                ready=diagnostic_inputs.poll() if diagnostic_inputs else True
                next_tick=time.monotonic()+(sampler.interval if ready else .05)
                writer.emit(dict(t='health',ts=raw_clock(),kind='heartbeat',sev='info',
                    detail=dict(scope='monitor_tick',started_raw_ns=state['started_raw_ns'])))
            if now>=next_consume:
                next_consume=now+.2;consumer.poll()
            time.sleep(min(.05,sampler.interval/4))
        for points,health in sampler.finish(): publish(points,health)
        if diagnostic_inputs:
            deadline=time.monotonic()+1
            while not diagnostic_inputs.poll() and time.monotonic()<deadline:
                consumer.poll()
        drain_notifications(root, writer, config['queue_capacity'])
        return 0
    except BaseException as error:
        with edit(root) as state:
            state.update(status='capacity' if isinstance(error,CapacityError) else 'failed',
                         worker_error=type(error).__name__+': '+str(error)[:2048])
        return 4 if isinstance(error,CapacityError) else 2
    finally:
        if sampler: sampler.close()
        if diagnostic_inputs:diagnostic_inputs.close()
        if writer: writer.close()
        try:
            if minutes:minutes.flush()
            if consumer:consumer.poll()
        except CapacityError:
            pass  # Raw evidence and the capacity state are already retained.
        finally:
            if consumer:consumer.close()


def start(root, vendor, devices='all', config=None, *, module='flagprof.monitor.node'):
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    lock = (root/'.service.lock').open('a+b')
    try: fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close(); raise ValueError('monitor root already supervised')
    child = server = None; leases = Leases(); stopping = False; old_handlers = {}
    def stop(*_):
        nonlocal stopping
        stopping = True
    try:
        state = initialize(root,vendor,devices,config or configuration())
        owner = state.get('worker_process')
        if owner and process_alive(owner) is not False:
            with edit(root) as state:state['status']='unreaped'
            raise ValueError('previous worker exit not confirmed')
        recover_closed(root)
        for sig in (signal.SIGTERM,signal.SIGINT): old_handlers[sig] = signal.signal(sig,stop)
        server = Server(root); expected_keys = None
        now=time.monotonic();raw=raw_clock()
        restarts=[now-(raw-row['ts'])/1e9 for row in state['restarts'] if 0<=raw-row['ts']<600*1e9]
        while not stopping:
            for name in ('probe.json','worker-grant.json','heartbeat'): (root/name).unlink(missing_ok=True)
            with edit(root) as state: state['status'] = 'starting'
            child = subprocess.Popen([sys.executable,'-B','-m',module,'--worker','--root',str(root)],
                                     env=module_environment(), stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            child_key = proc_key(child.pid)
            with edit(root) as state: state['worker_process']=child_key
            heartbeat = Heartbeat(root/'heartbeat',max(10,3*state['config']['interval_ms']/1000))
            granted = False; failure = None
            while child.poll() is None and not stopping:
                server.step()
                if not granted and (root/'probe.json').exists():
                    probe = json.loads((root/'probe.json').read_text())
                    if probe['process']['pid'] != child.pid: raise ValueError('probe child identity')
                    keys = set(device_keys(state,probe['inventory']))
                    if expected_keys is None:
                        leases.acquire(keys); expected_keys = keys
                    elif keys != expected_keys:
                        raise ValueError('device inventory changed during restart')
                    leases.owner(probe['process'])
                    atomic_json(root/'worker-grant.json',dict(process=probe['process']))
                    heartbeat.timeout=max(10,3*probe['inventory']['effective_interval_ms']/1000)
                    granted = True
                if heartbeat.expired(): failure = 'watchdog'; break
            result = reclaim(child)
            if not result['confirmed']:
                with edit(root) as state: state.update(status='unreaped',reclaim=result)
                return 3
            leases.owner(None)
            recover_closed(root)
            if stopping:
                # A killed worker cannot drain its final committed segment. The
                # supervisor owns consumption only after its exit is confirmed.
                final_consumer=Consumer(root)
                try:final_consumer.poll()
                finally:final_consumer.close()
                break
            current = load(root)
            if current['status']=='capacity': return 4
            now = time.monotonic(); restarts = [t for t in restarts if now-t<600]
            with edit(root) as state:
                timestamp=raw_clock()
                state['gaps'].append(dict(ts=timestamp,reason=failure or 'collector_exit',returncode=child.returncode))
                pending=root/'notifications';pending.mkdir(exist_ok=True)
                atomic_json(pending/('gap-'+str(len(state['gaps']))+'.json'),dict(t='health',ts=timestamp,
                    kind='heartbeat',sev='warn',detail=dict(scope='monitor_gap',entity='node',
                        reason=failure or 'collector_exit',returncode=child.returncode)))
                # Bound metadata by stopping, not by silently discarding evidence.
                if len(state['gaps']) >= state['config']['max_segments']:
                    state['status']='capacity'; return 4
            if len(restarts)>=3:
                with edit(root) as state: state['status']='restart_limit'
                return 2
            delay=(1,2,4)[len(restarts)]; restarts.append(now)
            with edit(root) as state:
                state['restarts'].append(dict(ts=raw_clock(),backoff_s=delay))
            until=time.monotonic()+delay
            while not stopping and time.monotonic()<until: server.step()
        with edit(root) as state: state.update(status='stopped',ended_raw_ns=raw_clock())
        return 0
    except BaseException as error:
        if (root/'session.json').exists():
            with edit(root) as state:
                if state['status']!='unreaped':state['status']='failed'
                state['supervisor_error']=type(error).__name__+': '+str(error)[:2048]
        raise
    finally:
        if child is not None and child.poll() is None:
            result=reclaim(child)
            if result['confirmed']: leases.owner(None)
            else:
                with edit(root) as state: state.update(status='unreaped',reclaim=result)
        if server: server.close()
        leases.close(); lock.close()
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--root',required=True); args=parser.parse_args()
    if not args.worker: parser.error('use flagprof watch start')
    return worker(args.root)


if __name__=='__main__': raise SystemExit(main())
