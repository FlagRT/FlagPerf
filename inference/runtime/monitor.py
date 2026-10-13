"""Per-run management-plane collector, isolated from model processes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from runtime.common import ROOT, read_json, write_json
from runtime.monitor_events import ENV
from vendors.telemetry import metric, observation


HOST_METRICS = [metric('host.memory.available', 'B', 'Linux MemAvailable', 'host'),
                metric('process.cpu', 'cores', 'delta CPU seconds / elapsed seconds; one fully busy core = 1', 'process'),
                metric('process.rss', 'B', 'Linux resident pages; shared pages not deduplicated', 'process')]


def read_events(folder):
    rows, errors = [], []
    for path in sorted((Path(folder)/'events').glob('*.jsonl')):
        for index, line in enumerate(path.read_text().splitlines()):
            try:
                rows.append(json.loads(line))
            except ValueError:
                errors.append(f'{path.name}:{index+1}: incomplete event')
    return sorted(rows, key=lambda r: r['timestamp_ns']), errors


class HostReader:
    def __init__(self):
        self.previous, self.mapping = {}, {}
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.page = os.sysconf('SC_PAGE_SIZE')

    def resolve(self, event):
        key = (event['pid_namespace'], event['pid'])
        if key in self.mapping:
            return self.mapping[key]
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if (proc/'ns/pid').stat().st_ino != key[0]:
                    continue
                status = (proc/'status').read_text()
                ids = next(line.split()[1:] for line in status.splitlines() if line.startswith('NSpid:'))
                if int(ids[-1]) == key[1]:
                    self.mapping[key] = proc
                    return proc
            except (OSError, ValueError, StopIteration):
                continue
        return None

    def read(self, events):
        rows, entities = [], {}
        now = time.perf_counter_ns()
        try:
            available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:'))
            rows.append(observation('host', HOST_METRICS[0], available, now, source='/proc/meminfo'))
        except (OSError, ValueError, StopIteration):
            rows.append(observation('host', HOST_METRICS[0], None, now, error='MemAvailable unavailable'))
        active = {}
        for event in events:
            if event['event'] == 'worker_start': active[event['worker']] = event
            if event['event'] == 'worker_end': active.pop(event['worker'], None)
        for worker, event in active.items():
            entity = 'process:'+worker
            entities[entity] = dict(scope='process', worker=worker, ranks=[event['rank']])
            proc = self.resolve(event)
            try:
                if proc is None: raise OSError('worker PID mapping unavailable')
                stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                identity = (str(proc), int(stat[19]))  # PID plus start ticks protects against PID reuse.
                if event.get('process_start_ticks', identity[1]) != identity[1]:
                    raise OSError('worker PID was reused')
                ticks, rss = int(stat[11])+int(stat[12]), int(stat[21])*self.page
                previous = self.previous.get(entity)
                cpu = None
                if previous and previous[0] == identity and now > previous[1] and ticks >= previous[2]:
                    cpu = (ticks-previous[2])/self.ticks/((now-previous[1])/1e9)
                self.previous[entity] = (identity, now, ticks)
                rows.append(observation(entity, HOST_METRICS[1], cpu, now, raw=ticks,
                    error=None if cpu is not None else 'first counter sample', source='/proc/PID/stat'))
                rows.append(observation(entity, HOST_METRICS[2], rss, now, source='/proc/PID/stat'))
            except (OSError, ValueError, IndexError):
                for definition in HOST_METRICS[1:]:
                    rows.append(observation(entity, definition, None, now, error='process exited or PID access unavailable'))
        return rows, entities


def collect(request_path):
    request = read_json(request_path)
    folder = Path(request_path).parent
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    interval = request['interval_ms']/1000
    provider = None
    inventory = dict(schema_version=1, metrics=HOST_METRICS, entities={'host': {'scope': 'host'}},
                     provider='unavailable', unavailable={}, requested_devices=request['devices'],
                     clock=time.get_clock_info('perf_counter').implementation,
                     clock_anchor_ns=time.perf_counter_ns(), utc_anchor_ns=time.time_ns(),
                     interval_ms=request['interval_ms'], status='running')
    write_json(folder/'heartbeat.json', {'time_ns': time.perf_counter_ns()})
    try:
        from runtime.adapters import create
        provider = create('vendors', request['vendor']).telemetry_provider()
        if provider is None: raise RuntimeError('vendor has no telemetry provider')
        device = provider.discover(request['devices'])
        inventory.update({k: v for k, v in device.items() if k not in ('metrics', 'entities')})
        inventory['metrics'] += device['metrics']
        inventory['entities'].update(device['entities'])
    except Exception as error:
        inventory['device_error'] = str(error)
        provider = None
    write_json(folder/'inventory.json', inventory)
    reader, sequence = HostReader(), 0
    try:
        with (folder/'samples.jsonl').open('a', buffering=1) as stream:
            while not stop.is_set():
                started = time.perf_counter_ns()
                write_json(folder/'heartbeat.json', {'time_ns': started})
                events, _ = read_events(folder)
                rows, entities = reader.read(events)
                if entities.keys() - inventory['entities'].keys():
                    inventory['entities'].update(entities)
                    write_json(folder/'inventory.json', inventory)
                if provider:
                    try:
                        rows += provider.read()
                    except Exception as error:
                        stream.write(json.dumps(dict(kind='error', sequence=sequence, error=str(error),
                                                     timestamp_ns=time.perf_counter_ns()))+'\n')
                if sequence == 0:
                    inventory['initial_probe'] = [{k:r.get(k) for k in ('entity','metric','status','reason','source')}
                                                  for r in rows]
                    write_json(folder/'inventory.json', inventory)
                for row in rows:
                    stream.write(json.dumps(dict(kind='sample', sequence=sequence, **row), allow_nan=False)+'\n')
                sequence += 1
                write_json(folder/'heartbeat.json', {'time_ns': time.perf_counter_ns()})
                stop.wait(max(0, interval-(time.perf_counter_ns()-started)/1e9))
    finally:
        if provider: provider.close()
        inventory.update(status='stopped', iterations=sequence)
        write_json(folder/'inventory.json', inventory)


class MonitorSession:
    def __init__(self, cfg, root):
        self.root, self.folder = Path(root), Path(root)/'monitor'
        self.process, self.log = None, None
        self.stop_watch = threading.Event()
        self.error = None
        self.thread = None
        options = cfg.get('performance', {})
        self.enabled = options.get('monitor', 'on') == 'on'
        self.interval_ms = options.get('monitor_interval_ms', 1000)
        self.previous_env = os.environ.get(ENV)
        self.cfg = cfg

    def start(self):
        if not self.enabled: return self
        try:
            (self.folder/'events').mkdir(parents=True, exist_ok=True)
            runtime = self.cfg['runtime']
            devices = runtime['devices'] if runtime.get('parallelism') == 'tp' else [runtime['device']]
            write_json(self.folder/'request.json', dict(schema_version=1, vendor=runtime['vendor'],
                devices=devices, interval_ms=self.interval_ms))
            env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
            env.pop(ENV, None)
            self.log = (self.folder/'collector.log').open('w')
            self.started_ns = time.perf_counter_ns()
            self.process = subprocess.Popen([sys.executable, '-u', '-m', 'runtime.monitor',
                str(self.folder/'request.json')], cwd=ROOT, env=env, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
            os.environ[ENV] = str(self.root)
            self.thread = threading.Thread(target=self._watch, daemon=True, name='monitor-watchdog')
            self.thread.start()
        except (OSError, ValueError) as error:
            self.error = str(error)
        return self

    def _watch(self):
        timeout = max(10, 3*self.interval_ms/1000)
        while not self.stop_watch.wait(1):
            if self.process.poll() is not None: return
            try:
                stamp = read_json(self.folder/'heartbeat.json')['time_ns']
            except (OSError, ValueError, KeyError):
                stamp = self.started_ns
            if (time.perf_counter_ns()-stamp)/1e9 > timeout:
                self.error = 'collector heartbeat timeout'
                self.process.kill()
                return

    def finish(self):
        self.stop_watch.set()
        if self.thread: self.thread.join(timeout=2)
        if self.previous_env is None: os.environ.pop(ENV, None)
        else: os.environ[ENV] = self.previous_env
        if not self.enabled: return {'status': 'not_requested', 'enabled': False}
        try:
            if self.process:
                if self.process.poll() is None:
                    self.process.terminate()
                    try: self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill(); self.process.wait(timeout=3)
                        self.error = 'collector shutdown timeout'
                if self.process.returncode not in (0, None) and not self.error:
                    self.error = f'collector exited with {self.process.returncode}'
            if self.log: self.log.close()
            from runtime.monitor_analysis import summarize
            return summarize(self.root, self.error)
        except Exception as error:
            return {'status': 'failed', 'enabled': True, 'error': str(error)}


def finish_host(session):
    """Host is sole writer after container exit; never changes measurement/exit status."""
    monitoring = session.finish()
    root = session.root
    try:
        result = read_json(root/'result.json')
        result['monitoring'] = monitoring
        write_json(root/'result.json', result)
        from runtime.monitor_report import safe_render
        safe_render(root, result)
        source_path = root/'report-source.json'
        if source_path.is_file():
            from runtime.common import file_hash
            source = read_json(source_path)
            source['result_sha256'] = file_hash(root/'result.json')
            source.setdefault('generator_sha256', {})['monitor_report.py'] = file_hash(ROOT/'runtime/monitor_report.py')
            write_json(source_path, source)
    except Exception as error:
        print('Monitor finalization: '+str(error), file=sys.stderr)


if __name__ == '__main__':
    collect(sys.argv[1])
