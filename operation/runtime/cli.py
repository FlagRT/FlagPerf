# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Host/container-neutral orchestration. Vendor policy is delegated to adapters."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import shutil
import sys
import time
import uuid

from runtime import catalog
from runtime.progress import tracked, stage
from runtime.diagnostics import PROTOCOL as DIAGNOSTIC_PROTOCOL
from runtime.evidence import read, write, sha, seal, report, merge_runs
from vendors import AVAILABLE, get_vendor

ROOT = catalog.ROOT


class CleanupError(RuntimeError):
    pass


class WorkloadOption(argparse.Action):
    """Reject both spellings even when an explicit value equals the default."""
    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, '_workload_option', None)
        if previous is not None and previous != option_string:
            parser.error('--profile and --workload cannot be used together')
        namespace._workload_option = option_string
        setattr(namespace, self.dest, values)


class Lease:
    def __init__(self, root, devices, run_id):
        self.root, self.devices, self.run_id = root, devices, run_id
        self.streams = []

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for device in sorted(self.devices):
                stream = (self.root / f'logical-device-{device}.lock').open('a+')
                try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    raise RuntimeError(f'device {device} already leased')
                self.streams.append(stream)
                stream.seek(0);stream.truncate()
                stream.write(json.dumps({'run_id': self.run_id, 'pid': os.getpid(), 'kind': 'operation'}));stream.flush()
        except BaseException:
            self.release();raise

    def release(self):
        for stream in self.streams:
            fcntl.flock(stream, fcntl.LOCK_UN);stream.close()
        self.streams = []


def ids(value):
    values = []
    for part in value.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-', 1));values.extend(range(lo, hi + 1))
        else: values.append(int(part))
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise ValueError('device IDs must be nonnegative and unique')
    return values


def parser():
    p = argparse.ArgumentParser(description='Portable operation benchmark; no host.yaml or SSH initialization.')
    sub = p.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list', help='list cases and supported input types')
    listing.add_argument('--vendor', choices=AVAILABLE)
    list_mode = listing.add_mutually_exclusive_group()
    list_mode.add_argument('--names-only', action='store_true', help='print one case name per line')
    list_mode.add_argument('--case', metavar='CASE', help='print supported input types for one case')
    run = sub.add_parser('run', allow_abbrev=False)
    run.add_argument('--vendor', choices=AVAILABLE, required=True)
    run.add_argument('--device-ids', required=True, help='host IDs for docker; runtime-visible IDs for local')
    run.add_argument('--execution', choices=['docker', 'local'], default='docker')
    run.add_argument('--image')
    run.add_argument('--case', action='append', help='repeat to select cases; omitted means every case')
    run.add_argument('--dtype', nargs='+', choices=[*catalog.FLOATS, *catalog.INTEGERS, 'INT64'])
    run.add_argument('--oplib', choices=['nativetorch', 'flaggems', 'both'], default='nativetorch', help='both compares native/FlagGems within this run only; no cross-run comparison')
    workload = run.add_mutually_exclusive_group()
    workload.add_argument('--workload', choices=['daily', 'smoke'], default='daily', action=WorkloadOption, help='workload preset')
    workload.add_argument('--profile', dest='workload', choices=['daily', 'smoke'], default=argparse.SUPPRESS,
                         action=WorkloadOption, help=argparse.SUPPRESS)
    run.add_argument('--profiling', choices=['off', 'timeline', 'full'], default='off', help='independent device capture; off keeps daily measurements only')
    run.add_argument('--size', action='append', help='repeat dimension sets, e.g. M=128,N=256,K=128')
    run.add_argument('--warmup', type=int);run.add_argument('--iters', type=int);run.add_argument('--rounds', type=int)
    run.add_argument('--spectflops', type=float, help='peak for the selected device scope and single dtype; omitted means FU=N/A')
    run.add_argument('--seed', type=int, default=2026)
    run.add_argument('--watchdog', type=float, default=300, help='per-phase hang safeguard, independent of the 120s soft target')
    run.add_argument('--result-root', type=Path, default=ROOT / 'result')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--diagnostics', choices=['off', 'failures'], default='failures')
    run.add_argument('--allow-privileged-root', action='store_true')
    run.add_argument('--container-device', action='append', default=[], help='explicit additional Docker device mapping')
    run.add_argument('--mount', action='append', default=[], help='explicit Docker mount source:destination:mode')
    rebuild = sub.add_parser('report');rebuild.add_argument('--run-dir', type=Path, action='append', required=True)
    rebuild.add_argument('--output-dir', type=Path, help='new directory for merging multiple source runs')
    from runtime.diagnosis import add_parser
    add_parser(sub)
    return p


def resolve(args):
    selected = ids(args.device_ids)
    if not math.isfinite(args.watchdog) or args.watchdog <= 0:
        raise ValueError('watchdog must be positive and finite')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in 0..2**32-1')
    if args.spectflops is not None:
        if not math.isfinite(args.spectflops) or args.spectflops <= 0:
            raise ValueError('spectflops must be positive and finite')
        if args.dtype and len(args.dtype) != 1:
            raise ValueError('a peak requires a single explicit dtype')
    adapter = get_vendor(args.vendor)
    image = args.image or (adapter.default_image() if args.execution == 'docker' else None)
    if args.execution == 'docker' and not image:
        raise ValueError('this vendor requires --image')
    tasks = catalog.expand(args.case, args.dtype, args.oplib, profile=args.workload,
                           sizes=args.size, warmup=args.warmup, iters=args.iters, rounds=args.rounds)
    tasks = [{**t, 'device_id': d, 'worker_device': 0 if args.execution == 'docker' else d,
              'vendor': args.vendor, 'seed': args.seed, 'spectflops': args.spectflops,
              'profiling_mode': args.profiling, 'oplib_selection': args.oplib, 'performance_protocol': 'operation-performance-v1',
              'watchdog_s': args.watchdog,
              'diagnostics_mode': args.diagnostics, 'diagnostic_protocol': DIAGNOSTIC_PROTOCOL} for d in selected for t in tasks]
    from runtime.metrics import pair_key
    for task in tasks:
        task['pair_id'] = pair_key(task) if args.oplib == 'both' else None
    return adapter, {'schema_version': 2, 'protocol': 'operation-v2', 'vendor': args.vendor,
                     'execution': args.execution, 'image': image, 'device_ids': selected,
                     'workload': args.workload, 'profiling': args.profiling, 'oplib_selection': args.oplib, 'soft_target_s': 120, 'watchdog_per_phase_s': args.watchdog,
                     'tasks': tasks}


class Workers:
    """Isolated persistent workers per device, library and phase.

    Each request has its own log and completion record. Worker death/timeout
    tears down the pool; never advance past unconfirmed cleanup.
    """
    def __init__(self, root, adapter, args, image_id, run_id):
        self.root, self.adapter, self.args = root, adapter, args
        self.image_id, self.run_id = image_id, run_id
        self.active = {}
        self.generation = 0

    def start(self, task, mode):
        key = (task['device_id'], task['oplib'], mode)
        if key in self.active: return self.active[key]
        self.generation += 1
        label = f'd{key[0]}-{key[1]}-{mode}-{self.generation}'
        control = self.root / 'workers' / label
        control.mkdir(parents=True, exist_ok=True)
        name = f'flagperf-op-{self.run_id}-{label}'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        relative = str(control.relative_to(self.root))
        if self.args.execution == 'docker':
            operation_source = self.root / 'source'
            command = ['docker', 'run', '-d', '--rm', '--name', name, '--network=none',
                       '-v', f'{operation_source}:/operation:ro', '-v', f'{self.root}:/evidence:rw',
                       '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0',
                       '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1']
            if mode not in ('reference', 'diagnose', 'reference-check'):
                command += self.adapter.docker_options(task['device_id'], self.args)
                for k,v in self.adapter.environment(task['device_id'], mode).items():
                    command += ['-e', f'{k}={v}']
                for device in self.args.container_device: command += ['--device', device]
                for mount in self.args.mount: command += ['-v', mount]
            command += [self.image_id, 'python3', '-X', 'faulthandler', '/operation/runtime/worker.py', '--root', '/evidence',
                        '--phase', mode, '--serve', relative]
            write(control / 'command.json', {'argv':command})
            # Track the exact name before launch: a client timeout may leave a live container.
            worker = {'name':name, 'control':control, 'proc':None}
            self.active[key] = worker
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f'worker launch failed: {result.stderr}')
            worker = {'name':name, 'control':control, 'proc':None}
        else:
            if mode not in ('reference', 'diagnose', 'reference-check'): env.update(self.adapter.environment(task['device_id'], mode, local=True))
            command = [sys.executable, str(self.root / 'source/runtime/worker.py'), '--root', str(self.root),
                       '--phase', mode, '--serve', relative]
            write(control / 'command.json', {'argv':command})
            with (control / 'server.log').open('w') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            worker = {'name':name, 'control':control, 'proc':proc}
        self.active[key] = worker
        return worker

    def phase(self, root, task, mode):
        stage(self.args, mode, root / 'profiling-progress.json' if mode == 'profiling' else None)
        started = time.monotonic()
        last_liveness_check = started
        worker = self.start(task, mode)
        control = worker['control']
        request_id = uuid.uuid4().hex
        write(control / 'request.json', {'id': request_id, 'directory':root.name})
        while True:
            done = control / 'done.json'
            if done.exists():
                record = read(done)
                if record['id'] == request_id:
                    write(root / f'{mode}-exit.json', {**record, 'elapsed_s':time.monotonic()-started,
                          'worker': str(control.relative_to(self.root)), 'initialization_reused':record.get('request_count',1)>1})
                    if record['returncode'] != 0:
                        raise RuntimeError(f'{mode}: {record.get("error", "worker failed")}')
                    return
            if time.monotonic()-started > self.args.watchdog:
                raise RuntimeError(f'{mode}: watchdog exceeded {self.args.watchdog}s')
            if worker['proc'] is not None and worker['proc'].poll() is not None:
                raise RuntimeError(f'{mode}: worker exited unexpectedly')
            if task.get('common_diagnosis') and worker['proc'] is None and time.monotonic() - last_liveness_check >= 2:
                state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', worker['name']],
                                       capture_output=True, text=True, timeout=10)
                last_liveness_check = time.monotonic()
                if state.returncode or state.stdout.strip() != 'true':
                    raise RuntimeError(f'{mode}: container exited unexpectedly: {state.stderr.strip()}')
            time.sleep(0.05)

    def close(self, mode=None):
        stage(self.args, 'cleanup' + (f'-{mode}' if mode else ''))
        errors = []
        for key, worker in list(self.active.items()):
            if mode is not None and key[2] != mode:
                continue
            try:
                if worker['proc'] is None:
                    result = subprocess.run(['docker','rm','-f',worker['name']], capture_output=True, text=True, timeout=30)
                    write(worker['control'] / 'cleanup.json', {'returncode':result.returncode, 'stderr':result.stderr})
                    if result.returncode and 'No such container' not in result.stderr:
                        raise CleanupError(result.stderr)
                else:
                    proc = worker['proc']
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL);proc.wait(timeout=5)
                del self.active[key]
            except Exception as exc: errors.append(str(exc))
        if errors: raise CleanupError('; '.join(errors))


def snapshot(root):
    files = [ROOT / 'run.py', *list((ROOT / 'runtime').glob('*.py')),
             *list((ROOT / 'vendors').rglob('*.py')), *list((ROOT / 'benchmarks').glob('*/main.py')),
             *list((ROOT / 'benchmarks/drivers').glob('*.py'))]
    for p in files:
        dest = root / 'source' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes(p.read_bytes())
    for label, cmd in [('git-head.txt', ['git','rev-parse','HEAD']), ('git-status.txt', ['git','status','--short']),
                       ('git-diff.patch', ['git','diff','HEAD','--','operation'])]:
        result = subprocess.run(cmd, cwd=ROOT.parent, capture_output=True, text=True)
        (root / label).write_text(result.stdout)


@tracked
def execute(args, adapter, plan):
    progress = args._progress
    stage(args, 'snapshot')
    token = uuid.uuid4().hex[:12]
    root = args.result_root.resolve() / f'operation-{token}'
    root.mkdir(parents=True)
    args._result_root = root
    write(root / 'plan.json', plan);snapshot(root)
    summary = {'schema_version': 2, 'status': 'partial', 'tasks': [], 'vendor': args.vendor,
               'execution': args.execution, 'device_ids': plan['device_ids'], 'protocol': plan['protocol'],
               'run_id': token, 'oplib_selection': args.oplib, 'workload': args.workload, 'profiling': args.profiling}
    lease = Lease(adapter.lease_root(), plan['device_ids'], token)
    began = time.monotonic();ran_device=False;image_id=None;pool=None
    try:
        stage(args, 'runtime-identity')
        if args.execution == 'docker':
            info = json.loads(subprocess.check_output(['docker','image','inspect',plan['image']], text=True))[0]
            summary['runtime'] = adapter.runtime_identity(plan['image'], info)
            image_id = info['Id'];summary['actual_image_id'] = image_id
        else:
            summary['runtime'] = {'validation': 'local runtime packages recorded per worker; image provenance externally managed',
                                  'visibility': {k:v for k,v in os.environ.items() if k.endswith('VISIBLE_DEVICES')}}
        stage(args, 'preflight')
        lease.acquire()
        if args.execution == 'docker':
            summary['preflight'] = adapter.preflight(root, plan['device_ids'])
        else:
            summary['preflight'] = {'status': 'externally-managed', 'boundary': 'local lease protects this namespace only; caller reserves host devices'}
        pool = Workers(root, adapter, args, image_id, token)
        references = {}
        for index, task in enumerate(plan['tasks']):
            progress.task(task, index + 1, len(plan['tasks']))
            task['run_id'] = token
            task['actual_image_id'] = image_id
            item = {**task, 'status': 'not-run'}
            summary['tasks'].append(item)
            if not task['applicable']:
                item.update(status='not-applicable', reason=f"{task['case']} accepts {catalog.dtypes(task['case'])}")
                progress.done(item['status'])
                continue
            directory = f'{index:04d}-{task["case"]}-{task["dtype"]}-{task["oplib"]}-d{task["device_id"]}'
            work = root / directory;work.mkdir();write(work / 'task.json', task)
            item['directory'] = directory;started = time.monotonic()
            try:
                item['phase'] = 'reference'
                pair = task.get('pair_id')
                if pair and pair in references:
                    stage(args, 'reference-reuse')
                    for name in ('inputs.pt', 'reference.pt', 'reference.json'):
                        source = references[pair] / name
                        if source.exists(): shutil.copyfile(source, work / name)
                else:
                    pool.phase(work, task, 'reference')
                    if pair: references[pair] = work
                ran_device = True
                item['phase'] = 'probe'
                pool.phase(work, task, 'probe')
                item['routing'] = adapter.route(read(work / 'probe.json'), (work / 'probe.log').read_text())
                item['correctness'] = read(work / 'correctness.json')
                item['phase'] = 'measure'
                pool.phase(work, task, 'measure')
                item['measurement'] = read(work / 'measurement.json')
                # Preparation fallback is recorded separately; it cannot prove target fallback.
                measured_log = (work / 'measure.log').read_text()
                item['measurement_fallback_observed'] = adapter.measurement_fallback(measured_log)
                correct = item['correctness']['status'] == item['measurement']['correctness']['status'] == 'passed'
                if not correct or item['routing']['status'] == 'failed':
                    item['status'] = 'failed'
                else:
                    item['status'] = 'passed' if item['routing']['status'] == 'passed' and not item['measurement_fallback_observed'] else 'partial'
                if not correct:
                    item.update(failure_stage='correctness', diagnosis='numerical-mismatch: original tolerance exceeded')
                elif item['status'] != 'passed':
                    item.update(failure_stage='routing', diagnosis='routing-evidence: inspect target dispatch and fallback evidence')
                item['phase'] = 'completed'
                # Extra measurement errors do not overwrite clean timing or correctness.
                for extra in ('memory', 'profiling'):
                    if extra == 'profiling' and task.get('profiling_mode', 'off') == 'off':
                        item[extra] = {'status': 'off'}
                        continue
                    if item['status'] != 'passed':
                        item[extra] = {'status': 'skipped', 'reason': 'requires passed correctness and routing'}
                        continue
                    try:
                        pool.phase(work, task, extra)
                        item[extra] = read(work / f'{extra}.json')
                        if item[extra].get('execution_error'):
                            pool.close()
                            if args.execution == 'docker':
                                try:
                                    adapter.preflight(root, plan['device_ids'], label=f'health-{extra}-{index}')
                                except Exception as health_error:
                                    raise CleanupError(f'{extra} recovery health check failed: {health_error}') from health_error
                    except CleanupError:
                        raise
                    except Exception as exc:
                        item[extra] = {'status': 'partial', 'reason': str(exc)}
                        write(work / f'{extra}.json', item[extra])
                        pool.close()
                        if args.execution == 'docker':
                            try:
                                adapter.preflight(root, plan['device_ids'], label=f'health-{extra}-{index}')
                            except Exception as health_error:
                                raise CleanupError(f'{extra} recovery health check failed: {health_error}') from health_error
                        if extra == 'memory':
                            item['profiling'] = {'status': 'skipped', 'reason': 'memory execution error; no extra device replay'}
                        break
                    finally:
                        # Isolate only the supplementary worker; preserve ordinary worker reuse.
                        pool.close(mode=extra)
            except CleanupError:
                item.update(status='failed', diagnostic_execution_blocked=True);raise
            except (KeyboardInterrupt, SystemExit):
                item.update(status='interrupted', diagnostic_execution_blocked=True);raise
            except Exception as exc:
                diagnosis = adapter.diagnose(str(exc), work)
                item.update(status='blocked' if diagnosis else 'failed', error=f'{type(exc).__name__}: {exc}',
                            diagnosis=diagnosis or 'unclassified; inspect phase logs', failure_stage=item['phase'])
                if not diagnosis:
                    pool.close()
                    if ran_device and args.execution == 'docker':
                        adapter.preflight(root, plan['device_ids'], label=f'health-after-{index}')
            finally:
                for extra in ('memory', 'profiling'):
                    if extra not in item:
                        item[extra] = {'status': 'off' if extra == 'profiling' and task.get('profiling_mode', 'off') == 'off' else 'skipped',
                                       'reason': f'original task {item["status"]}; no supplemental device execution'}
                    if not (work / f'{extra}.json').exists():
                        write(work / f'{extra}.json', item[extra])
                try:
                    stage(args, 'failure-diagnostics')
                    from runtime.diagnostics import collect
                    collect(pool, work, task, item, identity=summary)
                finally:
                    item['elapsed_s'] = time.monotonic() - started
                    item['soft_target_exceeded'] = item['elapsed_s'] > 120
                    write(work / 'result.json', item)
                    write(root / 'summary.json', summary)
                    progress.done(item['status'])
            print(f'[{index+1}/{len(plan["tasks"])}] {task["case"]} {task["dtype"]} {task["oplib"]}: {item["status"]} ({item["elapsed_s"]:.1f}s)', flush=True)
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if pool is not None:
            try: pool.close()
            except Exception as exc: summary['cleanup_error'] = str(exc)
        if ran_device and args.execution == 'docker':
            stage(args, 'postflight')
            try: summary['postflight'] = adapter.preflight(root, plan['device_ids'], label='postflight')
            except (Exception, SystemExit) as exc: summary['postflight_error'] = str(exc)
        lease.release()
        # Unstarted work is explicit and remains part of the coverage denominator.
        for task in plan['tasks'][len(summary['tasks']):]:
            summary['tasks'].append({**task, 'status': 'not-run'})
        statuses = [t['status'] for t in summary['tasks'] if t['status'] != 'not-applicable']
        summary['status'] = 'failed' if summary.get('error') or summary.get('postflight_error') or summary.get('cleanup_error') or 'failed' in statuses else 'passed' if statuses and all(s == 'passed' for s in statuses) else 'partial'
        summary['elapsed_s'] = time.monotonic() - began
        summary['counts'] = {s:sum(t['status']==s for t in summary['tasks']) for s in sorted({t['status'] for t in summary['tasks']})}
        from runtime.metrics import comparisons
        summary['comparisons'] = comparisons(summary)
        stage(args, 'seal-evidence')
        write(root / 'summary.json', summary);seal(root)
        stage(args, 'report')
        report(root)
        print(f'{summary["status"]}', flush=True)
    return {'passed':0, 'partial':2, 'failed':1}[summary['status']]


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'list':
            names = catalog.names()
            if args.names_only:
                for name in names:
                    print(name)
            elif args.case is not None:
                if args.case not in names:
                    raise ValueError(f'unknown case: {args.case!r}')
                print(f'{args.case}: {" ".join(catalog.dtypes(args.case))}')
            else:
                print(json.dumps([{'case': n, 'dtypes': catalog.dtypes(n)} for n in names], indent=2))
            return 0
        if args.command == 'report':
            if len(args.run_dir) > 1 or args.output_dir is not None:
                if args.output_dir is None: raise ValueError('multiple sources require --output-dir')
                print(merge_runs(args.run_dir,args.output_dir))
            else: print(report(args.run_dir[0]))
            return 0
        if args.command == 'diagnose':
            from runtime.diagnosis import run
            return run(args)
        adapter, plan = resolve(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2));return 0
        return execute(args, adapter, plan)
    except KeyboardInterrupt:
        print('KeyboardInterrupt: operation interrupted', file=sys.stderr);return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr);return 1
    finally:
        root = getattr(args, '_result_root', None)
        if root is not None:
            # tracked() has stopped/joined the heartbeat before this final output.
            try:
                if (root / 'report.md').is_file():
                    print(f'Report: {root / "report.md"}', flush=True)
                if (root / 'summary.json').is_file():
                    print(f'Summary: {root / "summary.json"}', flush=True)
                print(f'Results: {root}', flush=True)
            except (OSError, ValueError):
                pass
            del args._result_root
