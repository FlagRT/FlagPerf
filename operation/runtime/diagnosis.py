# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Common evidence checks and optional same-input replay, without causal controls."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from runtime import catalog, diagnostics
from runtime.progress import tracked, stage
from runtime.diagnostics import inspect_evidence as inspect_source, numeric_observation
from runtime.evidence import read, write, sha, seal
from vendors import AVAILABLE, get_vendor

PROTOCOL = 'operation-diagnosis-v1'


def add_parser(sub):
    p = sub.add_parser('diagnose', help='inspect sealed task evidence; optionally replay the original operation')
    p.add_argument('--source-task', type=Path, required=True)
    p.add_argument('--replay', action='store_true', help='explicitly run one same-input probe; default is offline')
    p.add_argument('--vendor', choices=AVAILABLE, help='must match source; inferred when omitted')
    p.add_argument('--device-ids', help='one local device; inferred from source when omitted')
    p.add_argument('--execution', choices=['docker'], default='docker')
    p.add_argument('--image', help='replay image; must match the recorded source image ID')
    p.add_argument('--result-root', type=Path, default=catalog.ROOT / 'result')
    p.add_argument('--watchdog', type=float, default=300, help='per replay phase timeout')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--allow-privileged-root', action='store_true')
    p.add_argument('--container-device', action='append', default=[])
    p.add_argument('--mount', action='append', default=[])


def source_task(directory):
    """Verify every consumed source file; allow absent outputs on blocked tasks."""
    directory = Path(directory).resolve()
    parent = directory.parent
    manifest = read(parent / 'artifacts.json')
    if manifest.get('schema_version') != 2:
        raise ValueError('diagnosis requires sealed schema 2 task evidence')
    files = manifest['files']
    prefix = directory.name + '/'
    selected = {rel[len(prefix):]: digest for rel, digest in files.items() if rel.startswith(prefix)}
    if 'task.json' not in selected or 'summary.json' not in files:
        raise ValueError('sealed task.json and summary.json are required')
    for rel in ['summary.json', *(prefix + n for n in selected)]:
        path = (parent / rel).resolve()
        if not path.is_relative_to(parent) or (rel != 'summary.json' and not path.is_relative_to(directory)) or not path.is_file() or sha(path) != files[rel]:
            raise ValueError(f'missing/corrupt source artifact: {rel}')
    # Never read an unsealed log/JSON/tensor merely because it exists on disk.
    for p in directory.rglob('*'):
        if p.is_file() and str(p.relative_to(directory)) not in selected:
            if p.name != 'report.md' and '__pycache__' not in p.parts:
                raise ValueError(f'unsealed source artifact: {p.name}')
    summary = read(parent / 'summary.json')
    task = read(directory / 'task.json')
    if summary.get('protocol') in (PROTOCOL, 'operation-attribution-v1'):
        if directory.name != 'source-task' or task != summary.get('source_task'):
            raise ValueError('source task differs from diagnostic provenance')
        result = summary.get('source_result', {'correctness': read(directory / 'correctness.json') if 'correctness.json' in selected else {}})
        identity = summary.get('source_identity', {k: summary[k] for k in ('actual_image_id', 'runtime', 'preflight') if k in summary})
    else:
        if summary.get('protocol') != 'operation-v2' or summary.get('source_runs'):
            raise ValueError('use an original operation-v2 task, not an aggregate directory')
        matches = [t for t in summary.get('tasks', []) if t.get('directory') == directory.name]
        if len(matches) != 1 or any(matches[0].get(k) != v for k, v in task.items()):
            raise ValueError('source task differs from sealed run summary')
        result = matches[0]
        identity = {k: summary[k] for k in ('actual_image_id', 'runtime', 'preflight', 'postflight', 'vendor', 'execution', 'status', 'cleanup_error', 'postflight_error') if k in summary}
    return task, selected, result, identity


def replay(root, args, task, identity, data):
    stage(args, 'replay-preparation')
    from runtime.cli import ids, Lease, Workers
    if task.get('case') not in catalog.names() or task.get('dtype') not in catalog.dtypes(task['case']) or task.get('oplib') not in ('nativetorch', 'flaggems'):
        raise ValueError('source case/dtype/path is not supported by the current Case contract')
    adapter = get_vendor(task['vendor'])
    devices = ids(args.device_ids or str(task['device_id']))
    if len(devices) != 1:
        raise ValueError('common replay requires one local device')
    if not identity.get('actual_image_id'):
        raise ValueError('replay requires a recorded source Docker image ID')
    image = args.image or identity['actual_image_id']
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True, timeout=30))[0]
    if info['Id'] != identity['actual_image_id']:
        raise ValueError('replay image differs from source; use a separate benchmark for a changed runtime')
    # Some vendors bind policy to a locked tag as well as its immutable ID.
    # Inspect/execute the source ID, while validating against that vendor policy.
    data['replay_runtime'] = adapter.runtime_identity(adapter.default_image() or image, info)
    data['replay_image_id'] = info['Id']
    data['replay_device_ids'] = devices
    lease = Lease(adapter.lease_root(), devices, root.name)
    pool = None
    ran_device = False
    try:
        lease.acquire()
        stage(args, 'preflight')
        data['preflight'] = adapter.preflight(root, devices)
        work = root / 'replay'
        work.mkdir()
        for name in ('inputs.pt', 'reference.pt'):
            source = root / 'source-task' / name
            if source.exists(): shutil.copyfile(source, work / name)
        current = {**task, 'device_id': devices[0], 'worker_device': 0, 'common_diagnosis': True, 'diagnostics_mode': 'failures', 'diagnostic_protocol': diagnostics.PROTOCOL}
        current.pop('attribution', None)
        write(work / 'task.json', current)
        pool = Workers(root, adapter, args, info['Id'], root.name)
        phases = ['reference-check', 'probe']
        reference_row = None
        item = {'status': 'not-run'}
        for index, phase in enumerate(phases):
            if phase == 'probe': ran_device = True
            before = len(data['checks'])
            try:
                row = diagnostics.run_check(pool, work, current, phase, scope='new-replay',
                                            records=data['checks'], close_after=True)
            finally:
                for recorded in data['checks'][before:]:
                    recorded['evidence'] = ['replay/' + p for p in recorded['evidence']]
                write(root / 'summary.json', data)
            if row['execution_status'] != 'completed':
                raise RuntimeError(row['observation'])
            if phase == 'reference-check':
                reference_row = row
                if row['status'] not in ('passed', 'not-applicable'):
                    raise ValueError('reference check failed; stop before device replay')
            else:
                routing = adapter.route(read(work / 'probe.json'), (work / 'probe.log').read_text())
                write(work / 'routing.json', routing)
                item = {'status': 'completed', 'routing': routing, 'correctness': read(work / 'correctness.json')}
                data['checks'].append(diagnostics.check_record('routing', routing['status'], 'new-replay',
                    '目标调用的路径证据；不等于设备时间线。', ['replay/routing.json'], execution_status='completed'))
            print(f'[{index+1}/2] {phase}: {row["status"]} ({row["elapsed_s"]:.2f}s)', flush=True)
        try:
            stage(args, 'supplemental-diagnostics')
            diagnostics.collect(pool, work, current, item, trigger='replay', reference_row=reference_row,
                                identity={'actual_image_id': info['Id'], 'preflight': data['preflight']})
        finally:
            supplemental = item.get('diagnostics', {'checks': [], 'status': 'failed'})
            for check in supplemental['checks']:
                if check['execution_status'] != 'recorded':
                    data['checks'].append({**check, 'evidence': ['replay/' + p for p in check['evidence']]})
            write(work / 'result.json', item)
        if supplemental['status'] == 'failed':
            raise RuntimeError('supplemental checks failed; inspect replay/diagnostic-checks.json')
    finally:
        try:
            if pool: pool.close()
        except Exception as exc:
            data['cleanup_error'] = str(exc)
        try:
            stage(args, 'postflight')
            if ran_device: data['postflight'] = adapter.preflight(root, devices, label='postflight')
        except Exception as exc:
            data['postflight_error'] = str(exc)
        lease.release()


@tracked
def run(args):
    from runtime.cli import snapshot, ids
    stage(args, 'verify-source')
    task, digests, source_result, identity = source_task(args.source_task)
    if args.vendor and args.vendor != task.get('vendor'):
        raise ValueError('vendor must match the source task')
    if not math.isfinite(args.watchdog) or args.watchdog <= 0:
        raise ValueError('watchdog must be positive and finite')
    if args.device_ids and len(ids(args.device_ids)) != 1:
        raise ValueError('diagnose accepts one local replay device')
    missing = [n for n in ('inputs.pt',) if n not in digests]
    if task.get('case') not in ('dropout', 'native_dropout') and 'reference.pt' not in digests:
        missing.append('reference.pt')
    if args.replay:
        from runtime.worker import tolerances
        saved_gate = read(args.source_task / 'correctness.json') if 'correctness.json' in digests else {}
        atol, rtol = tolerances(task.get('dtype'))
        if saved_gate.get('atol', atol) != atol or saved_gate.get('rtol', rtol) != rtol:
            raise ValueError('source tolerance differs from the current comparison protocol')
    if args.replay and missing:
        raise ValueError('replay prerequisites missing: ' + ', '.join(missing) + '; offline diagnosis is available')
    plan = {'protocol': PROTOCOL, 'mode': 'replay' if args.replay else 'offline', 'source_task': task,
            'checks': ['integrity', 'inputs', 'reference', 'correctness', 'routing', 'execution', 'runtime'],
            'replay_phases': ['reference-check', 'probe', 'diagnose', 'trace (only if route remains partial)'] if args.replay else [],
            'check_protocol': diagnostics.CHECK_PROTOCOL,
            'boundary': 'common checks only; no specialized causal controls or automatic repair',
            'watchdog_per_phase_s': args.watchdog, 'replay_device': args.device_ids or task.get('device_id'),
            'replay_image': args.image or identity.get('actual_image_id')}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2));return 0
    display_task = dict(task)
    if args.replay and args.device_ids:
        display_task['device_id'] = ids(args.device_ids)[0]
    args._progress.task(display_task)
    root = args.result_root.resolve() / ('diagnosis-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True)
    args._result_root = root
    began = time.monotonic()
    data = {'schema_version': 2, 'protocol': PROTOCOL, 'status': 'partial', 'execution_status': 'running',
            'mode': plan['mode'], 'check_protocol': diagnostics.CHECK_PROTOCOL, 'source_task': task, 'source_result': source_result, 'source_identity': identity,
            'source_directory': str(args.source_task.resolve()), 'source_hashes': digests,
            'attribution': {'status': 'not-performed', 'reason': '当前仅执行公共检查，不运行专项归因。'},
            'checks': []}
    interrupted = False
    try:
        stage(args, 'snapshot-and-copy')
        write(root / 'plan.json', plan)
        snapshot(root)
        source = root / 'source-task';source.mkdir()
        for name, digest in digests.items():
            destination = source / name;destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source_task / name, destination)
            if sha(destination) != digest: raise ValueError('source changed while copying: ' + name)
        for name in ('artifacts.json', 'summary.json'):
            shutil.copyfile(args.source_task.resolve().parent / name, root / ('source-' + name))
        write(root / 'source-result.json', source_result)
        write(root / 'source-identity.json', identity)
        stage(args, 'offline-checks')
        data['checks'] = inspect_source(source, task, source_result, identity, digests)
        if args.replay: replay(root, args, task, identity, data)
        data['execution_status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        data.update(execution_status='failed', error=f'{type(exc).__name__}: {exc}')
    errors = any(data.get(k) for k in ('error', 'cleanup_error', 'postflight_error'))
    if errors: data['execution_status'] = 'failed'
    data['status'] = 'failed' if errors else 'partial' if any(c['status'] in ('missing', 'partial', 'not-run', 'unsupported') for c in data['checks']) else 'passed'
    data['elapsed_s'] = time.monotonic() - began
    args._progress.done('interrupted' if interrupted else data['execution_status'])
    stage(args, 'seal-evidence')
    write(root / 'summary.json', data);seal(root)
    stage(args, 'report')
    render(root, data)
    print(f'公共诊断：{data["execution_status"]}；覆盖：{data["status"]}；专项归因：未执行', flush=True)
    return {'passed': 0, 'partial': 2, 'failed': 1}[data['status']]


def render(root, data):
    from runtime.reporting import diagnosis_report
    lines = diagnosis_report(root, data)
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return root / 'report.md'
