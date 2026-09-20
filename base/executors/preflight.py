"""Case-independent, bounded provider qualification through the unified facade."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

from executors.common import (BASE_DIR, BaseRunContext, ConfigurationError, DeviceLease, load_host_config,
                              runtime_artifact_paths, runtime_lock_record, validate_runtime_identity, utc_now, write_json)
from executors.lifecycle import HostCommands, ManagedContainer
from base.vendors.registry import get_provider


def add_cli_arguments(parser):
    parser.add_argument('--config', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--physical-device-ids')
    selection.add_argument('--npu-ids')
    selection.add_argument('--device-ids')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-candidate-runtime', action='store_true')
    parser.add_argument('--probe-mode', choices=('identity', 'timeout-check'), default='identity')
    parser.add_argument('--monitor', choices=('on', 'off'), default='on')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--privilege-command', default='')
    parser.add_argument('--reservation-end')
    parser.add_argument('--reservation-reference')
    parser.add_argument('--result-dir', type=Path)


def plan(args):
    context = BaseRunContext(config=args.config, physical_device_ids=args.physical_device_ids,
                             npu_ids=args.npu_ids, device_ids=args.device_ids, dry_run=args.dry_run,
                             timeout=args.timeout)
    _, config = load_host_config(args.config)
    provider = get_provider(config['vendor'])
    if not getattr(provider, 'supports_preflight', False):
        raise ConfigurationError('vendor has no standalone preflight contract')
    provider.validate_selection(context)
    selection = context.selection_request()
    if len(selection['requested_ids']) != 1:
        raise ConfigurationError('PR2 standalone hardware qualification requires exactly one physical device')
    if not set(selection['requested_ids']).issubset(config['expected_device_ids']):
        raise ConfigurationError('selected device is outside configured inventory')
    if not 30 <= args.timeout <= 180:
        raise ConfigurationError('probe timeout must be 30..180 seconds')
    try:
        HostCommands(args.privilege_command)  # validation only, no command executed
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    lock = runtime_lock_record(config.get('runtime_profile'), vendor=provider.name)
    if config['image'] != lock['image_manifest']['image']:
        raise ConfigurationError('host image differs from locked image')
    return {'schema_version': 1, 'kind': 'benchmark-preflight', 'vendor': provider.name,
            'mode': 'static-dry-run' if args.dry_run else 'execution', 'runtime_lock': lock,
            'selection_request': selection, 'host_identity': 'deferred-until-host-preflight',
            'runtime_bindings': 'deferred-until-container-UUID-probe', 'probe_mode': args.probe_mode,
            'permissions': provider.container_policy(config), 'monitoring': provider.monitor_policy(args.monitor == 'on'),
            'stages': ['image-identity', 'host-preflight', 'lease', 'locked-preflight', 'container-inspect',
                       'container-probe', 'cleanup', 'host-postflight', 'lease-release', 'report'],
            'measurement_status': 'not-run'}, config, provider


def reservation(args):
    if not args.reservation_end or not args.reservation_reference or not args.reservation_reference.strip():
        raise ConfigurationError('real preflight requires reservation-end and reservation-reference')
    try:
        end = datetime.fromisoformat(args.reservation_end.replace('Z', '+00:00'))
        if end.tzinfo is None:
            raise ValueError('timezone missing')
        remaining = (end - datetime.now(timezone.utc)).total_seconds()
    except ValueError as exc:
        raise ConfigurationError('invalid reservation end with timezone') from exc
    if remaining < args.timeout + 30:
        raise ConfigurationError('reservation must cover execution plus 30 seconds cleanup')
    return end, time.monotonic() + remaining


def render_report(root):
    summary = json.loads((root / 'summary.json').read_text())
    if summary.get('kind') != 'benchmark-preflight' or summary.get('schema_version') != 1:
        raise ConfigurationError('unsupported preflight evidence schema')
    lines = ['# Device preflight qualification', '',
             'Identity, bounded tensor readback and observation only. No performance measurement.', '']
    for key in ('run_id', 'vendor', 'status', 'probe_mode', 'failure_stage', 'error', 'measurement_status',
                'monitoring_status', 'cleanup_status', 'postflight_status', 'lease_released'):
        lines.append(f'- {key}: {str(summary.get(key, "not recorded")).replace(chr(10), " ")}')
    lines += ['', '## Verified runtime bindings', '', '```json',
              json.dumps(summary.get('device_bindings', []), sort_keys=True, indent=2), '```', '', '## Evidence', '']
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'report.md':
            relative = path.relative_to(root).as_posix()
            lines.append(f'- [{relative}]({relative})')
    content = '\n'.join(lines) + '\n'
    (root / 'report.md').write_text(content, encoding='utf-8')
    return {'status': 'passed', 'sha256': hashlib.sha256(content.encode()).hexdigest()}


def execute(args):
    static, config, provider = plan(args)
    if args.dry_run:
        print(json.dumps(static, indent=2, sort_keys=True))
        return 0
    if not args.allow_candidate_runtime and static['runtime_lock']['image_manifest']['validated'] is not True:
        raise ConfigurationError('candidate runtime requires --allow-candidate-runtime')
    end, absolute_deadline = reservation(args)
    if args.result_dir is None:
        raise ConfigurationError('real preflight requires a unique --result-dir')
    root = args.result_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'control').mkdir()
    (root / 'artifacts').mkdir(mode=0o770)
    (root / 'artifacts').chmod(0o770)
    run_id = 'preflight-' + uuid.uuid4().hex
    summary = {'schema_version': 1, 'kind': 'benchmark-preflight', 'run_id': run_id, 'vendor': provider.name,
               'probe_mode': args.probe_mode, 'status': 'running', 'started_at': utc_now(),
               'measurement_status': 'not-run', 'monitoring_status': 'not-run', 'lease_released': False}
    write_json(root / 'summary.json', summary)
    write_json(root / 'static-plan.json', static)
    commands = HostCommands(args.privilege_command, deadline=absolute_deadline - 30)
    container = ManagedContainer(commands, root, 'flagperf-' + run_id, run_id)
    lease, monitor, host = None, None, None
    stage = 'image-identity'
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        image_id = static['runtime_lock']['image_manifest']['image_id']
        inspection = commands.checked(['docker', 'image', 'inspect', image_id], privileged=True, evidence=root / 'image-inspect-command.json')
        records = json.loads(inspection['stdout'])
        if len(records) != 1:
            raise RuntimeError('image inspect must describe one image')
        identity = {k: records[0].get(k) for k in ('Id', 'RepoDigests', 'Architecture', 'Created')}
        validate_runtime_identity(config, identity, allow_candidate=args.allow_candidate_runtime)
        manifest = json.loads(runtime_artifact_paths(config.get('runtime_profile'), vendor=provider.name)[1].read_text())
        provider.validate_image(manifest, identity)
        write_json(root / 'image-identity.json', identity)
        stage = 'host-preflight'
        selected = static['selection_request']['requested_ids']
        host = provider.inspect_host(config, selected, commands, root / 'host-preflight')
        stage = 'lease'
        lease = DeviceLease([], run_id=run_id, kind='benchmark-preflight', **provider.lease_spec(host))
        lease.acquire()
        write_json(root / 'lease.json', lease.record())
        stage = 'locked-preflight'
        locked = provider.inspect_host(config, selected, commands, root / 'locked-preflight')
        provider.check_identity(host, locked)
        if absolute_deadline - time.monotonic() < args.timeout + 30:
            raise RuntimeError('insufficient reservation remaining before container start')
        context = {'schema_version': 1, 'kind': 'benchmark-preflight', 'run_id': run_id, 'host': locked,
                   'image_identity': identity, 'reservation_start': utc_now(), 'reservation_end': end.isoformat(),
                   'reservation_reference': args.reservation_reference, 'probe_mode': args.probe_mode,
                   'timeout': args.timeout, 'allow_candidate_runtime': args.allow_candidate_runtime}
        write_json(root / 'control/host-context.json', context)
        context_hash = hashlib.sha256((root / 'control/host-context.json').read_bytes()).hexdigest()
        arguments, expected = provider.probe_spec(BASE_DIR, root, config, locked, image_id)
        write_json(root / 'resolved-plan.json', {**static, 'host': locked, 'runtime_identity': identity,
                                                'container_arguments': arguments, 'expected_container': expected})
        write_json(root / 'code-identity.json', {'git_head': commands.checked(['git', '-C', str(BASE_DIR.parent), 'rev-parse', 'HEAD'])['stdout'].strip(),
                   'git_status': commands.checked(['git', '-C', str(BASE_DIR.parent), 'status', '--porcelain'])['stdout'],
                   'source_sha256': {str(p.relative_to(BASE_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for subdir in ('executors', 'vendors/kunlunxin', 'monitoring')
                                     for p in sorted((BASE_DIR / subdir).glob('*.py'))},
                   'context_sha256': context_hash})
        stage = 'container-inspect'
        container.create(arguments, expected)
        if absolute_deadline - time.monotonic() < args.timeout + 30:
            raise RuntimeError('insufficient reservation remaining after create')
        stage = 'container-probe'
        if args.monitor == 'on':
            monitor = provider.create_monitor(provider.monitor_targets(host))
            monitor.commands.deadline = absolute_deadline - 30
            monitor.start()
        container.start()
        returncode = container.wait(min(time.monotonic() + args.timeout, absolute_deadline - 30))
        summary['container_exit_code'] = returncode
        if returncode != 0:
            raise RuntimeError('container probe exited nonzero: ' + str(returncode))
        probe = json.loads((root / 'artifacts/probe.json').read_text())
        runtime_result = json.loads((root / 'artifacts/runtime-bindings.json').read_text())
        if probe.get('status') != 'passed' or probe.get('run_id') != run_id or runtime_result.get('run_id') != run_id:
            raise RuntimeError('probe result identity/status mismatch')
        summary['device_bindings'] = [b.record() for b in provider.resolve_bindings(host, context_hash, runtime_result)]
        if probe.get('context_sha256') != context_hash or probe.get('binding') != summary['device_bindings'][0]:
            raise RuntimeError('probe binding mismatch')
        summary['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='failed', failure_stage=stage, error=str(exc), error_type=type(exc).__name__)
    finally:
        try:
            # Cleanup is always bounded, including an interrupted/timed-out create.
            commands.deadline = min(absolute_deadline, time.monotonic() + 30)
            if monitor is not None:
                try:
                    monitor.stop()
                    windows = []
                    observation_path = root / 'artifacts/probe-observation.json'
                    if observation_path.is_file():
                        event = json.loads(observation_path.read_text())
                        if event.get('run_id') == run_id and type(event.get('finished_monotonic_ns')) is int:
                            windows = [{**event, 'role': 'probe-observation',
                                        'started_offset_s': event['started_monotonic_ns'] / 1e9 - monitor.origin_monotonic_s,
                                        'finished_offset_s': event['finished_monotonic_ns'] / 1e9 - monitor.origin_monotonic_s}]
                    result = monitor.finish(root, root / 'monitor', windows, [], primary_role='probe-observation',
                                            window_semantics='bounded identity probe observation, not performance measurement')
                    summary['monitoring_status'] = result['status']
                    if result['status'] != 'passed' and summary['status'] == 'passed':
                        summary['status'] = 'partial'
                except (Exception, KeyboardInterrupt) as exc:
                    summary.update(monitoring_status='failed', monitor_error=str(exc))
                    if summary['status'] == 'passed':
                        summary['status'] = 'partial'
            cleanup = container.cleanup()
            # One bounded retry if a transient Docker error left cleanup unknown.
            if cleanup['container_absent'] is not True and commands.deadline - time.monotonic() >= 8:
                write_json(root / 'cleanup-attempt01.json', cleanup)
                cleanup = container.cleanup()
            write_json(root / 'cleanup.json', cleanup)
            summary['cleanup_status'] = cleanup['status']
            if cleanup['container_absent'] is not True:
                summary.update(status='failed', recovery_required=True,
                               recovery_note='Container absence unconfirmed. Process exit releases flock; do not retry until recovery.')
            if lease is not None and lease.acquired_at is not None and cleanup['container_absent'] is True:
                try:
                    post = provider.inspect_host(config, selected, commands, root / 'host-postflight')
                    provider.check_identity(host, post)
                    summary['postflight_status'] = 'passed'
                except (Exception, KeyboardInterrupt) as exc:
                    summary.update(status='failed', postflight_status='failed', postflight_error=str(exc))
            else:
                summary['postflight_status'] = 'not-run'
        except (Exception, KeyboardInterrupt) as exc:
            summary.update(status='failed', cleanup_status='failed', recovery_required=True,
                           finalization_error=str(exc), recovery_note='Cleanup evidence incomplete; process exit releases flock.')
        finally:
            try:
                if lease is not None:
                    lease.release()
                    summary['lease_released'] = True
                    record = lease.record()
                    record['released_at'] = utc_now()
                    write_json(root / 'lease.json', record)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        summary['finished_at'] = utc_now()
        write_json(root / 'summary.json', summary)
        try:
            render_report(root)
        except Exception as exc:
            summary['report_error'] = str(exc)
            write_json(root / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'result_dir': str(root)}, indent=2))
    return 1 if summary['status'] == 'failed' or summary.get('report_error') else 2 if summary['status'] == 'partial' else 0
