"""Current-source, CPU-only postprocessing. Historical code is never imported."""
from pathlib import Path
import json
import os
import sys
import time
import resource

from runtime.common import read_json, write_json, file_hash, digest, ROOT


def stage(root, phase, **counts):
    value = {'phase':phase,'new_device_execution':False,'updated_unix_seconds':time.time(),
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**counts}
    write_json(Path(root)/'recovery-progress.json',value)
    print('[offline] '+json.dumps(value,ensure_ascii=False),file=sys.stderr,flush=True)


def copy_run(source, output):
    from runtime.reference_storage import run_members, copy_verified
    from runtime.preflight import capacity
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('offline output must be a new non-overlapping directory')
    entries = [row for row in run_members(source) if row[0] not in ('recovery-progress.json', 'recovery-failure.json')]
    physical = {}
    required = 0
    for _, path, _ in entries:
        stat = path.stat(); key = (stat.st_dev, stat.st_ino)
        if key not in physical: required += stat.st_size; physical[key] = None
    if required + 256*1024*1024 > capacity(output)['usable_bytes']:
        raise ValueError('insufficient capacity for portable offline recovery')
    output.mkdir(parents=True)
    (output/'.artifact-root').write_text('1\n')
    physical.clear()
    inventory = {}
    detail_names = {'result.json', 'summary.json', 'attribution.json', 'comparison.json',
                    'oracle-comparison.json', 'performance-comparison.json',
                    'candidate-performance-extract.json', 'a100-performance-extract.json', 'collectives.json'}
    copied_bytes = 0; started = updated = time.monotonic()
    stage(output, 'copying', files_total=len(entries), source_bytes=required)
    for number, (name, path, _) in enumerate(entries):
        stat = path.stat(); key = (stat.st_dev, stat.st_ino)
        target = output/name
        if key in physical:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(output/physical[key], target.parent))
            inventory[name] = {'alias': physical[key]}
        else:
            if path.name in detail_names and path.stat().st_size > 32*1024*1024:
                from runtime.artifacts import Store
                from runtime.stream_json import read_document
                before = file_hash(path)
                value = read_document(path, Store(output))
                write_json(target, value)
                if file_hash(path) != before: raise ValueError('source changed during offline copy')
                inventory[name] = {'source_sha256': before, 'source_bytes': stat.st_size,
                                   'sha256': file_hash(target), 'bytes': target.stat().st_size, 'storage_version': 1}
            else:
                inventory[name] = copy_verified(path, target)
            physical[key] = name
            copied_bytes += stat.st_size
        if time.monotonic()-updated >= 5:
            stage(output, 'copying', files_completed=number+1, files_total=len(entries),
                  source_bytes_read=copied_bytes, elapsed_seconds=time.monotonic()-started)
            updated = time.monotonic()
    write_json(output/'recovery-source.json', {'source': str(source), 'files': inventory,
               'new_device_execution': False, 'historical_code_executed': False})
    return output


def analysis_identity():
    from runtime.common import ANALYSIS_FILES
    names = set(ANALYSIS_FILES) | {'vendors/nvidia/trace.py', 'vendors/nvidia/trace_index.py',
             'vendors/nvidia/scope_index.py', 'vendors/ascend/trace.py', 'runtime/common.py',
             'runtime/artifacts.py', 'runtime/record_index.py', 'runtime/tensor_io.py',
             'runtime/reference_protocol.py', 'runtime/workload.py', 'runtime/input_layout.py'}
    # Adapters also select boundaries and validate contracts during CPU analysis.
    for directory in ('runtime', 'benchmarks', 'vendors', 'registry'):
        names.update(path.relative_to(ROOT).as_posix() for path in (ROOT/directory).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.json'))
    return {name: file_hash(ROOT/name) for name in sorted(names) if (ROOT/name).is_file()}


def profile_key(folder, options):
    inputs = {}
    from runtime.artifacts import Records
    def dependencies(value):
        if isinstance(value, Records):
            entry = value.store.entry(value.artifact_id)
            for shard in entry['shards']:
                path = value.store.directory/shard['path']
                actual = file_hash(path)
                if actual != shard['sha256']: raise ValueError('capture shard digest mismatch')
                inputs['artifact:'+value.artifact_id+':'+shard['path']] = actual
        elif isinstance(value, dict):
            for child in value.values(): dependencies(child)
        elif isinstance(value, list):
            for child in value: dependencies(child)
    for name in ('result.json', 'collectives.json'):
        path = folder/name
        if path.is_file():
            inputs[name] = file_hash(path)
            dependencies(read_json(path))
    for path in sorted((folder/'profiler').rglob('*')):
        if path.is_file(): inputs[path.relative_to(folder).as_posix()] = file_hash(path)
    return digest({'inputs': inputs, 'analysis': analysis_identity(), 'options': options})


def restore_profile(folder, key):
    checkpoint = folder/'analysis-checkpoint.json'
    output = folder/'attribution.json'
    if not checkpoint.is_file() or not output.is_file(): return None
    from runtime.checkpoints import verify_records
    try:
        record = read_json(checkpoint)
        if record.get('key') != key or record.get('sha256') != file_hash(output): return None
        value = read_json(output)
        if value.get('status') != 'completed' or value.get('communication', {}).get('status', 'completed') != 'completed': return None
        verify_records(value)
        return value
    except (OSError, ValueError, KeyError):
        return None  # Derived output can be rebuilt from the verified capture.


def checkpoint_profile(folder, key, value):
    if value.get('status') == 'completed' and value.get('communication', {}).get('status', 'completed') == 'completed':
        write_json(folder/'analysis-checkpoint.json', {'key': key, 'sha256': file_hash(folder/'attribution.json')})


def reanalyze_communication(root, result, cfg, checkpoints, timeout):
    from runtime.communication_analysis import analyze as aggregate, analyze_rank
    size = len(cfg['runtime']['devices'])
    dependencies = []
    for side, repeats in result.get('profiles', {}).items():
        for label, state in repeats.items():
            folder = root/'profiles'/side/label
            key = digest([profile_key(folder/f'rank-{rank}', {'rank':rank,'world_size':size})
                          for rank in range(size)])
            output = (folder/'communication.json').relative_to(root).as_posix()
            node = 'communication/'+side+'/'+label
            if checkpoints.reusable(node, key, [output]):
                communication = read_json(root/output)
            else:
                rows = {}
                for rank in range(size):
                    capture = read_json(folder/f'rank-{rank}'/'result.json')
                    if capture.get('status') != 'completed':
                        rows[str(rank)] = {'status':'partial','rank':rank,'issues':['capture incomplete'],'events':[]}
                    else:
                        rows[str(rank)] = analyze_rank(folder/f'rank-{rank}',rank,timeout=timeout)
                communication = aggregate(folder,size,rank_results=rows)
                if communication['status'] == 'completed': checkpoints.save(node,key,[output])
            state['communication'] = communication
            if communication['status'] != 'completed': result['status'] = 'partial'
            dependencies.append({'node':node,'key':key,'status':communication['status']})
    return dependencies


def _analyze(source, output, timeout=None):
    # CPU-only postprocessing must not auto-load accelerator plugins.
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import yaml
    source = Path(source).resolve()
    cfg = yaml.safe_load((source/'effective.yaml').read_text())
    if cfg.get('_command') in ('accuracy', 'oracle'):
        try:
            import torch
            import numpy
        except ImportError as error:
            raise ValueError('offline accuracy requires CPU PyTorch and NumPy; run this command in the configured inference image without accelerator mounts') from error
    root = copy_run(source, output)
    from runtime.tensor_io import session
    from runtime.verification import task
    from runtime.checkpoints import Checkpoints, capture_key
    with session(root) as tensor_reads, task(root):
        # Re-open references against the copied inventory, never the source catalog.
        result = read_json(root/'result.json')
        if result.get('command') in ('accuracy', 'oracle'):
            try:
                import torch
                import numpy
            except ImportError as error:
                raise ValueError('offline accuracy requires CPU PyTorch and NumPy in the configured inference image') from error
        source_status = result.get('status')
        source_error = result.get('error')
        capture_status = result.get('execution_status', source_status)
        result['status'] = capture_status
        if capture_status == 'completed':
            result.pop('error', None); result.pop('traceback', None)
        stage(root, 'validating_dependencies')
        checkpoints = Checkpoints(root)
        key = capture_key(root, result, cfg, analysis_identity())
        result['report_view'] = {'source': str(source), 'source_status': source_status, 'new_device_execution': False}
        if result.get('command') == 'performance':
            if 'layer' in result:
                stage(root, 'analyzing_profiles')
                from runtime.report_replay import reanalyze_profiles
                analysis = reanalyze_profiles(root, result, timeout or cfg['runtime'].get('analysis_timeout_seconds', 900))
                result['analysis_status'] = analysis['status']
                if analysis['status'] != 'completed' and result.get('status') == 'completed': result['status'] = 'partial'
                key = digest({'capture': key, 'profiles': [{k: row.get(k) for k in ('side', 'repeat', 'rank', 'input_key', 'status')}
                                                          for row in analysis['records']]})
            elif cfg['runtime'].get('parallelism') == 'tp' and result.get('profiles'):
                dependencies = reanalyze_communication(root, result, cfg, checkpoints,
                    timeout or cfg['runtime'].get('analysis_timeout_seconds', 900))
                key = digest({'capture':key,'communication':dependencies})
            reference = cfg.get('reference', {}).get('performance_path')
            outputs = ['a100-performance-extract.json', 'candidate-performance-extract.json', 'performance-comparison.json']
            if reference and result.get('status') == 'completed' and not checkpoints.reusable('performance_comparison', key, outputs):
                stage(root, 'comparing_performance')
                from runtime.performance_baseline import load_manifest
                from runtime.performance_comparison import extract, compare
                manifest = load_manifest(reference)
                baseline = read_json(Path(reference)/'metrics.json')
                candidate = extract(root, result, cfg)
                comparison = compare(baseline, candidate, manifest['contract'])
                comparison.update(baseline_id=manifest['baseline_id'], contract=manifest['contract'])
                write_json(root/'a100-performance-extract.json', baseline)
                write_json(root/'candidate-performance-extract.json', candidate)
                write_json(root/'performance-comparison.json', comparison)
                checkpoints.save('performance_comparison', key, outputs)
        elif result.get('command') == 'accuracy':
            if {'off', 'on'} <= result.get('paths', {}).keys() and not checkpoints.reusable('same_device_comparison', key, ['comparison.json']):
                stage(root, 'comparing_same_device')
                from runtime.coordinator import accuracy_diff
                contexts = None
                if (root/'comparison-context.json').exists():
                    contexts = {side: {'key': key} for side, key in read_json(root/'comparison-context.json')['identities'].items()}
                diff = accuracy_diff(cfg, root, root/'prepared', result['paths'], contexts)
                if contexts:
                    axis = read_json(root/'comparison-context.json')['comparison_component']
                    for comparison in diff['ranks'].values() if cfg['runtime'].get('parallelism') == 'tp' else [diff]:
                        comparison['reference'] = axis+' off (not an absolute oracle)'
                    diff['comparison_component'] = axis
                write_json(root/'comparison.json', diff)
                checkpoints.save('same_device_comparison', key, ['comparison.json'])
            if cfg.get('reference', {}).get('path') and not checkpoints.reusable('oracle_comparison', key, ['oracle-comparison.json']):
                stage(root, 'comparing_reference')
                from runtime.oracle_analysis import compare_run
                compare_run(cfg, root, result)
                if result.get('oracle', {}).get('status') == 'completed':
                    checkpoints.save('oracle_comparison', key, ['oracle-comparison.json'])
        elif result.get('command') == 'oracle':
            if capture_status == 'completed':
                from runtime.oracle import publish_capture
                repeat = read_json(root/'verification/result.json')
                publish_capture(cfg, root, result, repeat)
                checkpoints.executed.append('reference_publication')
        else:
            raise ValueError('analyze requires a saved accuracy, oracle or performance capture')
        if capture_status != 'completed': result['status'] = source_status
        if source_status != 'completed' and result.get('assessment', {}).get('status') in ('needs_review', 'invalid_evidence'):
            result['status'] = source_status
        if result.get('status') == 'completed': result['analysis_status'] = 'completed'
        result['offline_recovery'] = {'new_device_execution': False, 'source_status': source_status, 'source_error': source_error,
                                      'analysis_sources': analysis_identity(),
                                      'tensor_source_reads': tensor_reads.reads, 'tensor_cache_hits': tensor_reads.hits,
                                      'tensor_spilled_bytes': tensor_reads.spilled_bytes,
                                      'verification': __import__('runtime.verification', fromlist=['statistics']).statistics(),
                                      'reused_nodes': checkpoints.reused, 'executed_nodes': checkpoints.executed}
        if (result.get('command') == 'performance' and cfg.get('_performance_capture')
                and result.get('status') == 'completed' and result.get('analysis_status') == 'completed'):
            from runtime.performance_baseline import publish
            publish(cfg, root, result)
            checkpoints.executed.append('reference_publication')
        write_json(root/'result.json', result)
        stage(root, 'finished', status=result['status'])
    return root


def analyze(source, output, timeout=None):
    """Publish an offline result only after all requested nodes have finished."""
    import uuid
    if timeout is not None and (type(timeout) is not int or timeout < 1):
        raise ValueError('analysis timeout must be a positive integer')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('offline output must be a new non-overlapping directory')
    stage = output.with_name(output.name+'.incomplete-'+uuid.uuid4().hex)
    try:
        _analyze(source, stage, timeout)
        stage.rename(output)
        return output
    except BaseException as error:
        if stage.is_dir():
            # Never leave a copied completed result masquerading as recovered output.
            failure = {'status': 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                       'error': str(error), 'source': str(source), 'new_device_execution': False,
                       'published': False}
            try:
                write_json(stage/'recovery-failure.json', failure)
                result = read_json(stage/'result.json') if (stage/'result.json').is_file() else {}
                result.update(status='failed', analysis_status=failure['status'], offline_recovery=failure)
                write_json(stage/'result.json', result)
            except OSError:
                pass  # A full or failed filesystem must not turn the failure into success.
        raise


def main(args):
    if not args.source or not args.output:
        raise ValueError(args.command + ' requires --source and --output')
    if args.dry_run: raise ValueError('--dry-run applies to capture commands')
    from runtime.config import parser
    defaults = vars(parser().parse_args([args.command]))
    allowed = {'command', 'source', 'output'}
    if args.command == 'analyze':
        allowed.add('analysis_timeout_seconds')
        if args.analysis_timeout_seconds is not None and args.analysis_timeout_seconds < 1:
            raise ValueError('--analysis-timeout-seconds must be a positive integer')
    ignored = [name for name, value in vars(args).items()
               if name not in allowed and value != defaults[name]]
    if ignored:
        raise ValueError(args.command + ' does not accept: ' +
                         ', '.join('--' + name.replace('_', '-') for name in ignored))
    if args.command == 'repack-reference':
        from runtime.reference_storage import repack
        result = repack(args.source, args.output)
        print(json.dumps({'output': args.output, 'schema_version': result['schema_version'],
                          'capabilities': result['capabilities']}, ensure_ascii=False))
    else:
        output = analyze(args.source, args.output, args.analysis_timeout_seconds)
        print('Results: ' + str(output))
        return 0 if read_json(output/'result.json').get('status') == 'completed' else 1
    return 0
