"""CPU-only estimates, explicitly separate from a verified execution identity."""
import json
from pathlib import Path
import shutil


def reference_metadata(cfg, performance=False):
    from runtime.common import read_json, digest
    from runtime.oracle import safe_file
    root = Path(cfg['reference']['performance_path' if performance else 'path'])
    manifest = read_json(safe_file(root, 'manifest.json'))
    key = 'baseline_id' if performance else 'oracle_id'
    kind = 'a100_performance' if performance else 'a100_accuracy'
    if (manifest.get('kind') != kind or manifest.get('status') != 'complete' or
            manifest.get('schema_version') not in (1, 2, 3, 4) or
            manifest.get(key) != digest({k: v for k, v in manifest.items() if k != key})):
        raise ValueError('unsupported or invalid reference manifest')
    cfg['inputs']['path'] = str(safe_file(root, 'inputs.jsonl'))
    cfg['_reference_preflight'] = {'id': manifest[key], 'schema_version': manifest['schema_version'],
        'declared_bytes': sum(info['bytes'] for info in manifest['files'].values()),
        'full_sha256_and_contract': 'pending', 'verified': False}


def capacity(path):
    path = Path(path).resolve()
    while not path.exists(): path = path.parent
    usage = shutil.disk_usage(path)
    filesystem, longest = 'unknown', 0
    mounts = Path('/proc/self/mountinfo')
    if mounts.exists():
        for line in mounts.read_text().splitlines():
            fields = line.split(); mount = Path(fields[4].replace('\\040', ' '))
            if path.is_relative_to(mount) and len(str(mount)) >= longest:
                longest, filesystem = len(str(mount)), fields[fields.index('-')+1]
    available = None
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'): available = int(line.split()[1])*1024
    try:
        base = Path('/sys/fs/cgroup')
        limit = (base/'memory.max').read_text().strip()
        current = int((base/'memory.current').read_text())
        if limit != 'max': available = min(available or int(limit), max(0, int(limit)-current))
    except (OSError, ValueError): pass
    usable = min(usage.free, available) if filesystem == 'tmpfs' and available is not None else usage.free
    return {'filesystem': filesystem, 'disk_available_bytes': usage.free,
            'memory_available_bytes': available, 'usable_bytes': usable, 'path': str(path)}


def historical_storage(cfg):
    """Use declared reference sizes without decoding its observations or trace."""
    reference = cfg.get('reference', {}).get('performance_path')
    if not reference: return None
    from runtime.common import read_json
    manifest = read_json(Path(reference)/'manifest.json')
    sampling = manifest.get('sampling', {})
    old_repeats = sampling.get('repeats')
    if not isinstance(old_repeats, int) or old_repeats < 1: return None
    if manifest.get('schema_version') == 4:
        size = manifest.get('evidence_attachment', {}).get('bytes', 0)
        basis = 'reference attachment bytes (includes non-trace evidence)'
    else:
        unique = {}
        for name, info in manifest.get('files', {}).items():
            if '/profiler/' in name: unique[info['sha256']] = info['bytes']
        size = sum(unique.values())
        basis = 'unique reference profiler files'
    ratio = cfg['performance']['repeats']/old_repeats
    sides = 2 if any(cfg['performance'].get(k) == 'both' for k in ('flaggems','flagtree','flagcx')) else 1
    return {'source_reference': manifest.get('baseline_id'), 'basis': basis,
            'trace_and_evidence_bytes_estimate': int(size*ratio*sides),
            'limits': 'cross-vendor trace volume differs; estimate is not a capacity guarantee'}


def dependencies():
    from importlib.util import find_spec
    return {name: 'installed' if find_spec(name) is not None else 'missing_in_current_python'
            for name in ('yaml', 'numpy', 'torch', 'safetensors')}


def plan(cfg, command, output):
    mode = 'performance' if command == 'performance-baseline' else command
    section = cfg.get(mode, cfg['accuracy'])
    sides = 2 if any(section.get(k) == 'both' for k in ('flaggems', 'flagtree', 'flagcx')) else 1
    ranks = len(cfg['runtime']['devices']) if cfg['runtime']['parallelism'] == 'tp' else 1
    inputs = Path(cfg['inputs']['path'])
    samples = 0
    with inputs.open() as stream:
        for line in stream:
            if line.strip(): json.loads(line); samples += 1
    performance = cfg['performance']
    repeats = performance['repeats'] if mode == 'performance' else 1
    phases = ['prepare', 'forward_audit']
    if mode == 'performance':
        phases += ['performance']
        if performance['level'] == 'layer' or command == 'performance-baseline':
            phases += ['layer_timing', 'layer_profile', 'layer_memory', 'analysis']
    else:
        phases += ['forward', 'accuracy_analysis']
        if command == 'oracle': phases += ['independent_repeatability']
    phases += ['comparison', 'publication']
    workers = sides * (2 + repeats*(4 if 'layer_profile' in phases else 1))
    if command == 'oracle': workers += 1
    if command == 'preview':
        phases = ['prepare', 'baseline_audit', 'candidate_search', 'independent_verification', 'policy_publication']
        workers = None  # Search fanout depends on observed candidates and failures.
    elif command == 'export':
        phases = ['prepare', 'export']; workers = 2
    batch_size = cfg['inputs'].get('batch_size')
    batches = (samples+batch_size-1)//batch_size if isinstance(batch_size, int) and batch_size > 0 else None
    minimum = inputs.stat().st_size + 256*1024*1024
    layer_records = None
    if 'layer_timing' in phases and batches is not None:
        from runtime.adapters import model_adapter
        adapter = model_adapter(cfg)
        modules = performance['layers']
        if modules == ['all']: modules = adapter.instrumentation(cfg)
        layer_records = batches*ranks*sides*repeats*performance['measure_rounds']*len(modules)*adapter.execution_steps(cfg)
        minimum += layer_records*512  # Explicit planning estimate, before compression.
    historical = historical_storage(cfg) if mode == 'performance' else None
    # Keep both final evidence and a disk index while publishing atomically.
    scratch_estimate = historical['trace_and_evidence_bytes_estimate']*4 if historical else None
    if scratch_estimate is not None: minimum += scratch_estimate
    space = capacity(output)
    return {'status': 'estimated', 'verified': False, 'new_device_execution': False,
            'command': command, 'reference': cfg.get('_reference_preflight'), 'stages': phases, 'worker_processes_estimate': workers,
            'rank_processes_estimate': workers*ranks if workers is not None else None, 'ranks_per_worker': ranks,
            'inputs': {'samples': samples, 'batches_estimate': batches, 'bytes': inputs.stat().st_size},
            'sampling': {k: performance[k] for k in ('warmup_rounds', 'measure_rounds', 'repeats', 'layer_profile_rounds')},
            'record_count_estimate': batches*ranks*sides*repeats*performance['measure_rounds'] if batches and mode == 'performance' else None,
            'preview_budget_seconds': cfg.get('preview', {}).get('budget_seconds') if command == 'preview' else None,
            'layer_records_estimate': layer_records,
            'storage_estimation': '512 bytes per layer observation before compression; trace size remains pending',
            'minimum_workspace_bytes': minimum, 'capacity': space,
            'historical_storage': historical, 'temporary_space_estimate_bytes': scratch_estimate,
            'current_python_dependencies': dependencies(),
            'capacity_status': 'insufficient' if space['usable_bytes'] < minimum else 'minimum_budget_available',
            'pending_validation': ['accelerator availability', 'runtime dependencies', 'model loading',
                                   'profiler volume and peak temporary space'],
            'historical_duration_seconds': None}


def check(cfg, command, output):
    value = plan(cfg, command, output)
    if value['capacity_status'] == 'insufficient':
        raise ValueError('insufficient output workspace capacity: ' + json.dumps(value['capacity']))
    return value


def dry_run(args, cfg):
    value = plan(cfg, args.command, args.output or cfg['runtime']['output_root'])
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 2 if value['capacity_status'] == 'insufficient' else 0
