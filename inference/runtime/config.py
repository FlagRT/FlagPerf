"""One validated configuration shared by host and container entrypoints."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from runtime.common import ROOT
from runtime.adapters import descriptor, model_adapter, vendor_adapter, require_capability


DEFAULT_CONFIG = ROOT / 'config' / 'default.yaml'
PATH_FIELDS = {('model', 'path'), ('inputs', 'path'), ('runtime', 'output_root'),
               ('policy', 'path'), ('accuracy', 'data_path'), ('preview','resume_from'), ('reference','path'), ('reference','performance_path')}


class ConfigError(ValueError):
    pass


def parser():
    p = argparse.ArgumentParser(description='Inference accuracy and performance')
    p.add_argument('command', choices=['accuracy', 'performance', 'preview', 'export', 'list', 'report', 'oracle', 'performance-baseline', 'analyze', 'repack-reference'])
    p.add_argument('--dry-run', action='store_true', help='Estimate stages and cost without device execution')
    p.add_argument('--baseline', help='Sealed A100 performance package (performance/preview)')
    p.add_argument('--keep-evidence', action='store_true', default=None,
                   help='Keep complete performance evidence instead of compact results')
    p.add_argument('--oracle', help='Read-only, sealed A100 reference package (accuracy/preview)')
    p.add_argument('--keep-tensors', action='store_true', default=None,
                   help='Keep accuracy tensors and debugging evidence instead of compact results')
    p.add_argument('--source', help='Read-only source for report, analyze or repack-reference')
    p.add_argument('--ranks', nargs='+', type=int, help='Report view: selected logical ranks')
    p.add_argument('--shapes', nargs='+', help='Report view: input shapes, e.g. 4,20 4,256')
    p.add_argument('--phases', nargs='+', choices=['prefill','decode'], help='Report view: generation phases')
    p.add_argument('--generation-steps', nargs='+', type=int, help='Report view: zero-based generation steps')
    p.add_argument('--portable', action='store_true', help='Copy evidence into a self-contained report directory')
    p.add_argument('--reanalyze', action='store_true', help='Analyze profiler evidence in a new copied directory')
    p.add_argument('--analysis-timeout-seconds', type=int,
                   help='Independent CUDA analysis deadline (performance, analyze or report --reanalyze)')
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--output', help='New output directory; existing evidence is never overwritten')
    for component in ('flaggems', 'flagtree', 'flagcx'):
        p.add_argument('--'+component, choices=['off', 'on', 'both'])
    p.add_argument('--levels', nargs='+', choices=['model', 'layer'])
    p.add_argument('--level', choices=['total', 'layer'])
    p.add_argument('--layer-profile-rounds', type=int)
    p.add_argument('--warmup-rounds', type=int)
    p.add_argument('--measure-rounds', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--monitor', choices=['on', 'off'], help='Independent telemetry sidecar (default: on)')
    p.add_argument('--monitor-interval-ms', type=int, help='Telemetry polling interval; not sensor resolution')
    p.add_argument('--layers', nargs='+')
    p.add_argument('--resume-from', help='Read-only schema 3 preview source directory')
    p.add_argument('--budget-seconds', type=int, help='This preview run: baseline, search and verification budget')
    p.add_argument('--preview-evidence', choices=['lightweight','full'], help='Preview output evidence; all checks run in either mode')
    p.add_argument('--resume-cache', choices=['auto','off'], help='Import sealed cache groups from a compatible preview source')
    p.add_argument('--preview-search', choices=['grouped','sequential'], help='Preview candidate search strategy')
    p.add_argument('--preview-budget', choices=['adaptive','fixed'], help='Preview cost and verification reservation policy')
    p.add_argument('--policy')
    p.add_argument('--model-path')
    p.add_argument('--image', help='Local Docker image for host execution or CPU graph export')
    p.add_argument('--input-path')
    p.add_argument('--vendor', help='Registered vendor adapter ID')
    p.add_argument('--device', type=int)
    p.add_argument('--devices', type=int, nargs='+')
    p.add_argument('--parallelism', choices=['single','tp'])
    p.add_argument('--communication-profile-rounds', type=int)
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'])
    p.add_argument('--task-score', action='store_true', default=None)
    p.add_argument('--data-path')
    p.add_argument('--formats', nargs='+', choices=['fx', 'onnx'])
    p.add_argument('--internal-existing-root', action='store_true', help=argparse.SUPPRESS)
    return p


def load(args):
    public_command = args.command
    baseline_capture = args.command == 'performance-baseline'
    if baseline_capture:
        args.command = 'performance'
    if args.source is not None:
        raise ConfigError('--source requires report')
    path = Path(args.config).expanduser().resolve()
    def read_config(filename):
        try:
            return yaml.safe_load(filename.read_text())
        except yaml.YAMLError as error:
            raise ConfigError(f'invalid YAML in {filename}: {error}') from error
    default = read_config(DEFAULT_CONFIG)
    default['model'].setdefault('options', {})
    default['inputs'].setdefault('options', {})
    for entry in sorted((ROOT/'registry/vendors').glob('*.json')):
        info = descriptor('vendors', entry.stem)
        defaults = info.get('config_defaults', {'env': {}})
        selected = default['vendors'].setdefault(entry.stem, {})
        for key, value in defaults.items(): selected.setdefault(key, deepcopy(value))
    provided = read_config(path)
    if not isinstance(provided, dict):
        raise ConfigError('configuration must be a mapping')
    def merge(base, update, prefix=''):
        for k, value in update.items():
            if k not in base:
                raise ConfigError(f'unknown configuration: {prefix}{k}')
            if isinstance(base[k], dict):
                if not isinstance(value, dict):
                    raise ConfigError(f'{prefix}{k} must be a mapping')
                # Vendor environment dictionaries are intentionally extensible.
                if k == 'env' or (k == 'options' and prefix in ('model.', 'inputs.')):
                    base[k].update(value)
                else:
                    merge(base[k], value, prefix + k + '.')
            else:
                base[k] = value
    def resolve_paths(config, directory):
        for group, key in PATH_FIELDS:
            section = config.get(group)
            if isinstance(section, dict) and section.get(key) is not None:
                value = section[key]
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(f'{group}.{key} must be a nonempty path string')
                section[key] = str((directory / Path(value).expanduser()).resolve())

        for vendor, section in config.get('vendors', {}).items():
            if not isinstance(section, dict):
                raise ConfigError('vendor configuration must be a mapping')
            for field in descriptor('vendors', vendor).get('path_fields', []):
                if field not in section:
                    continue
                value = section[field]
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(f'vendors.{vendor}.{field} must be a path')
                section[field] = str((directory / Path(value).expanduser()).resolve())

    cfg = deepcopy(default)
    resolve_paths(cfg, DEFAULT_CONFIG.parent)
    resolve_paths(provided, path.parent)
    merge(cfg, provided)
    if args.device is not None and args.devices is not None:
        raise ConfigError('--device and --devices are mutually exclusive')
    if args.analysis_timeout_seconds is not None and args.command != 'performance':
        raise ConfigError('--analysis-timeout-seconds requires performance or report --reanalyze')
    if args.command == 'performance' and args.levels is not None:
        raise ConfigError('performance uses --level; --levels belongs to accuracy')
    if args.command != 'performance' and any(getattr(args,k) is not None for k in ('level','warmup_rounds','measure_rounds','repeats')):
        raise ConfigError('--level and performance sampling options require performance')
    if args.command != 'performance' and (args.monitor is not None or args.monitor_interval_ms is not None):
        raise ConfigError('--monitor and --monitor-interval-ms require performance')
    if args.command != 'preview' and (args.resume_from is not None or args.budget_seconds is not None):
        raise ConfigError('--resume-from and --budget-seconds require preview')
    if args.command != 'preview' and (args.preview_evidence is not None or args.resume_cache is not None):
        raise ConfigError('--preview-evidence and --resume-cache require preview')
    if args.command != 'preview' and (args.preview_search is not None or args.preview_budget is not None):
        raise ConfigError('--preview-search and --preview-budget require preview')
    if args.command != 'report' and (args.ranks is not None or args.shapes is not None or args.phases is not None
                                    or args.generation_steps is not None or args.portable or args.reanalyze):
        raise ConfigError('--ranks, --shapes, --phases, --generation-steps, --portable and --reanalyze require report')
    component_section = args.command if args.command in ['accuracy','performance','preview'] else 'accuracy'
    mapping = {'flaggems': (component_section,'flaggems'),
               'monitor': ('performance','monitor'), 'monitor_interval_ms': ('performance','monitor_interval_ms'),
               'flagtree': (component_section,'flagtree'), 'flagcx': (component_section,'flagcx'),
               'levels':('accuracy','levels'), 'level':('performance','level'),
               'warmup_rounds':('performance','warmup_rounds'),
               'measure_rounds':('performance','measure_rounds'), 'repeats':('performance','repeats'),
               'layers':('performance' if args.command == 'performance' else 'accuracy','layers'), 'policy':('policy','path'),
               'layer_profile_rounds':('performance','layer_profile_rounds'),
                'resume_from':('preview','resume_from'), 'budget_seconds':('preview','budget_seconds'),
               'preview_evidence':('preview','evidence_mode'), 'resume_cache':('preview','resume_cache'),
               'preview_search':('preview','search_strategy'), 'preview_budget':('preview','budget_mode'),
               'model_path':('model','path'), 'input_path':('inputs','path'), 'image':('container','image'),
               'vendor':('runtime','vendor'), 'device':('runtime','device'),
               'analysis_timeout_seconds':('runtime','analysis_timeout_seconds'),
               'devices':('runtime','devices'), 'parallelism':('runtime','parallelism'),
               'communication_profile_rounds':('performance','communication_profile_rounds'),
               'dtype':('model','dtype'), 'task_score':('accuracy','task_score'),
               'data_path':('accuracy','data_path'), 'formats':('export','formats'),
               'baseline':('reference','performance_path'), 'keep_evidence':('performance','keep_evidence'),
               'oracle':('reference','path'), 'keep_tensors':('accuracy','keep_tensors')}
    for arg, (group, key) in mapping.items():
        value = getattr(args, arg)
        if value is not None:
            if (group, key) in PATH_FIELDS:
                value = str(Path(value).expanduser().resolve())
            cfg[group][key] = value
    if args.device is not None and cfg['runtime'].get('parallelism') == 'tp':
        raise ConfigError('TP uses --devices, not --device')
    if args.communication_profile_rounds is not None and args.command != 'performance':
        raise ConfigError('--communication-profile-rounds requires performance')
    layer_mode = args.command == 'performance' and cfg['performance']['level'] == 'layer'
    if args.layer_profile_rounds is not None and not layer_mode:
        raise ConfigError('--layer-profile-rounds requires performance --level layer')
    if args.layers is not None and args.command not in ['accuracy','oracle'] and not layer_mode:
        raise ConfigError('--layers requires accuracy or performance --level layer')
    if layer_mode and args.communication_profile_rounds is not None:
        raise ConfigError('layer uses --layer-profile-rounds for shared layer/communication profiling')
    if args.command == 'export' and any(getattr(args, k) not in [None, 'off'] for k in ('flaggems','flagtree','flagcx')):
        raise ConfigError('export supports native off components only')
    if args.keep_tensors is not None and args.command != 'accuracy':
        raise ConfigError('--keep-tensors requires accuracy')
    model_adapter(cfg).normalize_config(cfg)
    if args.command == 'oracle':
        if args.levels is None and (path == DEFAULT_CONFIG.resolve() or 'levels' not in provided.get('accuracy',{})):
            cfg['accuracy']['levels'] = ['model','layer']
    if cfg.get('reference',{}).get('path'):
        if args.command not in ['accuracy','preview']:
            raise ConfigError('--oracle/reference.path requires accuracy or preview')
        from runtime.oracle import bind_reference
        explicit_input = args.input_path is not None or (path != DEFAULT_CONFIG.resolve() and 'path' in provided.get('inputs',{}))
        if args.dry_run:
            from runtime.preflight import reference_metadata
            reference_metadata(cfg)
        else:
            bind_reference(cfg, args.command, explicit_input)
    if args.keep_evidence is not None and args.command != 'performance':
        raise ConfigError('--keep-evidence requires performance')
    if baseline_capture:
        if cfg['reference'].get('performance_path') or cfg['reference'].get('path'):
            raise ConfigError('baseline generation cannot import a reference')
        if cfg['runtime']['vendor'] != 'nvidia' or cfg['model']['dtype'] != model_adapter(cfg).reference_dtype:
            raise ConfigError('performance-baseline requires NVIDIA A100 and the model reference dtype')
        if any(cfg['performance'][k] != 'off' for k in ('flaggems','flagtree','flagcx')):
            raise ConfigError('performance-baseline requires native components')
        if cfg['runtime']['parallelism'] == 'tp' and len(cfg['runtime']['devices'] or []) != 2:
            raise ConfigError('A100 baseline currently supports single or TP=2')
        cfg['performance']['level'] = 'layer'
        cfg['performance']['layers'] = ['all']
        cfg['_performance_capture'] = True
    if cfg['reference'].get('performance_path'):
        if cfg['reference'].get('path') or args.command not in ('performance','preview'):
            raise ConfigError('performance and accuracy references are mutually exclusive; baseline requires performance/preview')
        from runtime.performance_baseline import bind_reference
        explicit = args.input_path is not None or (path != DEFAULT_CONFIG.resolve() and 'path' in provided.get('inputs',{}))
        if args.dry_run:
            from runtime.preflight import reference_metadata
            reference_metadata(cfg, performance=True)
        else:
            bind_reference(cfg,args.command,explicit)
    if baseline_capture or cfg['reference'].get('performance_path'):
        cfg['_performance_native'] = True
    validate(cfg, args.command)
    if baseline_capture:
        args.command = 'performance-baseline'
    return cfg


def validate(c, command, vendor_command=None):
    if command == 'oracle':
        if c['model']['dtype'] != model_adapter(c).reference_dtype:
            raise ConfigError('oracle requires the model reference dtype (Qwen3 BF16 reference, DeepFM FP32 reference)')
        if c['runtime']['vendor'] != 'nvidia' or c['runtime'].get('parallelism') != 'single':
            raise ConfigError('oracle requires NVIDIA A100 single-device execution')
        if any(c['accuracy'].get(k,'off') != 'off' for k in ['flaggems','flagtree','flagcx']):
            raise ConfigError('oracle requires all three components off')
        if c.get('reference',{}).get('path'):
            raise ConfigError('oracle generation cannot import an oracle')
        return validate(c, 'accuracy', vendor_command='oracle')
    if c.get('reference',{}).get('path') and command not in ['accuracy','preview']:
        raise ConfigError('reference.path requires accuracy or preview')
    if command == 'accuracy' and type(c.get('accuracy',{}).get('keep_tensors',False)) is not bool:
        raise ConfigError('accuracy.keep_tensors must be boolean')
    if c['schema_version'] != 1:
        raise ConfigError('unsupported schema_version')
    try:
        model_entry = descriptor('models', c['model']['name'])
        vendor_entry = descriptor('vendors', c['runtime']['vendor'])
        model_adapter(c).validate_config(c)
        vendor_adapter(c).validate_config(c, vendor_command or command)
        if command != 'list':
            require_capability(model_entry, command)
            require_capability(vendor_entry, command)
        require_capability(model_entry, 'parallelism', c['runtime'].get('parallelism', 'single'))
        require_capability(vendor_entry, 'parallelism', c['runtime'].get('parallelism', 'single'))
        if command == 'export':
            for fmt in c['export']['formats']:
                require_capability(model_entry, 'export', fmt)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if type(c['runtime']['device']) is not int or c['runtime']['device'] < 0:
        raise ConfigError('device must be a non-negative integer')
    parallel = c['runtime'].get('parallelism','single')
    devices = c['runtime'].get('devices')
    if parallel not in ['single','tp']:
        raise ConfigError('parallelism must be single or tp')
    if parallel == 'tp':
        if command not in ['preview','accuracy','performance','list']:
            raise ConfigError('TP is supported by preview, accuracy and performance only')
        if (not isinstance(devices,list) or len(devices) not in model_entry['capabilities'].get('tp_degrees', [])
                or any(type(d) is not int or d < 0 for d in devices)
                or len(set(devices)) != len(devices)):
            supported=model_entry['capabilities'].get('tp_degrees',[])
            raise ConfigError(f'TP devices must contain a supported degree {supported} of unique nonnegative physical IDs')
    elif devices is not None:
        raise ConfigError('devices requires parallelism=tp; single uses device')
    positive_fields = [('runtime', 'timeout_seconds'), ('inputs', 'batch_size')]
    from runtime.measurement_quality import settings as variability_settings
    try:
        variability_settings(c.get('performance', {}).get('variability'))
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if 'analysis_timeout_seconds' in c['runtime']:
        positive_fields.append(('runtime', 'analysis_timeout_seconds'))
    if command == 'preview':
        positive_fields.append(('preview', 'budget_seconds'))
        if c['preview']['evidence_mode'] not in ['lightweight','full']:
            raise ConfigError('preview.evidence_mode must be lightweight or full')
        if c['preview']['resume_cache'] not in ['auto','off']:
            raise ConfigError('preview.resume_cache must be auto or off (quote YAML values)')
        if c['preview']['search_strategy'] not in ['grouped','sequential']:
            raise ConfigError('preview.search_strategy must be grouped or sequential')
        if c['preview']['budget_mode'] not in ['adaptive','fixed']:
            raise ConfigError('preview.budget_mode must be adaptive or fixed')
    if command == 'accuracy':
        positive_fields.append(('accuracy', 'worst_samples'))
    if command == 'performance':
        if c['performance'].get('monitor', 'on') not in ('on', 'off'):
            raise ConfigError('performance.monitor must be quoted on/off')
        interval = c['performance'].get('monitor_interval_ms', 1000)
        if type(interval) is not int or interval < 100:
            raise ConfigError('performance.monitor_interval_ms must be an integer >= 100')
        positive_fields.extend(('performance', key) for key in
                               ['warmup_rounds', 'measure_rounds', 'repeats',
                                'communication_profile_rounds', 'layer_profile_rounds'])
    for group, key in positive_fields:
        if type(c[group][key]) is not int or c[group][key] < 1:
            raise ConfigError(f'{group}.{key} must be a positive integer')
    if command in ['accuracy','performance','preview']:
        switches = {k:c[command].get(k,'off') for k in ('flaggems','flagtree','flagcx')}
        if any(v not in ['off','on','both'] for v in switches.values()):
            raise ConfigError('component switches must be off/on/both (quote YAML values)')
        if list(switches.values()).count('both') > 1:
            raise ConfigError('at most one component may be both')
        for component, switch in switches.items():
            if switch != 'off' and component not in vendor_entry['capabilities'].get('components', []):
                raise ConfigError(f'{component} is not supported by the selected vendor adapter')
    if command == 'performance':
        if type(c['performance'].get('keep_evidence',False)) is not bool:
            raise ConfigError('performance.keep_evidence must be boolean')
        if c['performance']['level'] not in ['total', 'layer']:
            raise ConfigError('performance supports level total or layer')
        names = c['performance']['layers']
        if (not isinstance(names, list) or not names or
                any(not isinstance(n, str) or not n.strip() for n in names) or
                len(set(names)) != len(names) or ('all' in names and names != ['all'])):
            raise ConfigError('performance.layers must be unique module names or [all]')
    if command in ['accuracy', 'export']:
        key, allowed = ('levels', {'model', 'layer'}) if command == 'accuracy' else ('formats', {'fx', 'onnx'})
        values = c[command][key]
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or v not in allowed for v in values)):
            raise ConfigError(f'invalid {command}.{key}')
    if command == 'accuracy':
        layers = c['accuracy']['layers']
        if not isinstance(layers,list) or not layers or any(not isinstance(x,str) for x in layers):
            raise ConfigError('layers must be a nonempty list of names or [all]')
        if 'all' in layers and layers != ['all']:
            raise ConfigError('all cannot be mixed with explicit layer names')
        if type(c['accuracy']['task_score']) is not bool:
            raise ConfigError('task_score must be boolean')
        if c['accuracy']['task_score']:
            data = c['accuracy']['data_path']
            if not data or not Path(data).exists():
                raise ConfigError('task scoring requires an existing --data-path; no fallback')
            raise ConfigError('task scoring is not implemented; no evaluation was run')
    if command == 'list':
        return
    if command == 'preview' and c['preview'].get('resume_from'):
        from runtime.preview import load_resume
        load_resume(c['preview']['resume_from'])
    try:
        model_adapter(c).validate_assets(c)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if command in ['accuracy','performance'] and c[command]['flaggems'] in ['on','both']:
        if not c['policy']['path'] or not Path(c['policy']['path']).is_file():
            raise ConfigError('FlagGems on requires --policy from a verified preview')
