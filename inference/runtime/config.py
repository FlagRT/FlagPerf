# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One validated configuration shared by host and container entrypoints."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from runtime.common import ROOT


DEFAULT_CONFIG = ROOT / 'config' / 'default.yaml'
PATH_FIELDS = {('model', 'path'), ('inputs', 'path'), ('runtime', 'output_root'),
               ('policy', 'path'), ('accuracy', 'data_path'), ('preview','resume_from')}


class ConfigError(ValueError):
    pass


def parser():
    p = argparse.ArgumentParser(description='Inference accuracy and performance')
    p.add_argument('command', choices=['accuracy', 'performance', 'preview', 'export', 'list', 'report'])
    p.add_argument('--source', help='Sealed performance run for offline reporting')
    p.add_argument('--ranks', nargs='+', type=int, help='Report view: selected logical ranks')
    p.add_argument('--shapes', nargs='+', help='Report view: input shapes, e.g. 4,20 4,256')
    p.add_argument('--portable', action='store_true', help='Copy evidence into a self-contained report directory')
    p.add_argument('--reanalyze', action='store_true', help='Analyze profiler evidence in a new copied directory')
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--output', help='New output directory; existing evidence is never overwritten')
    p.add_argument('--image', help='Compatible local Docker image; overrides container.image')
    for component in ('flaggems', 'flagtree', 'flagcx'):
        p.add_argument('--'+component, choices=['off', 'on', 'both'])
    p.add_argument('--levels', nargs='+', choices=['model', 'layer'])
    p.add_argument('--level', choices=['total', 'layer'])
    p.add_argument('--layer-profile-rounds', type=int)
    p.add_argument('--warmup-rounds', type=int)
    p.add_argument('--measure-rounds', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--layers', nargs='+')
    p.add_argument('--resume-from', help='Read-only schema 3 preview source directory')
    p.add_argument('--budget-seconds', type=int, help='This preview run: baseline, search and verification budget')
    p.add_argument('--preview-evidence', choices=['lightweight','full'], help='Preview output evidence; all checks run in either mode')
    p.add_argument('--resume-cache', choices=['auto','off'], help='Import sealed cache groups from a compatible preview source')
    p.add_argument('--preview-search', choices=['grouped','sequential'], help='Preview candidate search strategy')
    p.add_argument('--preview-budget', choices=['adaptive','fixed'], help='Preview cost and verification reservation policy')
    p.add_argument('--policy')
    p.add_argument('--model-path')
    p.add_argument('--input-path')
    p.add_argument('--vendor', choices=['ascend', 'nvidia'])
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
    if args.source is not None:
        raise ConfigError('--source requires report')
    path = Path(args.config).expanduser().resolve()
    default = yaml.safe_load(DEFAULT_CONFIG.read_text())
    provided = yaml.safe_load(path.read_text())
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
                if k == 'env':
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

        ascend = config.get('vendors', {}).get('ascend', {})
        if 'vendor_compiler' in ascend:
            value = ascend['vendor_compiler']
            if not isinstance(value, str) or not value.strip():
                raise ConfigError('vendors.ascend.vendor_compiler must be a path')
            ascend['vendor_compiler'] = str((directory / Path(value).expanduser()).resolve())

    cfg = deepcopy(default)
    resolve_paths(cfg, DEFAULT_CONFIG.parent)
    resolve_paths(provided, path.parent)
    merge(cfg, provided)
    if args.device is not None and args.devices is not None:
        raise ConfigError('--device and --devices are mutually exclusive')
    if args.command == 'performance' and args.levels is not None:
        raise ConfigError('performance uses --level; --levels belongs to accuracy')
    if args.command != 'performance' and any(getattr(args,k) is not None for k in ('level','warmup_rounds','measure_rounds','repeats')):
        raise ConfigError('--level and performance sampling options require performance')
    if args.command != 'preview' and (args.resume_from is not None or args.budget_seconds is not None):
        raise ConfigError('--resume-from and --budget-seconds require preview')
    if args.command != 'preview' and (args.preview_evidence is not None or args.resume_cache is not None):
        raise ConfigError('--preview-evidence and --resume-cache require preview')
    if args.command != 'preview' and (args.preview_search is not None or args.preview_budget is not None):
        raise ConfigError('--preview-search and --preview-budget require preview')
    if args.command != 'report' and (args.ranks is not None or args.shapes is not None or args.portable or args.reanalyze):
        raise ConfigError('--ranks, --shapes, --portable and --reanalyze require report')
    component_section = args.command if args.command in ['accuracy','performance','preview'] else 'accuracy'
    mapping = {'image': ('container', 'image'), 'flaggems': (component_section,'flaggems'),
               'flagtree': (component_section,'flagtree'), 'flagcx': (component_section,'flagcx'),
               'levels':('accuracy','levels'), 'level':('performance','level'),
               'warmup_rounds':('performance','warmup_rounds'),
               'measure_rounds':('performance','measure_rounds'), 'repeats':('performance','repeats'),
               'layers':('performance' if args.command == 'performance' else 'accuracy','layers'), 'policy':('policy','path'),
               'layer_profile_rounds':('performance','layer_profile_rounds'),
                'resume_from':('preview','resume_from'), 'budget_seconds':('preview','budget_seconds'),
               'preview_evidence':('preview','evidence_mode'), 'resume_cache':('preview','resume_cache'),
               'preview_search':('preview','search_strategy'), 'preview_budget':('preview','budget_mode'),
               'model_path':('model','path'), 'input_path':('inputs','path'),
               'vendor':('runtime','vendor'), 'device':('runtime','device'),
               'devices':('runtime','devices'), 'parallelism':('runtime','parallelism'),
               'communication_profile_rounds':('performance','communication_profile_rounds'),
               'dtype':('model','dtype'), 'task_score':('accuracy','task_score'),
               'data_path':('accuracy','data_path'), 'formats':('export','formats')}
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
    if args.layers is not None and args.command != 'accuracy' and not layer_mode:
        raise ConfigError('--layers requires accuracy or performance --level layer')
    if layer_mode and args.communication_profile_rounds is not None:
        raise ConfigError('layer uses --layer-profile-rounds for shared layer/communication profiling')
    if args.command == 'export' and any(getattr(args, k) not in [None, 'off'] for k in ('flaggems','flagtree','flagcx')):
        raise ConfigError('export supports native off components only')
    validate(cfg, args.command)
    return cfg


def validate(c, command):
    if c['schema_version'] != 1:
        raise ConfigError('unsupported schema_version')
    if c['model']['name'] != 'qwen3_embedding_0.6b':
        raise ConfigError('only qwen3_embedding_0.6b has a current model adapter; see support.json')
    if c['model']['dtype'] not in ['bfloat16','float16','float32'] or c['model']['attention'] != 'eager':
        raise ConfigError('dtype must be bfloat16/float16/float32; attention must be eager')
    if c['runtime']['vendor'] not in ['ascend','nvidia']:
        raise ConfigError('vendor must be ascend or nvidia')
    if type(c['runtime']['device']) is not int or c['runtime']['device'] < 0:
        raise ConfigError('device must be a non-negative integer')
    parallel = c['runtime'].get('parallelism','single')
    devices = c['runtime'].get('devices')
    if parallel not in ['single','tp']:
        raise ConfigError('parallelism must be single or tp')
    if parallel == 'tp':
        if command not in ['preview','accuracy','performance','list']:
            raise ConfigError('TP is supported by preview, accuracy and performance only')
        if c['runtime']['vendor'] != 'ascend':
            raise ConfigError('TP currently requires ascend/HCCL')
        if (not isinstance(devices,list) or len(devices) not in [2,4,8]
                or any(type(d) is not int or d < 0 for d in devices)
                or len(set(devices)) != len(devices)):
            raise ConfigError('TP devices must contain 2, 4 or 8 unique nonnegative physical IDs')
    elif devices is not None:
        raise ConfigError('devices requires parallelism=tp; single uses device')
    positive_fields = [('runtime', 'timeout_seconds'), ('inputs', 'batch_size'),
                       ('inputs', 'max_length')]
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
        positive_fields.extend(('performance', key) for key in
                               ['warmup_rounds', 'measure_rounds', 'repeats',
                                'communication_profile_rounds', 'layer_profile_rounds'])
    for group, key in positive_fields:
        if type(c[group][key]) is not int or c[group][key] < 1:
            raise ConfigError(f'{group}.{key} must be a positive integer')
    if c['inputs']['padding_side'] not in ['left','right']:
        raise ConfigError('padding_side must be left or right')
    if command in ['accuracy','performance','preview']:
        switches = {k:c[command].get(k,'off') for k in ('flaggems','flagtree','flagcx')}
        if any(v not in ['off','on','both'] for v in switches.values()):
            raise ConfigError('component switches must be off/on/both (quote YAML values)')
        if list(switches.values()).count('both') > 1:
            raise ConfigError('at most one component may be both')
        if c['runtime']['vendor'] != 'ascend' and any(switches[k] != 'off' for k in ['flagtree','flagcx']):
            raise ConfigError('FlagTree/FlagCX adapters currently require ascend')
    if c['runtime']['vendor'] == 'ascend' and c['vendors']['ascend']['env'].get('TASK_QUEUE_ENABLE') != '0':
        raise ConfigError('validated Ascend stack requires TASK_QUEUE_ENABLE=0')
    if command == 'performance':
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
    if not c['model']['path']:
        raise ConfigError('set model.path in your configuration or pass --model-path /path/to/Qwen3-Embedding-0.6B')
    model = Path(c['model']['path'])
    for name in ['config.json','tokenizer_config.json','model.safetensors']:
        with (model / name).open('rb') as stream:
            stream.read(16)
    if not Path(c['inputs']['path']).is_file():
        raise ConfigError('input JSONL path does not exist; no fallback')
    if command in ['accuracy','performance'] and c[command]['flaggems'] in ['on','both']:
        if not c['policy']['path'] or not Path(c['policy']['path']).is_file():
            raise ConfigError('FlagGems on requires --policy from a verified preview')
