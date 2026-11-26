# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Read a versioned input, retaining each value's base directory until resolution."""
import copy
import difflib
import json
import math
import os
from pathlib import Path
import re

# A trailing ? permits JSON null. Unions are decoded according to the destination field.
FIELDS = {
    'schema_version': 'int', 'mode': 'str', 'name': 'str',
    'target': dict(type='str', task='str', model='str?', texts='list?', inputs='str?',
                   inputs_from='str?', samples='list|str?', command='list?', cwd='str?'),
    'hardware': dict(vendor='str', nproc='int', device='str?', comm_backend='str'),
    'profile': dict(preset='str'),
    'runner': dict(dtype='str', batch_size='int?', warmup='int', forwards='int?', repeats='int',
                   numerics='str', attention='str', pooling='str', padding_side='str',
                   max_length='int', trust_remote_code='bool', generation_mode='str',
                   max_new_tokens='int', package='str?', score='str', arch='str?', seed='int'),
    'components': dict(flaggems=dict(enabled='bool', include='list|str?'),
                       flagtree='str', vendor_compiler='str?'),
    'collection': dict(profile_memory='bool', capture_only='bool', timeout='number',
                       source_max_bytes='int?', tensor_boundaries='list|str?', tensor_blob='list|str?',
                       tensor_blob_bytes='int?', tensor_ops_in='str?', dispatch='str?', layers='str?',
                       probes='str?', capture_mode='str', active='int?', windows='int?',
                       invalid_policy='str?', log_files='list', plugins='str?'),
    'monitor': dict(enabled='bool', topology='str', root='str?', interval_ms='int', diagnose='bool'),
    'output': dict(dir='str', job_dir='str?', verify='bool', report='bool',
                   attach_runner_results='bool|str', limits='dict|str?'),
    'export': dict(formats='list|str', bundle='str', verify_onnx='bool', batch_index='int', timeout='number'),
    'batch': dict(config='str?', models='list?', filter='str?'),
}
ROLES = ('ref', 'baseline', 'cmp')
LOCAL = ('target', 'hardware', 'runner', 'components', 'collection', 'monitor')
SIDE = dict(label='str', existing_run='str?', existing_report='str?', **{k:FIELDS[k] for k in LOCAL})
FIELDS['compare'] = dict.fromkeys(ROLES, SIDE)
FIELDS['compare'].update(declare='str|list|dict', note='str?', perf_ref_run='str?', perf_cmp_run='str?')
TYPES = {'int': (int,), 'number': (int, float), 'bool': (bool,), 'list': (list,),
         'str': (str,), 'dict': (dict,)}
PATHS = {'target.model', 'target.inputs', 'target.inputs_from', 'target.cwd', 'runner.package',
         'components.vendor_compiler', 'components.flaggems.include', 'collection.log_files',
         'collection.plugins', 'collection.tensor_boundaries', 'monitor.root',
         'output.dir', 'output.job_dir', 'output.limits'}
SPECIAL = {'collection.tensor_boundaries': ('model', 'layer'), 'runner.package': ('auto',)}
TOKEN = re.compile(r'\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def definition(parts):
    spec = FIELDS
    for i, key in enumerate(parts):
        valid = spec if isinstance(spec, dict) else {}
        if key not in valid:
            suggestion = difflib.get_close_matches(key, valid, n=1)
            hint = '，建议 ' + '.'.join(parts[:i] + suggestion) if suggestion else ''
            raise ValueError('未知配置项 ' + '.'.join(parts[:i+1]) + hint)
        spec = spec[key]
    return spec


def validate(spec, value, parts, origins, origin):
    key = '.'.join(parts)
    if isinstance(spec, dict):
        if value is None and parts == ['compare','baseline']:
            return
        if not isinstance(value, dict):
            raise ValueError(key + ' 需要对象')
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(key + ' 配置项名称需要字符串')
            validate(definition(parts+[k]), v, parts+[k], origins, origin)
        return
    origins[key] = origin
    kinds = spec.rstrip('?').split('|')
    if value is None and spec.endswith('?'):
        return
    if not any(type(value) in TYPES[k] for k in kinds):
        hint = '，YAML 中请写 "on" 或 "off"' if parts[-2:] == ['components','flagtree'] else ''
        raise ValueError(f'{key} 需要 {spec}{hint}')
    if isinstance(value, list):
        if key == 'batch.models':
            if not all(isinstance(x, dict) for x in value):
                raise ValueError(key + ' 需要模型对象数组')
        elif not all(isinstance(x, str) for x in value):
            raise ValueError(key + ' 需要字符串数组')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(key + ' 需要有限数值')


def cli_value(spec, raw, key):
    if raw == 'null':
        return None
    if isinstance(spec, str) and 'str' in spec and not raw.startswith('"'):
        if not (('list' in spec and raw.startswith('[')) or ('dict' in spec and raw.startswith('{'))
                or ('bool' in spec and raw in ('true', 'false'))):
            return raw
    try:
        return json.loads(raw)
    except ValueError as error:
        raise ValueError(key + ' 需要符合字段类型的 JSON 值') from error


def expand(value, key):
    def replacement(match):
        name = match.group(1)
        if name is None:
            return '$'
        if name not in os.environ:
            raise ValueError(f'{key} 缺少环境变量 {name}')
        return os.environ[name]
    return TOKEN.sub(replacement, value)


def path_values(config, change, prefix=''):
    """Apply to external path strings and command tokens, never to inline data."""
    for key in sorted(PATHS | {'target.command'}):
        parts = key.split('.')
        parent = config
        for part in parts[:-1]:
            parent = parent.get(part, {})
        field = parts[-1]
        value = parent.get(field)
        if isinstance(value, str) and value not in SPECIAL.get(key, ()):
            parent[field] = change(prefix+key, value)
        elif key in ('target.command', 'collection.log_files') and isinstance(value, list):
            parent[field] = [change(prefix+key, arg) for arg in value]
    comparison = config.get('compare', {})
    for role in ROLES:
        side = comparison.get(role)
        if isinstance(side, dict):
            path_values(side,change,'compare.'+role+'.')
            for field in ('existing_run','existing_report'):
                if side.get(field):
                    side[field] = change(f'compare.{role}.{field}',side[field])
    for field in ('declare','perf_ref_run','perf_cmp_run'):
        value = comparison.get(field)
        if isinstance(value, str) and not (field=='declare' and value=='auto'):
            comparison[field] = change('compare.'+field,value)
    batch = config.get('batch', {})
    roster = batch.get('config')
    if isinstance(roster, str) and not roster.startswith('builtin:'):
        batch['config'] = change('batch.config', roster)
    if batch.get('models') is not None:
        from flagprof.batch.batch_driver import model_rows, path_fields
        for key, fields, field in path_fields(model_rows(batch)):
            fields[field] = change(key, fields[field])


def resolve_path(key, value, base):
    value = expand(value, key)
    local = key.split('.', 2)[-1] if key.startswith('compare.') else key
    if local == 'target.command':
        return value
    model = (local == 'target.model' or re.fullmatch(r'batch\.models\.\d+\.model', local)
             or local.endswith('.args.model') and value.startswith('hf:'))
    if model:
        value = value.removeprefix('hf:')
    if not value:
        raise ValueError(key + ' 需要非空路径')
    absolute = str((base / Path(value).expanduser()).resolve())
    return 'hf:' + absolute if model else absolute


def snapshot(config):
    result = copy.deepcopy(config)
    path_values(result, lambda key, value: value.replace('$', '$$'))
    return result


def value_sources(config, explicit):
    result = {}
    validate(FIELDS, config, [], result, 'default')
    result.update(explicit)
    return result


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key),dict) and isinstance(value,dict) else copy.deepcopy(value)
    return result


def existing(side):
    return isinstance(side,dict) and any(side.get(k) is not None for k in ('existing_run','existing_report'))


def inherit(side, data):
    if not isinstance(side, dict) or existing(side):
        return side
    return merge({k:data[k] for k in LOCAL if k in data},side)


def assign(data, parts, value):
    parent = data
    for part in parts[:-1]:
        parent = parent.setdefault(part, {})
        if not isinstance(parent, dict):
            raise ValueError('.'.join(parts) + ' 的父字段需要对象')
    parent[parts[-1]] = value


def load(path, overrides, out):
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        data = json.loads(content)
    elif path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as error:
            raise ValueError('当前环境缺少 PyYAML。可执行 pip install PyYAML，或使用等价 JSON 配置。') from error
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ValueError('YAML 解析失败 ' + str(error)) from error
    else:
        raise ValueError('配置文件扩展名需要 .yaml、.yml 或 .json')
    if not isinstance(data, dict) or not data:
        raise ValueError('配置顶层需要非空对象')
    origins = {}
    def expand_side(role):
        side = data.get('compare',{}).get(role)
        if isinstance(side,dict) and not existing(side):
            prefix = 'compare.'+role+'.'
            for key, origin in list(origins.items()):
                if key.split('.')[0] in LOCAL:
                    origins.setdefault(prefix+key,origin)
            data['compare'][role] = inherit(side,data)
    if isinstance(data.get('compare'),dict):
        for role in ROLES:
            expand_side(role)
    for index, item in enumerate(overrides, 1):
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError('--set 需要 KEY=VALUE')
        parts = key.split('.')
        spec = definition(parts)
        value = cli_value(spec, raw, key)
        destinations = [parts]
        comparison = data.get('compare',{})
        if parts[0] in LOCAL and isinstance(comparison,dict):
            destinations.extend(['compare',role,*parts] for role in ROLES
                                if isinstance(comparison.get(role),dict) and not existing(comparison[role]))
        if len(parts)>2 and parts[0]=='compare' and parts[1] in ROLES and comparison.get(parts[1]) is None:
            data.setdefault('compare',{})[parts[1]] = {}
            expand_side(parts[1])
        for dest in destinations:
            prefix = '.'.join(dest)
            origins = {k:v for k,v in origins.items() if k!=prefix and not k.startswith(prefix+'.')}
            validate(spec, value, dest, origins, f'--set[{index}]')
            assign(data,dest,copy.deepcopy(value))
        if parts==['compare'] or (len(parts)==2 and parts[0]=='compare' and parts[1] in ROLES):
            for role in ROLES if len(parts)==1 else [parts[1]]:
                expand_side(role)
    if out is not None:
        if not isinstance(data.setdefault('output', {}), dict):
            raise ValueError('output 需要对象')
        data['output']['job_dir'] = out
        origins['output.job_dir'] = '--out'
    mode = data.get('mode', 'single')
    if mode not in ('single','compare','export','batch'):
        raise ValueError('mode 需要 single、compare、export 或 batch')
    for block in ('compare', 'export', 'batch'):
        if block in data and mode != block:
            raise ValueError(f'{mode} 不接受 {block} 配置')
    if data.get('mode') == 'compare':
        for key in LOCAL:
            if key in data:
                validate(FIELDS[key],data[key],[key],{},'config')
        # Local sides now own the selections. Unused shared paths must not require
        # this host's model, compiler or environment variables in an offline job.
        for key in LOCAL:
            data.pop(key,None)
        origins = {k:v for k,v in origins.items() if k.split('.')[0] not in LOCAL}
    defaults = {}
    validate(FIELDS, data, [], defaults, 'config')
    defaults.update(origins)
    origins = defaults
    def resolve(key, value):
        origin_key = 'batch.models' if key.startswith('batch.models.') else key
        base = path.parent if origins.get(origin_key, 'config') == 'config' else Path.cwd()
        return resolve_path(key, value, base)
    output = data.setdefault('output', {})
    if 'dir' not in output:
        output['dir'] = './flagprof-results'
    for side in [data]+[data.get('compare',{}).get(r) for r in ROLES]:
        if isinstance(side,dict) and side.get('target',{}).get('type')=='command' and not side['target'].get('cwd'):
            side['target']['cwd'] = str(path.parent).replace('$', '$$')
    path_values(data, resolve)
    return data, origins
