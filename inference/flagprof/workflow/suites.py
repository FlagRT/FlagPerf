# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile graph export and model rosters into their public CPU/runner commands."""
from argparse import Namespace
from importlib.resources import files
import json
from pathlib import Path
import sys

from .loader import resolve_path
from .planner import CONTROL, flags
from .preflight import check_destination, check_text_input, exists, overlap
from .schema import allowed_keys, choices


def export_plan(config, origins, job):
    from flagprof.graph.command import configuration
    from flagprof.training.registry import resolve

    target = config.get('target', {})
    allowed_keys('target', target, ('type', 'task', 'model', 'texts', 'inputs', 'inputs_from'))
    if target.setdefault('type', 'builtin') != 'builtin' or target.setdefault('task', 'embedding') != 'embedding':
        raise ValueError('export 需要内置 Qwen3 embedding 目标')
    if not target.get('model'):
        raise ValueError('export 需要 target.model')
    model = exists(target['model'][3:], 'target.model', 'dir')
    if json.loads(exists(model/'config.json', 'target.model').read_text()).get('model_type') != 'qwen3':
        raise ValueError('export 需要本地 Qwen3 embedding 模型')
    sources = [key for key in ('texts', 'inputs', 'inputs_from') if target.get(key) is not None]
    if len(sources) != 1:
        raise ValueError('export 需要 target.texts、inputs 或 inputs_from 中的一种')
    source = sources[0]
    runner = config.setdefault('runner', {})
    allowed_keys('runner', runner, ('dtype', 'batch_size', 'max_length', 'pooling', 'padding_side'))
    choices('runner.dtype', runner.setdefault('dtype', 'bfloat16'), ('float32', 'bfloat16'))
    if runner.get('pooling', 'auto') == 'auto':
        selected, origin = resolve(None, target['model'], kind='pooling')
        runner['pooling'] = selected or 'last'
        origins['runner.pooling'] = origin or 'default'
    choices('runner.pooling', runner['pooling'], ('last', 'mean'))
    if runner.get('padding_side', 'auto') == 'auto':
        runner['padding_side'] = 'left'
        origins['runner.padding_side'] = 'pooling'
    choices('runner.padding_side', runner['padding_side'], ('left', 'right'))
    runner.setdefault('max_length', 256)
    options = config.setdefault('export', {})
    formats = options.setdefault('formats', ['fx', 'onnx'])
    if isinstance(formats, str):
        formats = formats.split(',')
    options['formats'] = [name.strip() for name in formats]
    choices('export.bundle', options.setdefault('bundle', 'structure'), ('structure', 'full'))
    options.setdefault('verify_onnx', False)
    options.setdefault('batch_index', 0)
    options.setdefault('timeout', 1800)
    assets = {}
    inputs = str(job/'inputs/texts.jsonl') if source == 'texts' else target.get('inputs')
    if source == 'texts':
        assets[inputs] = target['texts']
    input_flag = '--inputs-from' if source == 'inputs_from' else '--inputs'
    input_path = target.get('inputs_from') or inputs
    formats_arg = ','.join(options['formats'])
    # Let the public exporter decide format, index, timeout and verification compatibility.
    settings = dict(runner, **options)
    settings.update(model=target['model'], inputs=inputs, inputs_from=target.get('inputs_from'),
                    formats=formats_arg, batch_size=runner.get('batch_size'))
    configuration(Namespace(**settings))
    if source == 'inputs_from':
        package = Path(input_path)
        exists(package/'manifest.json', 'target.inputs_from')
        for name in ('inputs.pt','samples.json','forward-inputs.json'):
            exists(package/'prepared'/name, 'target.inputs_from')
        batches = json.loads((package/'prepared/forward-inputs.json').read_text())['batches']
        count = len(batches)
        size = runner.get('batch_size')
        if size:
            count = (sum(len(batch['sample_ids']) for batch in batches) + size - 1) // size
        if options['batch_index'] >= count:
            raise ValueError(f'export.batch_index 超出固定输入包的 {count} 个 batch')
    else:
        check_text_input(target, runner.get('batch_size') or 4)
    argv = CONTROL + ['export', target['model'], input_flag, input_path]
    argv += flags(runner, ('dtype','batch_size','max_length','pooling','padding_side'))
    argv += flags(options, ('formats','bundle','verify_onnx','batch_index','timeout'))
    argv += ['--out', str(job/'export-bundle')]
    return argv, assets, dict(input_type=source, output_type='graph-bundle', component_owner='CPU export worker',
                              requested_formats=options['formats'], bundle=options['bundle'])


def batch_plan(config, origins, source, job):
    try:
        import yaml
    except ImportError as error:
        raise ValueError('批量名册需要 PyYAML，可安装 pip install PyYAML') from error
    from flagprof.batch.batch_driver import EXAMPLES, model_rows, path_fields

    batch = config.get('batch', {})
    hardware = config.setdefault('hardware', {})
    allowed_keys('hardware', hardware, ('device',))
    device = hardware.get('device') or 'auto'
    hardware['device'] = device
    reference = batch.get('config')
    if (batch.get('models') is None) == (reference is None):
        raise ValueError('batch.config 与 batch.models 需要二选一')
    if reference is None:
        rows = model_rows(batch)
        kind, origin = 'inline', origins.get('batch.models', 'config')
    else:
        if reference.startswith('builtin:'):
            name = reference.removeprefix('builtin:')
            if name not in EXAMPLES:
                raise ValueError('batch.config 内置名册可选 builtin:cpu、builtin:models、builtin:hf2')
            resource = files('flagprof.batch').joinpath(EXAMPLES[name])
            base, kind = source.parent, 'builtin'
        else:
            resource = exists(reference, 'batch.config')
            if overlap(job, resource):
                raise ValueError('任务目录与 batch.config 重叠')
            base, kind = resource.parent, 'external'
        try:
            rows = model_rows(yaml.safe_load(resource.read_text(encoding='utf-8')))
        except yaml.YAMLError as error:
            raise ValueError('批量名册 YAML 解析失败 ' + str(error)) from error
        for key, fields, field in path_fields(rows):
            fields[field] = resolve_path(key, fields[field], base)
        origin = reference
        if kind == 'builtin':
            # Replays read this immutable selection with its original resource base.
            del batch['config']
            batch['models'] = rows
            origins.pop('batch.config', None)
            origins['batch.models'] = origin
    delivery = job/'batch-results'
    for _, fields, field in path_fields(rows):
        path = Path(fields[field].removeprefix('hf:'))
        if overlap(path, job):
            raise ValueError(f'任务目录与名册输入重叠 {path}')
    selected = [r for r in rows if not batch.get('filter') or batch['filter'] in r['id']]
    missing_inputs = [
        dict(model_id=row['id'], field=field, path=fields[field])
        for row in selected for _, fields, field in path_fields([row])
        if not Path(fields[field].removeprefix('hf:')).exists()
    ]
    outputs = set()
    for row in selected:
        for repeat in (1, 2):
            dest = (delivery/f"{row['id']}-run{repeat}").resolve()
            if delivery not in dest.parents or dest in outputs:
                raise ValueError(f"batch 模型 id 需要独立的 batch-results 子目录 {row['id']}")
            outputs.add(dest)
    roster = str(job/'inputs/models.yaml')
    argv = [sys.executable, '-B', '-m', 'flagprof.batch.batch_driver', '--config', roster, '--out', str(delivery)]
    if device != 'auto':
        argv += ['--device', device]
    if batch.get('filter'):
        argv += ['--filter', batch['filter']]
    return argv, {roster: {'models': rows}}, dict(input_type=kind, output_type='batch-results',
                component_owner='batch driver', roster_source=origin, model_ids=[r['id'] for r in selected],
                missing_inputs=missing_inputs)


def prepare(config, origins, source, job):
    mode = config['mode']
    if config.get('schema_version') != 1:
        raise ValueError('schema_version 需要整数 1')
    common = {'schema_version','mode','name','output'}
    allowed_keys(mode, config, common | ({'target','runner','export'} if mode == 'export' else {'batch','hardware'}))
    allowed_keys('output', config['output'], ('dir','job_dir'))
    check_destination(job, config, source)
    if mode == 'export':
        argv, assets, info = export_plan(config, origins, job)
    else:
        argv, assets, info = batch_plan(config, origins, source, job)
    step = dict(id=mode, variant=mode, argv=argv, cwd=str(job), depends_on=[])
    if mode == 'batch':
        step['roster_source'] = info['roster_source']
    return dict(info, source='local', vendor=None, directory=str(job), runners=[], steps=[step], skipped={}), assets
