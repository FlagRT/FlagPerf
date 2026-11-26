# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Single-target semantics; keep runner-only and attachment-owned options distinct."""
import json
import os
from pathlib import Path
from flagprof.probes.config import PRESETS, resolve
from flagprof.runner import CTR_ARCH_CHOICES


def choices(key, value, allowed):
    if value not in allowed:
        raise ValueError(f'{key} 需要取值于 {", ".join(map(str, allowed))}')


def allowed_keys(block, values, allowed):
    extra = sorted(set(values) - set(allowed))
    if extra:
        raise ValueError(f'{block}.{extra[0]} 不适用于所选目标或预设')


def fill(block, **defaults):
    for key, value in defaults.items():
        if block.get(key) is None:
            block[key] = value


def normalize(config, origins):
    if config.get('schema_version') != 1:
        raise ValueError('schema_version 需要整数 1')
    choices('mode', config.setdefault('mode', 'single'), ['single'])
    config.setdefault('name', 'single')
    if not config.get('target'):
        raise ValueError('single 需要 target')
    target = config['target']
    target_type = target.setdefault('type', 'builtin')
    choices('target.type', target_type, ('builtin', 'command'))
    preset = config.setdefault('profile', {}).setdefault('preset', 'perf')
    choices('profile.preset', preset, PRESETS + ('none',))
    hardware = config.setdefault('hardware', {})
    fill(hardware, vendor='auto', nproc=1, device='auto')
    choices('hardware.nproc',hardware['nproc'],(1,2))
    tp = hardware['nproc']==2
    if tp and hardware['device']!='auto':
        raise ValueError('hardware.nproc=2 时由 launch 绑定设备，hardware.device 需要 auto')
    if not tp and hardware.get('comm_backend','auto')!='auto':
        raise ValueError('hardware.comm_backend 需要 hardware.nproc=2')
    if target_type == 'command':
        allowed_keys('target', target, ('type', 'command', 'cwd'))
        if 'runner' in config:
            raise ValueError('自定义 command 不接受 runner 配置')
        cmd = target.get('command')
        if not cmd or not Path(cmd[0]).name.startswith('python') or len(cmd) < 2:
            raise ValueError('target.command 需要直接启动 Python 的 argv 和目标')
    else:
        builtin(config, origins)
        if tp and (target['task'] not in ('embedding','generate') or config['runner'].get('trust_remote_code')):
            raise ValueError('hardware.nproc=2 仅支持已适配 TP2 的 embedding/generate')
    out = config['output']
    fill(out, verify=True, report=True, attach_runner_results='auto')
    choices('output.attach_runner_results', out['attach_runner_results'], ('auto', True, False))
    components = config.setdefault('components', {})
    fill(components, flagtree='inherit')
    gems = components.setdefault('flaggems', {})
    fill(gems, enabled=False)
    choices('components.flagtree', components['flagtree'], ('on', 'off', 'inherit'))
    owns_compiler = tp or preset != 'none' or target_type == 'builtin'
    if owns_compiler and not components.get('vendor_compiler') and os.environ.get('FLAGPROF_VENDOR_COMPILER'):
        components['vendor_compiler'] = str(Path(os.environ['FLAGPROF_VENDOR_COMPILER']).expanduser().resolve())
        origins['components.vendor_compiler'] = 'FLAGPROF_VENDOR_COMPILER'
    include = gems.get('include')
    if isinstance(include, str):
        include = json.loads(Path(include).read_text())
    if include is not None and (not isinstance(include, list) or not all(isinstance(x, str) and x for x in include)):
        raise ValueError('components.flaggems.include 需要算子名称数组')
    if bool(include) != gems['enabled']:
        raise ValueError('components.flaggems.include 与 enabled 不匹配，开启时需要非空列表')
    if target.get('task') == 'generate' and gems['enabled']:
        raise ValueError('generate 不支持 FlagGems')
    if preset == 'none':
        if tp:
            c = config.setdefault('collection',{})
            allowed_keys('collection',c,('timeout','capture_only','profile_memory'))
            fill(c,timeout=1800)
            if c['timeout']<=0 or c.get('capture_only') or c.get('profile_memory'):
                raise ValueError('TP2 none 需要正 timeout 且关闭采集')
        if (not tp and config.get('collection')) or config.get('monitor'):
            raise ValueError('profile.preset=none 不接受 collection 或 monitor')
        if target_type == 'command' and (gems['enabled'] or not tp and
                                         (components['flagtree'] != 'inherit' or components.get('vendor_compiler'))):
            raise ValueError('无插桩自定义命令由目标自行配置组件')
    else:
        collection(config, include, origins)
    can_attach = (preset in ('perf', 'modules', 'perf-lite', 'coverage') and target_type == 'builtin'
                  and not config.get('collection', {}).get('capture_only') and out['report'])
    if out['attach_runner_results'] is True and not can_attach:
        raise ValueError('output.attach_runner_results=true 需要可挂接内置 runner 的采集报告')
    return can_attach and out['attach_runner_results'] is not False


def builtin(config, origins):
    target = config['target']
    task = target.setdefault('task', 'embedding')
    choices('target.task', task, ('embedding', 'generate', 'rerank', 'ctr'))
    inputs = {'embedding': ('model','texts','inputs','inputs_from','samples'),
              'generate': ('model','inputs_from','samples'), 'rerank': ('model','samples'), 'ctr': ('samples',)}
    allowed_keys('target', target, ('type','task') + inputs[task])
    runner = config.setdefault('runner', {})
    common = ('dtype','batch_size','warmup','forwards','repeats','numerics')
    options = {'embedding': ('pooling','attention','padding_side','max_length','trust_remote_code'),
               'generate': ('attention','padding_side','max_length','generation_mode','max_new_tokens'),
               'rerank': ('package','score','attention','trust_remote_code'), 'ctr': ('package','arch','seed')}
    allowed_keys('runner', runner, common + options[task])
    fill(runner, dtype='bfloat16', warmup=0, repeats=1, forwards=1 if task=='generate' else 3, numerics='default')
    choices('runner.dtype', runner['dtype'], ('bfloat16','float16','float32'))
    choices('runner.numerics', runner['numerics'], ('default','reference'))
    if task != 'ctr':
        if not target.get('model'):
            raise ValueError('该任务需要 target.model')
        fill(runner, attention='eager')
    if task in ('embedding','generate'):
        fill(runner, max_length=256, padding_side='auto')
        if task == 'embedding':
            used = [k for k in ('inputs','inputs_from','texts') if target.get(k) is not None]
            if len(used) != 1 or (target.get('samples') and used != ['inputs_from']):
                raise ValueError('embedding 需要 texts/inputs/inputs_from 三选一；samples 需要 inputs_from')
            if runner.get('pooling','auto') == 'auto':
                from flagprof.training.registry import resolve as registry_value
                value, source = registry_value(None, target['model'], kind='pooling')
                runner['pooling'] = value or 'last'
                origins['runner.pooling'] = source or 'default'
            choices('runner.pooling', runner['pooling'], ('last','mean','cls'))
        if runner['padding_side'] == 'auto':
            runner['padding_side'] = 'right' if runner.get('pooling') == 'cls' else 'left'
            origins['runner.padding_side'] = 'pooling'
        choices('runner.padding_side', runner['padding_side'], ('right','left'))
    if task == 'generate':
        fill(runner, generation_mode='accuracy', max_new_tokens=32)
        choices('runner.generation_mode', runner['generation_mode'], ('performance','accuracy'))
        if not target.get('inputs_from') or runner.get('batch_size') not in (None, 1):
            raise ValueError('generate 需要 target.inputs_from 和 batch_size=1')
        if runner['generation_mode'] == 'accuracy' and (runner['warmup'],runner['forwards'],runner['repeats']) != (0,1,1):
            raise ValueError('generate accuracy 需要 warmup=0, forwards=repeats=1')
    if task in ('rerank','ctr') and not runner.get('package'):
        raise ValueError(task + ' 需要 runner.package')
    if task == 'rerank':
        if runner['attention'] != 'eager' or runner['package'] == 'auto':
            raise ValueError('rerank 需要 eager 和封存输入包')
        manifest = json.loads((Path(runner['package'])/'manifest.json').read_text())
        score = manifest.get('score_semantics')
        choices('manifest.score_semantics', score, ('causal_yesno','sequence_logit'))
        if runner.get('score','auto') == 'auto':
            runner['score'] = score
            origins['runner.score'] = 'package/manifest.json'
        choices('runner.score', runner['score'], (score,))
    if task == 'ctr':
        choices('runner.arch', runner.get('arch'), CTR_ARCH_CHOICES)
        fill(runner, seed=531)
    for key in ('warmup','forwards','repeats','batch_size','max_length','max_new_tokens'):
        if runner.get(key) is not None and runner[key] < (0 if key=='warmup' else 1):
            raise ValueError('runner.' + key + ' 超出允许范围')


def collection(config, include, origins):
    preset = config['profile']['preset']
    c = config.setdefault('collection', {})
    tp = config['hardware']['nproc']==2
    if tp:
        allowed_keys('collection',c,('timeout','source_max_bytes','capture_only','profile_memory','tensor_boundaries',
                     'tensor_ops_in','tensor_blob','tensor_blob_bytes'))
    fill(c, timeout=1800, capture_mode='final', capture_only=False, profile_memory=False)
    choices('collection.capture_mode', c['capture_mode'], ('final','windowed'))
    official = preset in ('perf','modules')
    if not official:
        c['capture_mode'] = 'windowed'
        origins['collection.capture_mode'] = 'preset'
    windowed = c['capture_mode'] == 'windowed'
    if (c['capture_only'] and (windowed or not official)) or (c['profile_memory'] and not official):
        raise ValueError('capture_only 需要 perf/modules final，profile_memory 需要 perf/modules')
    if c['timeout'] <= 0:
        raise ValueError('collection.timeout 需要正数')
    if c.get('source_max_bytes') is not None:
        if windowed and not tp:
            raise ValueError('collection.source_max_bytes 仅适用于 final')
        from flagprof.offline.checkpoints import source_budget
        source_budget(c['source_max_bytes'])
    for k in ('active','windows','invalid_policy','log_files','plugins'):
        if c.get(k) is not None and not windowed and (k!='log_files' or c[k]):
            raise ValueError('collection.' + k + ' 需要 windowed')
    for k in ('active','windows'):
        if c.get(k) is not None and c[k] < 1:
            raise ValueError('collection.' + k + ' 需要正整数')
    if c.get('invalid_policy') is not None:
        choices('collection.invalid_policy', c['invalid_policy'], ('stop','continue'))
    options = {k:v for k,v in c.items() if v is not None and (k.startswith('tensor_') or k in ('dispatch','layers','probes'))}
    if preset == 'accuracy':
        if c.get('tensor_blob') in (None,'auto'):
            c['tensor_blob'] = 'all'
        prefixes = c['tensor_blob']
        if tp and prefixes not in ('all','none'):
            raise ValueError('TP2 collection.tensor_blob 需要 all 或 none')
        if isinstance(prefixes, list) and (not prefixes or any(not p or ',' in p for p in prefixes)):
            raise ValueError('collection.tensor_blob 需要可用逗号分隔的非空前缀列表')
        fill(c, tensor_blob_bytes=1<<30)
        options.update(tensor_blob=c['tensor_blob'], tensor_blob_bytes=c['tensor_blob_bytes'])
        boundary = options.get('tensor_boundaries')
        if isinstance(boundary, str) and boundary not in ('model','layer'):
            options['tensor_boundaries'] = json.loads(Path(boundary).read_text())
        if c.get('tensor_ops_in') and config.get('runner', {}).get('batch_size') != 1:
            raise ValueError('collection.tensor_ops_in 需要 runner.batch_size=1')
    monitor = config.setdefault('monitor', {})
    fill(monitor, enabled=preset=='monitor', topology='auto', diagnose=False, interval_ms=1000)
    if monitor['diagnose'] and monitor['topology'] == 'auto':
        monitor['topology'] = 'agent'
    options.update(monitor='on' if monitor['enabled'] else 'off', diagnose='on' if monitor['diagnose'] else 'off',
                   monitor_interval_ms=monitor['interval_ms'], flaggems_include=include,
                   flaggems='on' if config['components']['flaggems']['enabled'] else 'off')
    if monitor['topology'] != 'auto':
        options['monitor_topology'] = monitor['topology']
    if monitor.get('root'):
        options['monitor_root'] = monitor['root']
    resolve(preset, options)
    if tp:
        c.pop('capture_mode')
