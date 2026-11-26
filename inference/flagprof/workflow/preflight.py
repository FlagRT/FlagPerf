# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Check inputs and output ownership without starting the workload."""
import json
import os
from pathlib import Path
import shutil
from flagprof.vendors import vendor_names, launch_backend


def exists(value, key, kind='file'):
    path = Path(value)
    if not (path.is_dir() if kind=='dir' else path.is_file()):
        raise ValueError(f'{key} 缺少{ "目录" if kind=="dir" else "文件" } {path}')
    return path


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def check_destination(job, config, source):
    job = Path(job).resolve()
    explicit = bool(config['output'].get('job_dir'))
    if explicit and job.exists() and (not job.is_dir() or any(job.iterdir())):
        raise ValueError('任务目录需要为空或尚未创建 ' + str(job))
    target, runner = config.get('target',{}), config.get('runner', {})
    roots = [target.get('inputs_from'), (target.get('model') or '').removeprefix('hf:'),
             runner.get('package'), config.get('components',{}).get('vendor_compiler'), config.get('monitor', {}).get('root')]
    if any(overlap(job, Path(p)) for p in roots if p and p != 'auto'):
        raise ValueError('任务目录与输入目录重叠 ' + str(job))
    ancestors = (job, *job.parents) if explicit else job.parents
    for ancestor in ancestors:
        if any((ancestor / marker).is_file() for marker in
               ('resolved-config.json','plan-b-capture.json','record-status.json','group.json','result.json')) or (
                (ancestor/'manifest.json').is_file() and (ancestor/'writers').is_dir()):
            raise ValueError('任务目录位于既有输入或工作流产物内 ' + str(ancestor))
    if job in source.parents or job == source:
        raise ValueError('任务目录包含配置文件 ' + str(source))
    parent = next((p for p in ancestors if p.exists()), None)
    if not parent or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError('任务目录无法创建 ' + str(job))


def check_command(target, collection):
    cwd = exists(target['cwd'], 'target.cwd', 'dir')
    exe, *args = target['command']
    binary = str(cwd / exe) if os.sep in exe else shutil.which(exe)
    if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise ValueError('target.command 找不到 Python ' + exe)
    if collection.get('capture_mode') == 'final':
        while args and args[0] in ('-B','-u'):
            args = args[1:]
        if not args or (args[0].startswith('-') and args[0] not in ('-c','-m')):
            raise ValueError('final 支持 Python 脚本、-m 或 -c')
        if args[0] in ('-m','-c') and len(args) < 2:
            raise ValueError('target.command 缺少模块名称或代码')
    elif collection.get('capture_mode') == 'windowed':
        for arg in args:
            if arg in ('-c','-m','--') or not arg.startswith('-'):
                break
            if arg.startswith('-') and any(c in arg[1:] for c in 'SIE') and not arg.startswith(('-W','-X','--')):
                raise ValueError('windowed 采集需要 Python site 与 PYTHONPATH，不能使用 ' + arg)
    if args and not args[0].startswith('-'):
        exists(cwd/args[0], 'target.command')


def check_text_input(target, size):
    if target.get('inputs'):
        from itertools import islice
        with exists(target['inputs'], 'target.inputs').open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in islice((s for s in stream if s.strip()), size)]
        valid = [r if isinstance(r,str) else r.get('text', r.get('query',r.get('input')))
                 if isinstance(r,dict) else None for r in rows]
    else:
        valid = target['texts'][:size]
    if len(valid) < size or not all(isinstance(x,str) for x in valid):
        raise ValueError(f'target 输入需要至少 {size} 条文本 (runner.batch_size)')


def check(config, source, job):
    target, hardware = config['target'], config['hardware']
    runner, c = config.get('runner', {}), config.get('collection', {})
    check_destination(job, config, source)
    tp = hardware['nproc']==2
    if target['type'] == 'command':
        check_command(target, {'capture_mode':'final'} if tp else c)
    else:
        if target.get('model'):
            model = exists(target['model'][3:], 'target.model', 'dir')
            exists(model/'config.json', 'target.model')
            if tp:
                info=json.loads((model/'config.json').read_text())
                if info.get('model_type')!='qwen3' or (target['task']=='generate' and
                        (info.get('hidden_size'),info.get('num_hidden_layers'),info.get('vocab_size'))!=(4096,36,176384)):
                    raise ValueError('hardware.nproc=2 的内置目标需要受支持的 Qwen3 模型结构')
        if target.get('inputs') or target.get('texts') is not None:
            check_text_input(target, runner.get('batch_size') or 4)
        package = runner.get('package') or target.get('inputs_from')
        if package and package != 'auto':
            manifest = json.loads(exists(Path(package)/'manifest.json', '输入包').read_text())
            if target['task'] == 'generate':
                contract = manifest.get('contract', {})
                expected = {'batch_size': 1, 'max_length': runner['max_length'], 'padding_side': runner['padding_side']}
                model = contract.get('model', {})
                if (any(contract.get('inputs', {}).get(k) != v for k,v in expected.items())
                        or any(model.get(k) != runner[k] for k in ('dtype','attention'))
                        or model.get('options', {}).get('max_new_tokens') != runner['max_new_tokens']):
                    raise ValueError('输入包 contract 与 generate 的输入长度、数值或生成步数不符')
    compiler = config['components'].get('vendor_compiler')
    if compiler:
        exists(Path(compiler)/'manifest.json', 'components.vendor_compiler')
    limits = config['output'].get('limits')
    if limits is not None:
        from flagprof.analyzers.workspace import Limits
        values = json.loads(Path(limits).read_text()) if isinstance(limits,str) else limits
        try:
            Limits(**values)
        except (TypeError, ValueError) as error:
            raise ValueError('output.limits ' + str(error)) from error
    vendor = hardware['vendor']
    if not c and not tp:
        if vendor != 'auto' and vendor not in vendor_names():
            raise ValueError('未知 hardware.vendor ' + vendor)
        return None
    plugins = c.get('plugins')
    if plugins:
        from flagprof.registry import Registry
        records = json.loads(Path(plugins).read_text())
        if not isinstance(records, list) or not all(isinstance(d, dict) for d in records):
            raise ValueError('collection.plugins 需要组件描述对象数组')
        descriptors = Registry(records).descriptors()
        candidates = [d['id'] for d in descriptors if d['kind']=='vendor']
        if vendor == 'auto' and len(candidates) == 1:
            vendor = candidates[0]
        if vendor not in candidates:
            raise ValueError('hardware.vendor 需选插件厂商 ' + str(candidates))
    else:
        try:
            backend=hardware.get('comm_backend','auto')
            vendor, backend = launch_backend(None if vendor=='auto' else vendor, None if backend=='auto' else backend)
            if tp:
                hardware['comm_backend']=backend
        except ValueError as error:
            raise ValueError('请使用 --set hardware.vendor=厂商名 选择采集适配器。' + str(error)) from error
        device = hardware['device']
        if target['type'] == 'builtin' and device not in ('auto','cpu'):
            from flagprof.vendors import adapter
            expected = adapter(vendor, 'execution').TORCH_DEVICE
            if device.split(':', 1)[0] != expected:
                raise ValueError(f'hardware.device={device} 与 {vendor} 的设备类型 {expected} 不匹配')
    if target['type'] == 'command' and config.get('monitor', {}).get('enabled') and hardware['device'] != 'auto':
        from flagprof.devices import validate_target_device
        format_vendor = next(d['format_vendor'] for d in descriptors if d['id'] == vendor) if plugins else vendor
        validate_target_device(hardware['device'], format_vendor)
    if hardware['device'] == 'cpu' and not plugins:
        raise ValueError('CPU 基准使用 profile.preset=none')
    for path in c.get('log_files', []):
        exists(path, 'collection.log_files')
    monitor = config.get('monitor',{})
    if monitor.get('topology') == 'agent':
        from flagprof.monitor.session import snapshot
        root = monitor.get('root')
        hint = f'flagprof watch start --root {root} --vendor {vendor} --devices all'
        try:
            session = snapshot(root)
        except (OSError, ValueError):
            session = {}
        if session.get('status') != 'running' or session.get('vendor') != vendor:
            raise ValueError('缺少匹配的运行中 watch 会话，请先运行 ' + hint)
    return vendor
