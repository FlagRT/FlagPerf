# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Import checks for the interpreters that will actually execute the plan."""
import os
import subprocess
import tempfile

from .executor import error_detail, execution_environment


def python_options(argv):
    """Retain interpreter flags without forwarding a script or code selector."""
    command = [argv[0]]
    options = iter(argv[1:])
    for option in options:
        if option in ('-X', '-W', '--check-hash-based-pycs'):
            argument = next(options, None)
            if argument is None:
                raise ValueError(f'Python 参数 {option} 缺少取值')
            command.extend([option, argument])
        elif option.startswith(('-X', '-W')) or (
                option.startswith('-') and len(option) > 1 and set(option[1:]) <= set('bBdEIOPqsSuv')):
            command.append(option)
        else:
            break
    return command


def runner_modules(task):
    return ['torch', 'numpy'] + (['transformers'] if task != 'ctr' else [])


def dependencies(config, plan, assets):
    from flagprof.vendors import adapter, torch_runtimes, vendor_names
    for step in plan['steps']:
        label, argv = step['id'], step['argv']
        action = step.get('action', label)
        if action == 'replay.run':
            device = config['replay'][step['variant']]['device']
            extension = torch_runtimes().get(device.split(':')[0])
            modules = ['torch', 'numpy']
            if extension:
                modules.append(extension)
            if step['flaggems']:
                modules.append('flag_gems')
            yield label, [argv[0]], None, modules, step['compiler_selection']
            continue
        if action == 'export':
            options = config['export']
            modules = runner_modules('embedding')
            if 'onnx' in options['formats']:
                modules.extend(['onnx', 'onnxscript'])
            if options['verify_onnx']:
                modules.append('onnxruntime')
            yield label, [argv[0]], None, modules, None
            continue
        if action == 'batch':
            yield label, [argv[0]], None, ['yaml'], None
            roster = assets[argv[argv.index('--config') + 1]]['models']
            selected = set(plan['model_ids'])
            python = os.environ.get('BATCH_PYTHON') or argv[0]
            device = config.get('hardware', {}).get('device') or 'auto'
            extension = torch_runtimes().get(device.split(':', 1)[0]) if device not in ('auto', 'cpu') else None
            for row in roster:
                if row['id'] in selected and row['task'] in ('embedding','generate','rerank','ctr'):
                    modules = runner_modules(row['task'])
                    if extension:
                        modules.append(extension)
                    yield f"batch.{row['id']}", [python], None, modules, None
            continue
        if action not in ('record', 'launch', 'runner', 'command'):
            continue
        role = step['variant']
        side = config['compare'][role] if config.get('mode') == 'compare' else config
        target, collection = side['target'], side.get('collection', {})
        component, preset = side['components'], side['profile']['preset']
        builtin = target['type'] == 'builtin'
        gems = component['flaggems']['enabled']
        vendor_profiler = preset in ('perf', 'modules') and not collection.get('plugins')
        tp = action == 'launch'
        default_probes = 'none' if preset in ('monitor', 'none') else 'default'
        torch_probe = preset != 'none' and collection.get('probes', default_probes) != 'none'
        modules = runner_modules(target['task']) if builtin else []
        if not builtin and (tp or vendor_profiler or gems or torch_probe):
            modules.append('torch')
        vendor = side['hardware']['vendor']
        device = side['hardware'].get('device', 'auto')
        extension = None
        if (tp or vendor_profiler or gems) and vendor in vendor_names():
            extension = getattr(adapter(vendor, 'execution'), 'TORCH_EXTENSION', None)
        elif builtin and device not in ('auto', 'cpu'):
            extension = torch_runtimes().get(device.split(':', 1)[0])
        if extension:
            modules.append(extension)
        if gems:
            modules.append('flag_gems')
        selection = None
        if component['flagtree'] != 'inherit':
            selection = dict(flagtree=component['flagtree'], vendor_compiler=component.get('vendor_compiler'))
        if modules or selection:
            if action in ('record','launch'):
                argv = argv[argv.index('--') + 1:]
            yield label, python_options(argv), target.get('cwd'), modules, selection


def check_dependencies(config, plan, assets):
    checked = {}
    if config.get('mode') == 'replay':
        from pathlib import Path
        from flagprof.replay import prepare
        replay = config['replay']
        with tempfile.TemporaryDirectory(prefix='flagprof-check-') as scratch:
            prepare(replay['ref']['run'], replay['cmp']['run'], replay['call_key'], Path(scratch)/'package')
    for step, interpreter, cwd, modules, selection in dependencies(config, plan, assets):
        program = ''
        if selection:
            program = 'from flagprof.components.compiler import select\nselect(' + repr(selection) + ')\n'
        program += '\n'.join('import ' + module for module in modules)
        key = (tuple(interpreter), cwd, program)
        if key in checked:
            continue
        with tempfile.TemporaryDirectory(prefix='flagprof-check-') as empty, tempfile.TemporaryFile() as errors:
            try:
                result = subprocess.run([*interpreter, '-B', '-c', program], cwd=cwd or empty,
                                        env=execution_environment(), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=errors, timeout=60)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ValueError(f"{step} 依赖检查无法运行 {interpreter[0]}: {error}") from error
            if result.returncode:
                detail = error_detail(errors) or f'退出码 {result.returncode}'
                raise ValueError(f"{step} 依赖检查失败 {interpreter[0]} ({', '.join(modules)}): {detail}")
        checked[key] = dict(step=step, python=interpreter[0], imports=modules)
    return list(checked.values())
