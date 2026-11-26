# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile a single target into public CLI commands and typed dependencies."""
from pathlib import Path
import sys

RUN = '<record.run>'
CONTROL = [sys.executable, '-B'] + (['-S'] if sys.flags.no_site else []) + ['-m','flagprof']


def flags(values, names):
    result = []
    for key in names:
        value = values.get(key)
        if value is None:
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, ','.join(value) if isinstance(value,list) else str(value)])
    return result


def compile_plan(config, job, vendor, attach, *, variant='single'):
    target, output = config['target'], config['output']
    preset = config['profile']['preset']
    tp = config['hardware']['nproc'] == 2
    builtin = target['type']=='builtin'
    c = config.get('collection', {})
    inputs, raw, report = job/'inputs', job/'store', job/'report'
    runner_outputs = job/'runner-outputs'
    runners = [str(runner_outputs/f'rank-{r}'/'runner.json') for r in range(2)] if tp else [str(runner_outputs/'runner.json')]
    run = RUN if variant=='single' else f'<{variant}.run>'
    manifest = str(raw/'group.json')
    steps, assets, skipped = [], {}, {}
    def step(name, argv, dependencies=(), cwd=job):
        steps.append(dict(id=name, variant=variant, argv=argv, cwd=str(cwd), depends_on=list(dependencies)))
    def input_path(key, value, *, jsonl=False):
        if isinstance(value,(list,dict)):
            path = inputs / (key + ('.jsonl' if jsonl else '.json'))
            assets[str(path)] = value
            return str(path)
        return value
    component = config['components']
    gems = component['flaggems']
    component_args = ['--flaggems', 'on' if gems['enabled'] else 'off']
    if gems.get('include'):
        component_args += ['--flaggems-include', input_path('flaggems-include', gems['include'])]
    compiler_args = flags(component, ['vendor_compiler'])
    if component['flagtree'] != 'inherit':
        compiler_args += ['--flagtree', component['flagtree']]
    cwd = job if builtin else Path(target['cwd'])
    before = []
    if builtin:
        runner = config['runner'].copy()
        if runner.get('package') == 'auto':
            runner['package'] = str(inputs/'ctr-package')
            step('make-package', [sys.executable, '-B','-m','flagprof.targets.make_ctr_package','--samples','8',
                                  '--seed', str(runner['seed']), '--out',runner['package']])
            before.append('make-package')
        command = [sys.executable, '-B','-m','flagprof.runner','--task',target['task']]
        if target.get('model'):
            command.append(target['model'])
        if target.get('texts') is not None:
            command += ['--inputs', input_path('texts',target['texts'],jsonl=True)]
        command += flags(target, ('inputs','inputs_from','samples'))
        command += flags(runner, sorted(runner))
        if config['hardware']['device'] != 'auto':
            command += ['--device', config['hardware']['device']]
        command += ['--outputs',str(runner_outputs)]
    else:
        command = target['command'].copy()
    if preset!='none' or tp:
        monitor = config.get('monitor',{})
        settings = dict(c)
        if settings.get('tensor_boundaries') is not None:
            settings['tensor_boundaries'] = input_path('boundaries',settings['tensor_boundaries'])
        producer = 'launch' if tp else 'record'
        argv = CONTROL + [producer,'--vendor',vendor,'--preset',preset,'--store',str(raw)]
        if tp:
            argv += ['--nproc-per-node','2','--comm-backend',config['hardware']['comm_backend']]
        elif not builtin and monitor.get('enabled') and config['hardware']['device'] != 'auto':
            argv += ['--target-device', config['hardware']['device']]
        argv += flags(settings, ('capture_mode','timeout','source_max_bytes','capture_only','profile_memory',
                                 'active','windows','invalid_policy','plugins','tensor_blob','tensor_blob_bytes',
                                 'tensor_boundaries','tensor_ops_in','dispatch','layers','probes'))
        for log in c.get('log_files', []):
            argv += ['--log-file',log]
        if monitor:
            argv += ['--monitor', 'on' if monitor['enabled'] else 'off',
                     '--monitor-interval-ms', str(monitor['interval_ms']), '--diagnose', 'on' if monitor['diagnose'] else 'off']
        if monitor.get('topology','auto') != 'auto':
            argv += ['--monitor-topology',monitor['topology']]
        if monitor.get('root'):
            argv += ['--monitor-root',monitor['root']]
        if tp and preset=='none':
            if builtin and target['task']!='generate':
                command += component_args
            product = 'runner' if builtin else 'execution'
        else:
            argv += component_args
            product = ('capture-group' if tp else 'capture') if c['capture_only'] else ('group' if tp else 'store')
        step(producer,argv+compiler_args+['--']+command,before,cwd)
    else:
        producer = 'runner' if builtin else 'command'
        product = 'runner' if builtin else 'execution'
        step(producer, command + (component_args+compiler_args if builtin else []), before, cwd)
    deps = [producer]
    if product == 'group' and output['verify']:
        for rank in range(2):
            ident = f'inspect-rank-{rank}'
            step(ident, CONTROL+['inspect',f'<{variant}.rank-{rank}.run>','--verify','--json'],[producer])
            steps[-1]['rank'] = rank
            deps.append(ident)
    elif product == 'store' and output['verify']:
        step('inspect', CONTROL + ['inspect', run, '--verify', '--json'], deps)
        deps.append('inspect')
    else:
        skipped['inspect'] = 'output.verify=false' if product in ('store','group') else f'{product} 没有待校验的 Store'
    if output['report'] and product in ('store','group','runner'):
        source = {'store':run,'group':manifest,'runner':'baseline'}[product]
        argv = CONTROL + ['report', source, '--out',str(report)]
        if attach or product == 'runner':
            argv += ['--runner-results',*runners]
        if output.get('limits') is not None:
            argv += ['--limits',input_path('limits',output['limits'])]
        step('report',argv,deps)
    else:
        skipped['report'] = {'capture':'原始采集已安排封存，导入后可生成报告','capture-group':'双 rank 原始采集在导入后可生成报告',
                             'execution':'自定义命令未提供标准 runner 输入'}.get(product,'output.report=false')
    plan = dict(input_type=target.get('task','command'), source='local', output_type=product,
                component_owner=('runner / launch (compiler)' if tp and builtin and preset=='none' else producer), vendor=vendor,
                directory=str(job),runners=runners if builtin else [],attach_runner=attach, steps=steps, skipped=skipped)
    if not builtin and c.get('tensor_ops_in'):
        plan['runtime_requirement'] = (f"采集 {c['tensor_ops_in']} 时，脚本需调用 register_model(model) 注册模型，"
                                       "并在每次前向前调用 bind_tensor_inputs([sample_id], inputs) 绑定一个样本。")
    return plan, assets
