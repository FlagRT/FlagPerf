# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Scenario templates and the workflow entry point."""
from contextlib import nullcontext
from datetime import datetime
from importlib.resources import files
import json
from pathlib import Path
import re
import shlex
import uuid
from flagprof.progress import finish_display, phase

TEMPLATES = {'quickstart':'quickstart.yaml', 'cpu-demo':'cpu_demo.yaml', 'custom-script':'custom_script.yaml',
             'compare':'compare_components.yaml', 'tp2':'tp2_launch.yaml',
             'export':'graph_export.yaml', 'batch':'batch_suite.yaml', 'replay':'replay.yaml'}


def initialize(args):
    if args.list:
        print('\n'.join(TEMPLATES))
        return 0
    if args.template not in TEMPLATES:
        raise ValueError('未知模板，可使用 init --list')
    filename = TEMPLATES[args.template]
    target = Path(args.out or ('flagprof.yaml' if args.template=='quickstart' else filename))
    template = files(__name__).joinpath('templates').joinpath(filename).read_text(encoding='utf-8')
    try:
        with target.open('w' if args.force else 'x', encoding='utf-8') as stream:
            stream.write(template)
    except FileExistsError as error:
        raise ValueError(f'{target} 已存在，可指定 --out 或 --force') from error
    print('[OK] 配置文件 ' + str(target.resolve()))
    return 0


def display(result):
    if 'error' in result:
        print('[FAIL] ' + result['error'])
    if 'output_type' in result:
        print(f"产物 {result['output_type']}   组件执行者 {result['component_owner']}   厂商 {result['vendor'] or '无需探测'}")
    if result.get('action') in ('check','dry-run'):
        print(f"[OK] 输入 {result['input_type']}")
        for row in result.get('dependencies', []):
            print(f"[OK] {row['step']}  {row['python']}  {', '.join(row['imports'])}")
        if 'model_ids' in result:
            print(f"     名册 {result['roster_source']}  模型 {', '.join(result['model_ids']) or '无匹配项'}")
            for item in result['missing_inputs']:
                print(f"[WARN] {item['model_id']} 缺少 {item['field']}  {item['path']}")
        for role,side in result.get('variants',{}).items():
            print(f"     {role}  {side['source']} → {side['output_type']}  {side.get('external') or side['directory']}")
        for role, side in result.get('variants', {'single': result}).items():
            if 'runtime_requirement' in side:
                print(f"     {role}  运行时要求  {side['runtime_requirement']}")
        if 'declaration' in result:
            print('允许变化的条件  '+json.dumps(result['declaration'],ensure_ascii=False))
        if 'planned_steps' in result:
            print('步骤  ' + ' → '.join(result['planned_steps']))
    if result.get('action') == 'dry-run':
        print(json.dumps(result['config'],ensure_ascii=False,indent=2))
    for i, step in enumerate(result.get('steps', []),1):
        code = step.get('exit_code')
        label = '[OK]' if code==0 else '[FAIL]' if code is not None else '[计划]'
        print(f"{label} {i}. {step['id']}  " + shlex.join(step['argv']))
        if result.get('action') == 'dry-run':
            print('     cwd ' + step['cwd'] + '    依赖 ' + (', '.join(step['depends_on']) or '无'))
            if step.get('if_runner_missing'):
                for alternative in step['if_runner_missing']:
                    print('     缺少共同 runner 时  ' + shlex.join(alternative['argv']))
        if 'log' in step:
            print('     ' + step['log'])
    for name, reason in result.get('skipped', {}).items():
        print(f'[SKIP] {name}  {reason}')
    for row in result.get('evidence', []):
        label = '[WARN]' if row['status'] in ('partial','unavailable','not_comparable','blocked') else '[证据]'
        if row['status'] in ('failed', 'self_reproduction_failed'):
            label = '[FAIL]'
        detail = ('  '+str(row['observations'])+' 次观测') if 'observations' in row else ''
        if row['subject'] == 'batch':
            detail = '  ' + ' '.join(f'{k}={row[k]}' for k in ('total','passed','failed','blocked'))
        elif row['subject'] == 'batch.model':
            detail = '  ' + row['model_id']
        location=' '.join(str(row[k]) if k=='variant' else 'rank '+str(row[k]) for k in ('variant','rank') if row.get(k) is not None)
        print(f"{label} {location} {row['subject']}  {row['status']}{detail}  " + ' / '.join(map(str,row['reasons'])))
    for item in result.get('artifacts', []):
        print(f"{item['type']}  {item['path']}")
    if 'job_dir' in result and 'exit_code' in result:
        print('摘要  ' + str(Path(result['job_dir'])/'summary.json'))


def dispatch(args):
    try:
        if args.verb == 'init':
            return initialize(args)
        from .loader import load, value_sources
        from .executor import execute, reserve
        filename = Path(args.config).expanduser()
        source = filename.parent.resolve() / filename.name
        config, origins = load(source,args.overrides,args.out)
        if config.get('mode') in ('export', 'batch'):
            from .suites import prepare
        elif config.get('mode') == 'replay':
            from .replay import prepare
        else:
            from .comparison import prepare
        out = config['output']
        name = re.sub(r'[^\w.-]+','-',config.get('name',config.get('mode','single'))).strip('-.') or 'single'
        job = Path(out.get('job_dir') or Path(out['dir'])/(name+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
        if not out.get('job_dir') and job.exists():
            job = job.with_name(job.name+'-'+uuid.uuid4().hex[:8])
        with nullcontext() if args.dry_run else phase('检查工作流配置'):
            plan,assets=prepare(config,origins,source,job)
        origins = value_sources(config, origins)
        if args.check or args.dry_run:
            result = dict(action='check' if args.check else 'dry-run',status='ready',**plan)
            if args.dry_run:
                result.update(config=config,value_sources=origins,job_dir=str(job))
            else:
                from .dependencies import check_dependencies
                with phase('检查运行依赖'):
                    result['dependencies'] = check_dependencies(config, plan, assets)
                result['planned_steps'] = [s['id'] for s in result.pop('steps')]
        else:
            reserved = reserve(job, not out.get('job_dir'))
            if reserved!=job:
                job=reserved
                config,origins=load(source,args.overrides,args.out)
                plan,assets=prepare(config,origins,source,job)
                origins=value_sources(config,origins)
            result = execute(config,source,origins,job,plan,assets)
    except (ValueError,OSError,KeyError,TypeError) as error:
        result = dict(status='failed',exit_code=2,error=str(error))
    finish_display()
    if getattr(args,'json',False):
        print(json.dumps(result,ensure_ascii=False))
    else:
        display(result)
    return result.get('exit_code',0)
