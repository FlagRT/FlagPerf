# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Plan an operator replay and bind its two generated Stores to the final Oracle."""
import json
from pathlib import Path
import re

from flagprof.components.flaggems import gems_identity
from flagprof.offline.limits import read_json
from flagprof.replay import digest
from .comparison import linked_input, store_meta
from .planner import CONTROL
from .preflight import check_destination, exists, overlap
from .receipts import store_states
from .schema import allowed_keys


def side_inputs(side, role, job, origins):
    prefix = 'replay.' + role
    if not side.get('run'):
        raise ValueError(prefix + '.run 需要一个 accuracy Store 目录')
    path = Path(side['run'])
    if overlap(path, job) or path.is_file() or (path/'group.json').exists():
        raise ValueError(prefix + '.run 需要与输出目录分开的单个 accuracy Store')
    if store_meta(path)['status'] != 'complete':
        raise ValueError(prefix + '.run 需要完整的 accuracy Store')
    identities = [state['oracle_identity'] for state in store_states(path) if state.get('oracle_identity')]
    if len(identities) != 1 or not isinstance(identities[0].get('device'), str):
        raise ValueError(prefix + '.run 缺少唯一的 Oracle 设备身份')
    identity = identities[0]
    recorded = identity['device']
    if side.get('device') is None:
        side['device'] = recorded
        origins[prefix + '.device'] = 'source Store'
    device = side['device']
    if (not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*(?::[0-9]+)?', device)
            or device.split(':')[0] != recorded.split(':')[0]):
        raise ValueError(f'{prefix}.device 需要与源设备 {recorded} 同类型，当前为 {device}')
    compiler = side.get('vendor_compiler')
    if compiler and overlap(exists(compiler, prefix + '.vendor_compiler', 'dir'), job):
        raise ValueError(prefix + '.vendor_compiler 与输出目录重叠')
    return identity


def prepare(config, origins, source, job):
    if config.get('schema_version') != 1:
        raise ValueError('schema_version 需要整数 1')
    allowed_keys('replay', config, ('schema_version', 'name', 'mode', 'output', 'replay'))
    allowed_keys('output', config['output'], ('dir', 'job_dir', 'limits'))
    check_destination(job, config, source)
    options = config.get('replay', {})
    key = options.get('call_key', '')
    if not key.strip() or '\0' in key:
        raise ValueError('replay.call_key 需要算子行的完整调用键')
    sides = ('ref', 'cmp')
    identities = {role: side_inputs(options.get(role, {}), role, job, origins) for role in sides}
    sources = ['--ref', options['ref']['run'], '--cmp', options['cmp']['run']]
    package = str(job/'package')
    steps = [dict(id='prepare', variant='replay', action='replay.prepare', depends_on=[], cwd=str(job),
                  argv=CONTROL + ['replay', 'prepare', *sources, '--call-key', key, '--out', package])]
    oracle = CONTROL + ['report', 'oracle', *sources]
    for role in sides:
        side, identity = options[role], identities[role]
        argv = CONTROL + ['replay', 'run', package, '--side', role, '--device', side['device'],
                          '--store', str(job/role/'store')]
        if side.get('vendor_compiler'):
            argv.extend(['--vendor-compiler', side['vendor_compiler']])
        compiler = identity.get('compiler') or {}
        selection = None
        if compiler.get('status') == 'available':
            selection = dict(flagtree='on' if compiler['provider'] == 'flagtree' else 'off',
                             vendor_compiler=side.get('vendor_compiler'))
        steps.append(dict(id='run.' + role, variant=role, action='replay.run', depends_on=['prepare'],
                          cwd=str(job), argv=argv, compiler_selection=selection,
                          flaggems=gems_identity(identity.get('flaggems'))[0] == 'on'))
        oracle.extend(['--replay-' + role + '-run', f'<replay.{role}.run>'])
        field = f'perf_{role}_run'
        if options.get(field):
            oracle.extend(['--perf-' + role + '-run', linked_input(Path(options[field]), 'store', job)])
    declare = options.get('declare', {})
    differences = declare.get('differences', [])
    if not isinstance(differences, list) or not all(isinstance(k, str) for k in differences):
        raise ValueError('replay.declare.differences 需要身份字段名称数组')
    declaration_path = str(job/'inputs/declare.json')
    assets = {declaration_path: declare}
    oracle.extend(['--declare', declaration_path, '--out', str(job/'report')])
    limits = config['output'].get('limits')
    if limits is not None:
        from flagprof.analyzers.workspace import Limits
        Limits(**(read_json(limits) if isinstance(limits, str) else limits))
        if isinstance(limits, dict):
            limits_path = str(job/'inputs/limits.json')
            assets[limits_path] = limits
        else:
            limits_path = limits
        oracle.extend(['--limits', limits_path])
    steps.append(dict(id='oracle', action='oracle', variant='replay', depends_on=['run.ref', 'run.cmp'],
                      cwd=str(job), argv=oracle))
    return dict(input_type='accuracy Stores', output_type='oracle-report', component_owner='replay run',
                directory=str(job), vendor=None, runners=[], skipped={}, declaration=declare, steps=steps), assets


def consume_receipt(execution, step, log):
    """Only accept a producer's final receipt for the exact package, key and side."""
    from .executor import artifact, evidence
    from flagprof.offline.limits import METADATA_BYTES

    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - METADATA_BYTES))
        lines = stream.read().splitlines()
    try:
        receipt = json.loads(lines[-1]) if lines else None
    except (ValueError, UnicodeError):
        receipt = None
    if not isinstance(receipt, dict):
        raise ValueError('replay 缺少结构化回执 ' + str(log))
    key = execution.config['replay']['call_key']
    package = execution.job/'package'
    manifest = package/'replay.json'
    summary = execution.summary
    if step['action'] == 'replay.prepare':
        if (receipt.get('package') != str(package) or receipt.get('call_key') != key
                or receipt.get('manifest') != digest(manifest)):
            raise ValueError('replay prepare 回执与请求不一致 ' + str(log))
        artifact(summary, 'replay-package', manifest)
        evidence(summary, 'replay.prepare', 'complete', [], log, call_key=key)
        return
    role = step['variant']
    if not isinstance(receipt.get('run'), str) or not Path(receipt['run']).is_absolute():
        raise ValueError('replay run 回执缺少 Store 路径 ' + str(log))
    run = Path(receipt['run']).resolve()
    if run.parent != execution.job/role/'store' or store_meta(run)['status'] != 'complete':
        raise ValueError('replay run 回执指向意外的 Store ' + str(run))
    path = run/'replay-result.json'
    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError('replay run 结果格式错误 ' + str(path))
    status = result.get('status')
    cases = result.get('cases', {})
    if (result.get('schema') != 'operator-replay-result/1' or result.get('side') != role
            or result.get('device') != execution.config['replay'][role]['device']
            or result.get('package') != digest(manifest) or result.get('call_key') != key
            or status not in ('complete', 'self_reproduction_failed') or receipt.get('status') != status
            or not isinstance(cases, dict) or set(cases) != {'ref', 'cmp'}
            or not all(isinstance(case, dict) for case in cases.values()) or receipt.get('cases') != cases):
        raise ValueError('replay run 回执与请求不一致 ' + str(path))
    self_check = cases[role].get('self_reproduction')
    if not isinstance(self_check, dict) or (self_check.get('digest_equal') is True) != (status == 'complete'):
        raise ValueError('replay run 回执与自复现结果不一致 ' + str(path))
    artifact(summary, 'replay-store', run, variant=role)
    artifact(summary, 'replay-result', path, variant=role)
    reasons = [] if status == 'complete' else [f'{role} 侧自复现失败，见 {path}']
    evidence(summary, 'replay.run', status, reasons, path, variant=role, call_key=key)
    if reasons:
        raise ValueError(reasons[0])
    execution.locations[f'<replay.{role}.run>'] = run
