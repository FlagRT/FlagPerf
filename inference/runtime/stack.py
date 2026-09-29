# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Expand one comparison axis while sealing each compiler/communication identity."""
from copy import deepcopy
import time
from pathlib import Path
import yaml
from runtime.common import read_json, write_json, digest, identity_differences

COMPONENTS=('flaggems','flagtree','flagcx')


def expand(cfg,command):
    requested={k:cfg[command].get(k,'off') for k in COMPONENTS}
    axes=[k for k,v in requested.items() if v=='both']
    if len(axes)>1 or any(v not in ['off','on','both'] for v in requested.values()):
        raise ValueError('exactly zero or one comparison component is allowed')
    axis=axes[0] if axes else None
    labels=['off','on'] if axis else [requested['flaggems']]
    return axis,{side:{k:(side if k==axis else v) for k,v in requested.items()} for side in labels}


def common_identity(identity,axis):
    """Only declared implementation fields may vary; never ignore all environment differences."""
    value=deepcopy(identity)
    if axis=='flagtree':
        value.pop('compiler',None); value.pop('triton_import',None)
        compiler_packages={'triton','triton-ascend','flagtree'}
        value['packages']=[p for p in value['packages'] if p[0].lower().replace('_','-') not in compiler_packages]
    if axis=='flagcx':
        value.pop('communication_backend',None)
        if 'parallelism' in value: value['parallelism'].pop('backend',None)
    return value


def assert_observation(ctx,value):
    expected=ctx['key']
    if value.get('identity_key')!=expected:
        raise ValueError('worker identity differs from its selected component context')
    for rank, row in value.get('ranks',{}).items():
        if row.get('identity_key')!=expected:
            raise ValueError('rank '+rank+' identity differs from selected context')


def fully_hit(value,include):
    if value.get('status')!='completed' or value.get('numerical_anomalies'):
        return False
    from runtime.preview import usable
    if not usable(value): return False
    calls=value.get('route',{}).get('actual_function_calls',{})
    ranks=value.get('route',{}).get('rank_function_calls',{})
    return bool(include) and all(calls.get(n,0)>0 for n in include) and all(
        all(c.get(n,0)>0 for n in include) for c in ranks.values())


def load_policy(cfg,root,contexts):
    active=[c for c in contexts.values() if c['stack']['flaggems']=='on']
    if not active: return
    policy=yaml.safe_load(Path(cfg['policy']['path']).read_text())
    if policy.get('schema_version')!=3 or policy.get('status')!='verified':
        raise ValueError('component execution requires a new verified preview policy (schema 3; run a fresh preview)')
    include=policy.get('include')
    if not isinstance(include,list) or not include or any(not isinstance(n,str) for n in include) or len(set(include))!=len(include):
        raise ValueError('invalid common FlagGems selection')
    if policy.get('verified_selection_sha256')!=digest(include):
        raise ValueError('policy selection changed after verification')
    for key, profile in policy.get('profiles', {}).items():
        if 'identity' in profile and digest(profile['identity']) != key:
            raise ValueError('policy identity snapshot does not match its identity key')
    for ctx in active:
        entry=policy.get('profiles',{}).get(ctx['key'])
        if not entry or entry.get('status')!='verified' or entry.get('selection_sha256')!=digest(include):
            actual = read_json(ctx['prepared']/'identity.json') if ctx.get('prepared') else None
            candidates = []
            for key, profile in policy.get('profiles', {}).items():
                if actual is not None and profile.get('identity'):
                    candidates.append({'profile': key, 'changed_fields': identity_differences(profile['identity'], actual)})
            detail = {'expected_profile': ctx['key'], 'candidates': candidates,
                      'reason': '策略身份不匹配；当前环境必须重新 preview。旧策略无身份快照时无法展开字段差异。'}
            write_json(root/'policy-mismatch.json', detail)
            fields = min(candidates, key=lambda x:len(x['changed_fields']))['changed_fields'] if candidates else []
            explanation = ', '.join(fields[:20]) if fields else '旧策略缺少身份快照，无法展开具体差异'
            raise ValueError('policy identity mismatch: '+explanation+'; see policy-mismatch.json; run preview again')
        ctx['include']=include
        ctx['policy']=dict(entry,include=include)
    (root/'policy.yaml').write_text(yaml.safe_dump(policy,sort_keys=False,allow_unicode=True))


def preview_stack(cfg,root,contexts,invoke,progress,restored=None):
    from runtime.preview import run_profiles
    return run_profiles(cfg,root,contexts,invoke,progress,restored)


def audits(root,contexts,invoke,progress):
    from runtime.coordinator import route_summary
    result={}; summaries={}
    for side,ctx in contexts.items():
        progress.phase=f'performance/component-audit/{side}'
        folder=root/'audit'/side
        value=invoke(ctx['cfg'],root,'forward',folder,ctx['prepared'],ctx['include'],True)
        result[side]=value
        if value.get('status')!='completed' or value.get('numerical_anomalies'):
            return result,summaries,f'{side} component audit failed'
        assert_observation(ctx,value)
        if ctx['include'] is not None:
            if not fully_hit(value,ctx['include']): return result,summaries,f'{side}: not every rank hit every selected function'
            native=invoke(ctx['cfg'],root,'forward',folder/'native',ctx['prepared'],None,True)
            if native.get('status')!='completed' or native.get('numerical_anomalies'):
                return result,summaries,f'{side} auxiliary native audit failed'
            assert_observation(ctx,native)
            summaries[side]=route_summary(native,value,ctx['policy'])
    return result,summaries,None


def run_stack(command,cfg,root,progress,invoke):
    axis,profiles=expand(cfg,command)
    import sys
    print('[plan] '+command+'; comparison='+str(axis)+'; '+str(profiles), file=sys.stderr, flush=True)
    if command == 'preview':
        print('[plan] preview 预算 '+str(cfg['preview']['budget_seconds'])+' 秒，按环境均分；基线、候选及所有复验共享本次预算', file=sys.stderr, flush=True)
    if command == 'performance':
        print('[plan] warmup/measure/repeats='+ '/'.join(str(cfg['performance'][k]) for k in ['warmup_rounds','measure_rounds','repeats'])+'；先独立取证，再正式计时；TP 通信为另一次采样', file=sys.stderr, flush=True)
        if cfg['performance'].get('level') == 'layer':
            print('[plan] layer='+str(cfg['performance']['layers'])+'；无插桩基准、Event计时、profiler和分组显存分别执行；诊断轮数='+str(cfg['performance']['layer_profile_rounds']), file=sys.stderr, flush=True)
    if axis == 'flagcx' and cfg['runtime'].get('parallelism') != 'tp':
        print('[notice] 单卡没有模型集合通信，FlagCX 比较不适用；继续保留执行结果，不能评价组件收益', file=sys.stderr, flush=True)
    env=cfg.get('vendors',{}).get(cfg['runtime'].get('vendor'),{}).get('env',{})
    print('[plan] 队列设置 '+str({k:env[k] for k in ['TASK_QUEUE_ENABLE','TRITON_ENABLE_TASKQUEUE'] if k in env}), file=sys.stderr, flush=True)
    if cfg['runtime'].get('parallelism') == 'tp' and any(p['flagcx']=='on' for p in profiles.values()):
        print('[notice] FlagCX 原生组保留至一次性 worker 退出；未验证长驻进程反复建组/销毁', file=sys.stderr, flush=True)
    contexts={}; environments={}; shared=None; restored=None
    copy_started=time.monotonic()
    if command == 'preview' and cfg['preview'].get('resume_from'):
        from runtime.preview import load_resume, import_resume
        restored = load_resume(cfg['preview']['resume_from'])
        shared = import_resume(cfg['preview']['resume_from'],root,restored)
    copy_seconds=time.monotonic()-copy_started
    prepare_started=time.monotonic()
    for side,stack in profiles.items():
        selected=deepcopy(cfg); selected['_stack']=stack
        key=(stack['flagtree'],stack['flagcx'])
        if key not in environments:
            folder=root/'prepared' if not environments else root/'prepared'/side
            progress.phase=f'prepare/{side}'
            value=invoke(selected,root,'prepare',folder,shared)
            if value['status']!='completed': raise RuntimeError('prepare: '+value.get('error','failed'))
            environments[key]=folder
            if len(environments)==1: shared=folder
        folder=environments[key]
        contexts[side]={'cfg':selected,'stack':stack,'prepared':folder,
                       'key':read_json(folder/'identity-key.json')['key'],'include':None,'policy':None}
    common={side:common_identity(read_json(c['prepared']/'identity.json'),axis) for side,c in contexts.items()}
    if len({digest(v) for v in common.values()})!=1:
        left,right=common.values()
        differences=[k for k in set(left)|set(right) if left.get(k)!=right.get(k)]
        raise ValueError('comparison changes uncontrolled conditions: '+', '.join(sorted(differences)))
    manifest={'comparison_component':axis,'component_profiles':profiles,'common_identity_key':digest(next(iter(common.values()))),
              'identities':{s:c['key'] for s,c in contexts.items()}}
    write_json(root/'comparison-context.json',manifest)
    prepare_seconds=time.monotonic()-prepare_started
    if command=='preview': result=preview_stack(cfg,root,contexts,invoke,progress,restored)
    else:
        load_policy(cfg,root,contexts)
        from runtime.coordinator import accuracy,performance
        result=(accuracy if command=='accuracy' else performance)(cfg,root,shared,progress,contexts=contexts)
    result['preparation_seconds']={'resume_validation_and_copy':copy_seconds,'prepare':prepare_seconds,
                                   'cache_import':result.get('cache_import_seconds',0)}
    result.update(manifest)
    if cfg['runtime'].get('parallelism')=='tp': result.update(parallelism='tp',devices=cfg['runtime']['devices'])
    return result
