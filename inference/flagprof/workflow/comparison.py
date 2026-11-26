# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Resolve independent comparison sides and build their common report inputs."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

from .loader import LOCAL, ROLES, existing
from .planner import CONTROL, compile_plan
from .preflight import check, check_destination, overlap
from .schema import fill, normalize


def local_plan(side, origins, source, job, role='single'):
    attach = normalize(side, origins)
    vendor = check(side,source,job)
    if vendor and side['hardware']['vendor']=='auto':
        side['hardware']['vendor']=vendor
        origins['hardware.vendor']='auto (plugins)' if side.get('collection',{}).get('plugins') else 'auto (SMI)'
    return compile_plan(side,job,vendor,attach,variant=role)


def report_metadata(path):
    from .report_input import report_info
    if path.name=='result.json':
        path=path.parent
    doc=report_info(path/'result.json')
    schema=doc.get('schema')
    summary=doc.get('runner_summary') or {}
    panels=doc.get('results',{})
    if schema=='flagprof-group-report/1':
        children=[(path/f'rank-{r}'/'result.json').is_file() for r in range(2)]
        from flagprof.report.performance import match_runner, runner_rank
        paired=set()
        for row in summary.get('sources',[]):
            identity=row.get('identity') or {}
            expected=row.get('store_identity')
            rank=runner_rank(row,identity,required=True)
            if rank not in (0,1) or rank in paired:
                raise ValueError(f'已有组报告 runner rank 重复或超出范围 {rank}')
            if expected:
                if expected.get('rank',rank)!=rank:
                    raise ValueError(f'已有组报告 rank {rank} Store 身份不一致')
                match_runner(identity,expected,{'source_sha256':row.get('flaggems_source_sha256')},rank)
                paired.add(rank)
        observed={r.get('rank') for r in summary.get('groups',[]) if r.get('observations',0)>0}
        runner=paired=={0,1} and {0,1}.issubset(observed)
        if not runner and not all(children):
            raise ValueError('组报告缺少可比较的 runner 摘要或 rank 子报告')
        return dict(report=str(path),group=True,runner=runner,children=all(children),
                    status=doc.get('status'),issues=doc.get('issues',[]))
    if schema=='runner-result/1' and summary.get('groups') or schema=='p1-result/1' and (
            summary.get('groups') or 'perf.device' in panels or 'perf.gap' in panels):
        return dict(report=str(path),group=False,runner=bool(summary.get('groups')),children=False)
    raise ValueError('existing_report 需要单侧性能报告、runner baseline 或组报告 ' + str(path))


def store_meta(path):
    manifest=path/'manifest.json'
    if not manifest.is_file():
        raise ValueError('需要已导入的 Store 或 group.json，原始 capture 须先导入 ' + str(path))
    data=json.loads(manifest.read_text())
    if not data.get('run_id') or not data.get('schema_version') or not (path/'writers').is_dir():
        raise ValueError('输入缺少 Store 身份 ' + str(manifest))
    return data


def oracle_task(path, rank=None, backend=None):
    from .receipts import store_states
    states=store_states(path)
    identities=[s for s in states if s.get('oracle_identity')]
    if len(identities)!=1:
        raise ValueError('Oracle 缺少可核对的精度工作负载身份 '+str(path))
    state=identities[0]
    identity=state['oracle_identity']
    if not state.get('tensor') and not identity.get('boundaries'):
        raise ValueError('Oracle 需要 accuracy Store，缺少张量边界 '+str(path))
    if rank is not None and (identity.get('rank',rank)!=rank or identity.get('comm_backend')!=backend
                             or identity.get('parallelism',{}).get('world_size')!=2):
        raise ValueError(f'rank {rank} 与 Oracle 组清单不一致')
    return identity.get('task','embedding')


def run_metadata(path, oracle):
    from flagprof.report.group import validate_manifest
    if (path/'group.json').is_file():
        path=path/'group.json'
    if path.is_file():
        group=validate_manifest(path)
        if group.get('status')=='captured' or oracle and group.get('status')!='complete':
            raise ValueError('Oracle 需要完整 group，capture 须先导入 ' + str(path))
        stores={}
        for row in group['ranks']:
            rank=row['rank']
            if not row.get('run') or not (path.parent/row['run']/'manifest.json').is_file():
                if oracle:
                    raise ValueError(f'group 缺少 rank {rank} Store')
                continue
            run=path.parent/row['run']
            meta=store_meta(run)
            if oracle and (row.get('status')!='complete' or meta['status']!='complete'):
                raise ValueError(f'Oracle 需要完整 rank {rank} Store')
            stores[rank]=str(run)
        if not stores or len(set(stores.values()))!=len(stores):
            raise ValueError('组清单缺少独立的逐 rank Store')
        return 'group',str(path),stores
    store_meta(path)
    return 'store',str(path),{0:str(path)}


def external_plan(side, role, output, directory, job, oracle):
    sources=[k for k in ('existing_report','existing_run') if side.get(k) is not None]
    if len(sources)!=1 or set(side)-{'label',*sources}:
        raise ValueError('已有侧只接受一个 existing_run/existing_report 和 label')
    field=sources[0]
    path=Path(side[field])
    root=path.parent if path.is_file() else path
    if overlap(root,job):
        raise ValueError('新任务目录与已有来源重叠 ' + str(root))
    plan=dict(source=field,component_owner='existing',vendor=None,directory=str(directory),steps=[],skipped={},
              runners=[],attach_runner=False)
    if field=='existing_report':
        if oracle:
            raise ValueError('Oracle 需要 Store 或完整 group.json，existing_report 只能参与性能对照')
        meta=report_metadata(path)
        if output['attach_runner_results'] is True and not meta['runner']:
            raise ValueError('显式要求 runner，已有报告缺少可用的 runner 摘要')
        plan.update(input_type='report',output_type='report',report=meta['report'],report_metadata=meta,
                    external=meta['report'])
        plan['skipped']['inspect']='已有报告没有待校验 Store'
        return plan,{}
    kind,external,stores=run_metadata(path,oracle)
    if output['attach_runner_results'] is True:
        raise ValueError('显式要求挂接 runner，已有 Store 侧请提供已挂接的 existing_report')
    plan.update(input_type=kind,output_type=kind,external=external,stores=stores)
    steps=plan['steps']
    if output['verify']:
        for rank,run in stores.items():
            steps.append(dict(id='inspect' if kind=='store' else f'inspect-rank-{rank}',variant=role,
                         rank=rank,argv=CONTROL+['inspect',run,'--verify','--json'],depends_on=[],cwd=str(directory)))
    if not oracle:
        argv=CONTROL+['report',external,'--out',str(directory/'report')]
        steps.append(dict(id='report',variant=role,argv=argv,depends_on=[s['id'] for s in steps],cwd=str(directory)))
    return plan,{}


def linked_input(path, kind, job):
    if kind=='group' and path.is_file():
        doc=json.loads(path.read_text())
        if doc.get('schema')!='flagprof-group/1':
            ranks=doc.get('ranks')
            if not isinstance(ranks,dict) or set(ranks)!={'0','1'} or not all(isinstance(v,str) for v in ranks.values()):
                raise ValueError('同侧性能链接需要 group.json 或含 rank 0、1 的清单')
            stores=[(path.parent/v).resolve() for v in ranks.values()]
            if len(set(stores))!=2 or any(overlap(p,job) for p in [path,*stores]):
                raise ValueError('同侧性能链接的 rank Store 与输出需独立')
            for store in stores:
                store_meta(store)
            return str(path)
    actual,source,stores=run_metadata(path,False)
    if actual!=kind or any(overlap(Path(s),job) for s in [source,*stores.values()]):
        raise ValueError('同侧性能链接需要同类型且独立的 Store 或完整 group')
    if actual=='group' and set(stores)!={0,1}:
        raise ValueError('同侧性能链接缺少 rank Store')
    return source


def rank_comparisons(spec):
    result=[]
    for rank in (0,1):
        argv=spec['argv'].copy()
        start=argv.index('--reports')+1
        for pos in range(start,argv.index('--declare',start)):
            argv[pos]=str(Path(argv[pos])/f'rank-{rank}')
        pos=argv.index('--out')+1
        argv[pos]=str(Path(argv[pos])/f'rank-{rank}')
        result.append(dict(spec,argv=argv,rank=rank,id=f'compare-rank-{rank}'))
    return result


def declaration(config, local, oracle):
    from flagprof.report.comparison import DECLARABLE
    compare=config['compare']
    value=compare.get('declare','auto')
    automatic=value=='auto'
    if isinstance(value,str) and not automatic:
        value=json.loads(Path(value).read_text())
    data={'differences':value} if isinstance(value,list) else deepcopy(value)
    if not automatic and (not isinstance(data,dict) or set(data)-{'differences','note'}
                          or not isinstance(data.get('differences'),list)):
        raise ValueError('compare.declare 需要 differences 数组和可选 note')
    cross_precision=oracle and not automatic and 'dtype' in data['differences']
    fields=set()
    maps={'hardware.vendor':('device',),'hardware.device':('device',),'hardware.nproc':('parallelism',),
          'hardware.comm_backend':('comm_backend','communication'),'components.flaggems.enabled':('flaggems','include'),
          'components.flaggems.include':('include',),'components.flagtree':('compiler','software.triton'),
          'components.vendor_compiler':('compiler','software.triton'),'runner.numerics':('numerics',)}
    def pick(c,key):
        for part in key.split('.'):
            c=c.get(part) if isinstance(c,dict) else None
        return c
    for (a,left),(b,right) in combinations(local.items(),2):
        for block in ('target','runner'):
            l,r=left.get(block,{}),right.get(block,{})
            for k in sorted(set(l)|set(r)):
                if oracle and block=='runner' and k=='dtype' and l.get(k)!=r.get(k):
                    if cross_precision:
                        continue
                    raise ValueError(f'{a}/{b} 工作负载不一致 runner.dtype。跨精度对照须在 compare.declare 中显式列出 dtype')
                if k!='numerics' and l.get(k)!=r.get(k):
                    raise ValueError(f'{a}/{b} 工作负载不一致 {block}.{k}')
        for key,names in maps.items():
            if pick(left,key)!=pick(right,key):
                if key=='runner.numerics' and not oracle:
                    raise ValueError('性能对照禁止改变 runner.numerics')
                fields.update(names)
    if automatic:
        data=dict(differences=sorted(fields))
    if 'note' in data and not isinstance(data['note'],str):
        raise ValueError('compare.declare.note 需要字符串')
    allowed=set(DECLARABLE) | ({'numerics'} if oracle else set())
    for name in data['differences']:
        if oracle and name=='dtype':
            continue
        if not isinstance(name,str) or name.split('.')[0] not in allowed or any(not x for x in name.split('.')):
            raise ValueError(f'compare.declare 不允许 {name}')
        if oracle and name.split('.')[0]=='parallelism' and any(s['target'].get('task')!='generate' for s in local.values()):
            raise ValueError('Oracle 仅 generate 可声明 parallelism')
    if compare.get('note') is not None:
        data['note']=compare['note']
    return data


def prepare(config, origins, source, job):
    """Compile all sides before any output is reserved or target started."""
    if config.get('mode','single')!='compare':
        plan,assets=local_plan(config,origins,source,job)
        return plan,assets
    if config.get('schema_version')!=1:
        raise ValueError('schema_version 需要整数 1')
    config.setdefault('name','compare')
    compare=config.get('compare',{})
    if not all(isinstance(compare.get(r),dict) for r in ('ref','cmp')):
        raise ValueError('compare 需要 ref 和 cmp 两侧')
    out=config['output']
    fill(out,verify=True,report=True,attach_runner_results='auto')
    if out['attach_runner_results'] not in ('auto',True,False):
        raise ValueError('output.attach_runner_results 需要 auto/true/false')
    if not out['report']:
        raise ValueError('compare 需要 output.report=true')
    from flagprof.probes.config import PRESETS
    preset=config.setdefault('profile',{}).setdefault('preset','perf')
    if preset not in (*PRESETS,'none'):
        raise ValueError('未知 profile.preset '+preset)
    oracle=preset=='accuracy'
    check_destination(job,config,source)
    local, sides, assets, steps, skips={}, {}, {}, [], {}
    limits=out.get('limits')
    if limits is not None:
        from flagprof.analyzers.workspace import Limits
        Limits(**(json.loads(Path(limits).read_text()) if isinstance(limits,str) else limits))
    for role in ROLES:
        side=compare.get(role)
        if side is None:
            continue
        directory=job/'runs'/role
        try:
            if existing(side):
                plan,files=external_plan(side,role,out,directory,job,oracle)
            else:
                values={k:v for k,v in side.items() if k in LOCAL}
                values.update(schema_version=1,profile=config['profile'],output=out)
                prefix='compare.'+role+'.'
                side_origins={k[len(prefix):]:v for k,v in origins.items() if k.startswith(prefix)}
                check_destination(job,values,source)
                plan,files=local_plan(values,side_origins,source,directory,role)
                origins.update({prefix+k:v for k,v in side_origins.items()})
                side.update({k:v for k,v in values.items() if k in LOCAL})
                local[role]=values
                if plan['output_type'] not in ('store','group','runner'):
                    raise ValueError('对照需要 Store、group 或 runner；可用 existing_report 提供已有报告')
            if plan['output_type']!='report':
                plan['report']=str(directory/'report')
            for step in plan.pop('steps'):
                if oracle and step['id']=='report':
                    plan['skipped']['report']='Oracle 直接读取 Store'
                    continue
                step.update(action=step['id'],id=role+'.'+step['id'],depends_on=[role+'.'+d for d in step['depends_on']])
                steps.append(step)
            skips.update({role+'.'+k:v for k,v in plan.pop('skipped').items()})
            sides[role]=plan
            assets.update(files)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise ValueError(f'compare.{role}: {error}') from error
    kinds={p['output_type'] for p in sides.values()}
    if oracle and (len(kinds)!=1 or not kinds<= {'group','store'}):
        raise ValueError('Oracle 需要全为 Store 或全为完整 group，当前 '+str({r:p['output_type'] for r,p in sides.items()}))
    if oracle:
        for role,p in sides.items():
            if p['source']!='existing_run':
                continue
            group=p['output_type']=='group'
            backend=json.loads(Path(p['external']).read_text())['backend'] if group else None
            try:
                tasks={oracle_task(s,r if group else None,backend) for r,s in p['stores'].items()}
                if len(tasks)!=1:
                    raise ValueError('组内任务身份不一致')
                p['task']=tasks.pop()
            except (ValueError,OSError) as error:
                raise ValueError(f'compare.{role}: {error}') from error
    declare=declaration(config,local,oracle)
    if oracle and any(n.split('.')[0]=='parallelism' for n in declare['differences']):
        if any(p.get('task', local.get(role,{}).get('target',{}).get('task'))!='generate' for role,p in sides.items()):
            raise ValueError('Oracle 仅 generate 可声明 parallelism')
    decl_path=job/'inputs/declare.json'
    assets[str(decl_path)]=declare
    comparison=job/'compare-report'
    argv=CONTROL+['report','oracle' if oracle else 'compare']
    if oracle:
        for role,p in sides.items():
            ref=p.get('external') or (str(Path(p['directory'])/'store/group.json') if p['output_type']=='group' else f'<{role}.run>')
            argv+=['--'+role,ref]
        for key in ('perf_ref_run','perf_cmp_run'):
            if compare.get(key):
                path=Path(compare[key])
                argv+=['--'+key.replace('_','-'),linked_input(path,next(iter(kinds)),job)]
    else:
        if compare.get('perf_ref_run') or compare.get('perf_cmp_run'):
            raise ValueError('perf_ref_run/perf_cmp_run 仅用于 Oracle')
        argv+=['--reports',*[p['report'] for p in sides.values()]]
    argv+=['--declare',str(decl_path),'--out',str(comparison)]
    if limits is not None:
        path=limits if isinstance(limits,str) else str(job/'inputs/limits.json')
        if isinstance(limits,dict):
            assets[path]=limits
        argv+=['--limits',path]
        for s in steps:
            if s['action']=='report' and '--limits' not in s['argv']:
                s['argv']+=['--limits',path]
    final=dict(id='compare',action='oracle' if oracle else 'compare',variant='comparison',argv=argv,
               cwd=str(job),depends_on=[s['id'] for s in steps])
    if not oracle and all(p['output_type']=='group' or p.get('report_metadata',{}).get('group') for p in sides.values()):
        if any(not p.get('report_metadata',{}).get('runner',p['attach_runner']) for p in sides.values()):
            if any(not p.get('report_metadata',{}).get('children',True) for p in sides.values()):
                raise ValueError('性能对照缺少共同的 runner 摘要或同 rank 子报告')
            steps.extend(rank_comparisons(final))
        else:
            if any(p['source']=='local' and out['attach_runner_results']=='auto' for p in sides.values()):
                final['if_runner_missing']=rank_comparisons(final)
            steps.append(final)
    else:
        steps.append(final)
    return dict(input_type='comparison',source='comparison',output_type='oracle' if oracle else 'performance',
                component_owner={r:p['component_owner'] for r,p in sides.items()},vendor=None,variants=sides,
                declaration=declare,declaration_source=compare.get('declare','auto'),steps=steps,skipped=skips),assets
