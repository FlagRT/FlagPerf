"""Group reports validate identities and collective sequence through StoreView."""
from html import escape
import json
from pathlib import Path
import tempfile

from flagprof.store import StoreView
from flagprof.analyzers.workspace import Workspace
from flagprof.analyzers.generation import trajectories
from . import generate


def validate_manifest(path):
    path = Path(path).resolve()
    state = json.loads(path.read_text())
    if state.get('schema') != 'flagprof-group/1' or state.get('world_size') != 2:
        raise ValueError('unsupported group manifest')
    rows = state.get('ranks', [])
    if [r['rank'] for r in rows] != [0, 1]:
        raise ValueError('missing, duplicate or unordered group ranks')
    for row in rows:
        if row.get('group_id') != state['group_id']:
            raise ValueError('worker group identity mismatch')
        if row.get('run'):
            run = Path(row['run'])
            if run.is_absolute() or '..' in run.parts or not (path.parent/run).resolve().is_relative_to(path.parent):
                raise ValueError('group Store path must stay relative to its delivery')
    return state


def check_ranks(states, group):
    if len(states) != 2:
        raise ValueError('two rank states required')
    if states[0]['identity'] != states[1]['identity']:
        raise ValueError('rank workload identity mismatch')
    if states[0]['trajectories'] != states[1]['trajectories']:
        raise ValueError('rank token or step control mismatch')
    if states[0].get('tensors') != states[1].get('tensors'):
        raise ValueError('rank replicated boundary mismatch')
    left, right = states[0]['calls'], states[1]['calls']
    if set(left) != set(right):
        raise ValueError('rank collective sequence missing or group mismatch')
    for key in left:
        a, b = left[key], right[key]
        for name in ('op','lib','bytes','dtype','count'):
            if a.get(name) != b.get(name):
                raise ValueError('rank collective metadata mismatch: '+name)
        if not key[0].startswith(group+'/'):
            raise ValueError('collective group identity mismatch')


def generate_group(manifest, out=None, *, work=None):
    manifest = Path(manifest).resolve()
    group = validate_manifest(manifest)
    out = Path(out).resolve() if out else manifest.parent.with_name(manifest.parent.name+'-report')
    if out.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(out):
        raise ValueError('group report must be outside input delivery')
    out.mkdir(parents=True, exist_ok=True)
    if work:
        Path(work).mkdir(parents=True, exist_ok=True)
    states, issues, reports = [], [], []
    if group.get('status')!='complete':
        issues.append('group execution did not complete')
    for row in group['ranks']:
        rank = row['rank']
        if not row.get('run'):
            issues.append('rank '+str(rank)+' has no Store')
            continue
        run = manifest.parent/row['run']
        reports.append(dict(rank=rank,**generate(run, out/('rank-'+str(rank)), work=work)))
        with StoreView(run) as view, tempfile.TemporaryDirectory(dir=work) as directory:
            workspace = Workspace(view, directory)
            try:
                workspace.ingest()
                from flagprof.analyzers.oracle import identity
                observed = identity(workspace)
                observed.pop('device', None)
                if observed.get('comm_backend')!=group['backend'] or observed.get('parallelism',{}).get('world_size')!=2:
                    raise ValueError('actual TP backend or world size differs from group manifest')
                calls = {}
                sequences = {}
                tensors = {}
                for _, _, _, event in workspace.events(('tensor',)):
                    key=event['call_key']
                    if key in tensors:raise ValueError('duplicate rank tensor call_key')
                    tensors[key]={k:event.get(k) for k in ('shape','dtype','digest','mask')}
                if observed.get('generation',{}).get('mode')=='accuracy' and not tensors:
                    raise ValueError('rank accuracy tensor evidence missing')
                for _, _, _, event in workspace.events(('comm',)):
                    if event.get('rank') != rank:
                        raise ValueError('event rank differs from group manifest')
                    key = event['group'], event['group_seq']
                    if key in calls:
                        raise ValueError('duplicate collective group sequence')
                    if key[1] != sequences.get(key[0], 0):
                        raise ValueError('collective group sequence missing or out of order')
                    sequences[key[0]] = key[1]+1
                    calls[key] = event
                    if event.get('ext',{}).get('communication',{}).get('role')=='unknown':
                        issues.append('rank '+str(rank)+' has an unregistered communication group')
                    if event.get('ext',{}).get('communication',{}).get('role')=='model' and event['lib']!=group['backend']:
                        raise ValueError('actual collective backend differs from group manifest')
                states.append(dict(rank=rank, identity=observed, trajectories=trajectories(workspace), calls=calls,tensors=tensors))
            finally:
                workspace.close()
    if len(states) == 2:
        try:
            check_ranks(states, group['group_id'])
        except ValueError as error:
            issues.append(str(error))
    from flagprof.analyzers.gap import comparable
    device_calls={};contexts={};request_rows=[]
    for report in reports:
        result=json.loads((out/('rank-'+str(report['rank']))/'result.json').read_text())
        panel=result['results'].get('perf.gap',{})
        context=dict(panel.get('coverage',{}));context['workload_identity']=dict(context.get('workload_identity') or {})
        context['workload_identity'].pop('device',None)
        contexts[report['rank']]=context
        request_rows.extend(dict(r,rank=report['rank']) for r in panel.get('data',[]) if r.get('kind')=='project_request')
        rows=out/('rank-'+str(report['rank']))/'exports/communication-device-calls.jsonl'
        device_calls[report['rank']]={(r['group'],r['group_seq']):r for r in map(json.loads,rows.read_text().splitlines())}
    comparison=comparable(contexts.get(0,{}),contexts.get(1,{}));aligned=[]
    keys=set(device_calls.get(0,{})) | set(device_calls.get(1,{}))
    for key in sorted(keys):
        a=device_calls.get(0,{}).get(key);b=device_calls.get(1,{}).get(key)
        values=[r['device_union_ns'] if r else None for r in (a,b)]
        row=dict(group=key[0],group_seq=key[1],rank_device_union_ns=values,comparison=comparison,
                 evidence=[r.get('evidence',[]) if r else [] for r in (a,b)])
        if a and b and comparison['status']=='comparable':
            row.update(cross_rank_min_observed_ns=min(values),cross_rank_max_observed_ns=max(values),cross_rank_difference_ns=max(values)-min(values))
        else:row['reason']='missing_rank_device_link' if not (a and b) else 'identity_not_comparable'
        aligned.append(row)
    counts = []
    for state in states:
        model = [e for e in state['calls'].values()
                 if e.get('ext',{}).get('communication',{}).get('role') == 'model'
                 and e['ext']['communication'].get('generation')]
        allreduce = sum(e['op']=='allreduce' for e in model)
        allgather = sum(e['op']=='allgather' for e in model)
        steps = sum(len(q['steps']) for q in state['trajectories'].values())
        counts.append(dict(rank=state['rank'], steps=steps, allreduce=allreduce, allgather=allgather,
                           expected_allreduce=72*steps, expected_allgather=steps,
                           model_count_matches=allreduce==72*steps and allgather==steps))
        if not counts[-1]['model_count_matches']:
            issues.append('rank '+str(state['rank'])+' model collective count mismatch')
    result = dict(schema='flagprof-group-report/1', group_id=group['group_id'], backend=group['backend'],
                  execution_status=group['status'], status='partial' if issues else 'complete', issues=issues,
                  rank_counts=counts, ranks=reports,
                  device_link_status='linked' if aligned and all('cross_rank_min_observed_ns' in r for r in aligned) else 'partial' if aligned else 'explicit_link_unavailable', device_duration_ns=None,
                  device_link_scope='explicitly_linked_collectives; control calls retain unavailable state in rank reports',
                  unlinked_logical_calls={s['rank']:len(s['calls'])-len(device_calls.get(s['rank'],{})) for s in states},
                  device_calls=aligned,comparison=comparison,clocks={rank:value.get('clock') for rank,value in contexts.items()},
                  request_windows=request_rows,
                  device_timing_note='跨 rank 最小观测占用、最大值与差值；差值不等于额外等待，最小值不代表纯通信下界',
                  timing_note='host enqueue duration is not device communication duration')
    (out/'exports').mkdir(exist_ok=True)
    with (out/'exports/communication-aligned.jsonl').open('w') as stream:
        for row in aligned:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    from . import html_rows
    with (out/'report.html').open('w') as page:
        page.write('<!doctype html><meta charset="utf-8"><title>TP2 通信对照</title>'
            '<style>body{font:16px sans-serif;max-width:1400px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccd;padding:7px}pre{white-space:pre-wrap}h2{margin-top:32px}</style>'
            '<h1>TP2 通信与生成</h1><p>状态：'+escape(result['status'])+'；设备关联：'+escape(result['device_link_status'])+
            '。按同一 (group, group_seq) 核对两 rank；模型通信与控制组分列。</p><p>'+escape(result['device_timing_note'])+
            '。时间单位 ns；主机调用耗时与设备占用分别解释。</p>')
        page.write('<h2>逻辑通信计数</h2>');html_rows(page,'group.counts',counts,2)
        page.write('<h2>完整请求窗口</h2><p>各 rank 按自己的来源时钟映射；误差界与映射状态分别记录。</p>')
        html_rows(page,'group.request_windows',[
            {k:r.get(k) for k in ('rank','request_id','coverage_status','start_ns','end_ns','window_ns','all_activity_union_ns','blank_ns')}
            for r in request_rows],100)
        page.write('<p>未关联逻辑调用（含控制组）：'+escape(json.dumps(result['unlinked_logical_calls']))+'</p>')
        page.write('<h2>同调用的设备占用</h2><p>显示前 100 条；全部 '+str(len(aligned))+
            ' 条见 <a href="exports/communication-aligned.jsonl">机器导出</a>。</p>')
        html_rows(page,'communication.aligned',aligned,100)
        page.write('<details><summary>身份、时钟与诊断范围</summary><pre>'+escape(json.dumps(
            {k:result[k] for k in ('comparison','clocks','issues','device_link_scope')},ensure_ascii=False,indent=2))+'</pre></details>')
        page.write(''.join('<p><a href="rank-'+str(row['rank'])+'/report.html">rank '+str(row['rank'])+' 诊断</a></p>' for row in reports))
    return dict(output=str(out), status=result['status'], issues=issues)
