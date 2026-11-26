# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Group reports validate identities and collective sequence through StoreView."""
from html import escape
import json
from pathlib import Path
import tempfile
import re

from flagprof.store import StoreView
from flagprof.analyzers.workspace import Workspace, Limits
from flagprof.analyzers.generation import trajectories
from . import generate


def replicated_boundary(event):
    """Local operator/shard tensors must never be compared between ranks."""
    if event.get('boundary') in ('pooled','output','hidden','token'):
        return True
    scope = event.get('call_key','').split('#')[0]
    return event.get('boundary') == 'out' and bool(re.fullmatch(
        r'<root>\.(?:model\.)?(?:layers\.\d+|embedding|embed_tokens|norm|lm_head)',scope))


def collective_forward_phase(workspace, run, writer, corr):
    """A host collective belongs to a phase only through its recorded module ancestry."""
    seen = set()
    while corr and corr not in seen:
        seen.add(corr)
        matches = workspace.db.execute(
            "select payload from events where run=? and w=? and corr=? and type='span' limit 2",
            (run, writer, str(corr))).fetchall()
        if len(matches) != 1:
            break
        span = json.loads(matches[0][0])
        if span.get('ph') != 'x' or span.get('cat') != 'module':
            break
        if span.get('name') == '<root>':
            phase = span.get('attrs', {}).get('phase')
            return phase if phase in ('warmup', 'measure') else 'unknown'
        corr = span.get('parent')
    return 'unknown'


def validate_manifest(path):
    path = Path(path).resolve()
    state = json.loads(path.read_text())
    if state.get('schema') != 'flagprof-group/1' or state.get('world_size') != 2:
        raise ValueError('unsupported group manifest')
    rows = state.get('ranks', [])
    if [r['rank'] for r in rows] != [0, 1] or any(type(r['rank']) is not int for r in rows):
        raise ValueError('missing, duplicate or unordered group ranks')
    seen = set()
    for row in rows:
        if row.get('group_id') != state['group_id']:
            raise ValueError('worker group identity mismatch')
        for field in ('run', 'capture'):
            if not row.get(field):
                continue
            entry = Path(row[field])
            resolved = (path.parent / entry).resolve()
            if entry.is_absolute() or '..' in entry.parts or path.parent not in resolved.parents:
                raise ValueError(f'group {field} path must stay relative to its delivery')
            if resolved in seen:
                raise ValueError(f'duplicate group {field} path for rank {row["rank"]}')
            seen.add(resolved)
    return state


def check_ranks(states, group):
    if len(states) != 2:
        raise ValueError('two rank states required')
    from flagprof.components.flaggems import gems_identity
    identities = [dict(state['identity']) for state in states]
    components = [gems_identity(identity.pop('flaggems', None)) for identity in identities]
    (left_switch, left_hash), (right_switch, right_hash) = components
    if (identities[0] != identities[1] or left_switch != right_switch
            or left_hash and right_hash and left_hash != right_hash):
        raise ValueError('rank workload identity mismatch')
    if states[0]['trajectories'] != states[1]['trajectories']:
        raise ValueError('rank token or step control mismatch')
    if states[0].get('tensors') != states[1].get('tensors'):
        raise ValueError('rank replicated boundary mismatch')
    if states[0].get('forward_counts') != states[1].get('forward_counts'):
        raise ValueError('rank model forward count or phase mismatch')
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


def generate_group(manifest, out=None, *, runner_results=None, limits=None):
    from flagprof.offline.worker import output_operation
    group = Path(manifest).resolve().parent
    return output_operation(out or group.with_name(group.name+'-report'), [group], _generate_group,
                            manifest, runner_results=runner_results, limits=limits)


def _generate_group(manifest, out=None, *,  runner_results=None, limits=None):
    limits=limits if isinstance(limits,Limits) else Limits(**(limits or {}))
    manifest = Path(manifest).resolve()
    group = validate_manifest(manifest)
    out = Path(out).resolve() if out else manifest.parent.with_name(manifest.parent.name+'-report')
    if out.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(out):
        raise ValueError('group report must be outside input delivery')
    out.mkdir(parents=True, exist_ok=True)
    states, issues, reports = [], [], []
    rank_identities = {}
    if group.get('status')!='complete':
        issues.append('group execution did not complete')
    for row in group['ranks']:
        rank = row['rank']
        if not row.get('run'):
            issues.append('rank '+str(rank)+' has no Store')
            continue
        run = manifest.parent/row['run']
        if not (run/'manifest.json').exists():
            issues.append('rank '+str(rank)+' Store unavailable')
            continue
        try:
            reports.append(dict(rank=rank,**generate(run, out/('rank-'+str(rank)),limits=limits)))
        except (OSError,ValueError,KeyError) as error:
            issues.append('rank '+str(rank)+' report: '+str(error))
            continue
        with StoreView(run) as view, tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(view, directory, limits)
            try:
                workspace.ingest()
                from flagprof.analyzers.oracle import identity
                observed = identity(workspace)
                if observed and observed.get('rank', rank) != rank:
                    raise ValueError(f'rank {rank} Store identity does not match group manifest')
                if observed:
                    rank_identities[rank] = dict(observed)
                observed.pop('device', None)
                observed.pop('rank', None)
                backend=observed.get('comm_backend');world=observed.get('parallelism',{}).get('world_size')
                if backend is None or world is None:
                    issues.append('rank '+str(rank)+' workload identity unavailable; model comparison unavailable')
                if backend not in (None,group['backend']) or world not in (None,2):
                    raise ValueError('actual TP backend or world size differs from group manifest')
                calls = {}
                sequences = {}
                tensors = {}
                forward_counts = dict(warmup=0, measure=0, unknown=0)
                model_call_phases = dict(warmup=0, measure=0, unknown=0)
                for _, _, _, event in workspace.events(('span',)):
                    if event.get('cat') == 'module' and event.get('ph') == 'x' and event.get('name') == '<root>':
                        phase = event.get('attrs', {}).get('phase', 'unknown')
                        forward_counts[phase if phase in forward_counts else 'unknown'] += 1
                for _, _, _, event in workspace.events(('tensor',)):
                    if not replicated_boundary(event):continue
                    key=event['call_key']
                    if key in tensors:raise ValueError('duplicate rank tensor call_key')
                    tensors[key]={k:event.get(k) for k in ('shape','dtype','digest','mask')}
                if observed.get('generation',{}).get('mode')=='accuracy' and not tensors:
                    raise ValueError('rank accuracy tensor evidence missing')
                for run_id, writer, _, event in workspace.events(('comm',)):
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
                    if event.get('ext',{}).get('communication',{}).get('role')=='model':
                        if event['lib']!=group['backend']:
                            raise ValueError('actual collective backend differs from group manifest')
                        phase = collective_forward_phase(workspace, run_id, writer, event.get('corr'))
                        model_call_phases[phase] += 1
                states.append(dict(rank=rank, identity=observed, trajectories=trajectories(workspace),
                                   calls=calls, tensors=tensors, forward_counts=forward_counts,
                                   model_call_phases=model_call_phases))
            except (OSError,ValueError,KeyError) as error:
                if group.get('status')=='complete':raise
                issues.append('rank '+str(rank)+' incomplete execution evidence: '+str(error))
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
        context['workload_identity'].pop('rank',None)
        contexts[report['rank']]=context
        request_rows.extend(dict(r,rank=report['rank']) for r in panel.get('data',[]) if r.get('kind')=='project_request')
        rows=out/('rank-'+str(report['rank']))/'exports/communication-device-calls.jsonl'
        device_calls[report['rank']]={(r['group'],r['group_seq']):r for r in map(json.loads,rows.read_text().splitlines())}
    comparison=comparable(contexts.get(0,{}),contexts.get(1,{}));aligned=[]
    control_calls = []
    control_keys = set()
    model_keys = set()
    for state in states:
        rank = state['rank']
        for key, event in state['calls'].items():
            role = event.get('ext', {}).get('communication', {}).get('role')
            if role == 'model':
                model_keys.add(key)
            elif role == 'control':
                control_keys.add(key)
                device = device_calls.get(rank, {}).get(key, {})
                control_calls.append(dict(rank=rank, group=key[0], group_seq=key[1],
                    backend=event['lib'], op=event['op'], host_duration_ns=event['dur'],
                    device_union_ns=device.get('device_union_ns')))
    keys=(set(device_calls.get(0,{})) | set(device_calls.get(1,{}))) - control_keys
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
                 if e.get('ext',{}).get('communication',{}).get('role') == 'model']
        allreduce = sum(e['op']=='allreduce' for e in model)
        allgather = sum(e['op']=='allgather' for e in model)
        steps = sum(len(q['steps']) for q in state['trajectories'].values())
        generation=state['identity'].get('task')=='generate'
        # Perf does not capture tensors. Count actual root calls, including warmup,
        # rather than inferring execution from precision output/sample records.
        forwards=sum(state['forward_counts'].values())
        basis_count = steps if generation else forwards
        expected_reduce = (72 if generation else 56)*basis_count if basis_count else None
        expected_gather = (steps if generation else 0) if basis_count else None
        matches=None if expected_reduce is None else allreduce==expected_reduce and allgather==expected_gather
        rank = state['rank']
        controls = sum(e.get('ext', {}).get('communication', {}).get('role') == 'control'
                       for e in state['calls'].values())
        timed = len(state['calls'].keys() & device_calls.get(rank, {}).keys())
        counts.append(dict(rank=state['rank'], steps=steps, forwards=forwards,
                           forward_phases=state['forward_counts'],
                           count_basis='generation_steps' if generation else 'root_module_spans',
                           model_call_phases=state['model_call_phases'],
                           logical_calls=len(state['calls']), model_calls=len(model),
                           control_calls_count=controls, other_calls=len(state['calls'])-len(model)-controls,
                           device_linked_calls=timed,
                           allreduce_per_basis=allreduce/basis_count if basis_count else None,
                           allreduce=allreduce, allgather=allgather,
                           expected_allreduce=expected_reduce, expected_allgather=expected_gather,
                           model_count_matches=matches))
        if matches is None:
            missing = 'generation step' if generation else 'model forward'
            issues.append(f"rank {rank} {missing} evidence missing")
        elif not matches:
            issues.append('rank '+str(state['rank'])+' model collective count mismatch')
    paired_keys = {(r['group'], r['group_seq']) for r in aligned if 'cross_rank_min_observed_ns' in r}
    result = dict(schema='flagprof-group-report/1', group_id=group['group_id'], backend=group['backend'],
                  execution_status=group['status'], status='partial' if issues else 'complete', issues=issues,
                  rank_counts=counts, ranks=reports, rank_execution=group['ranks'],
                  device_link_status='linked' if aligned and paired_keys == keys and model_keys <= paired_keys else 'partial' if aligned else 'explicit_link_unavailable', device_duration_ns=None,
                  device_link_scope='explicitly_linked_collectives; control calls are listed separately',
                  model_call_count=len(model_keys), paired_model_calls=len(model_keys & paired_keys), control_calls=control_calls,
                  unlinked_logical_calls={r['rank']:r['logical_calls']-r['device_linked_calls'] for r in counts},
                  device_calls=aligned,comparison=comparison,clocks={rank:value.get('clock') for rank,value in contexts.items()},
                  request_windows=request_rows,
                  device_timing_note='跨 rank 最小观测占用、最大值与差值；差值不等于额外等待，最小值不代表纯通信下界',
                  timing_note='host enqueue duration is not device communication duration')
    (out/'exports').mkdir(exist_ok=True)
    with (out/'exports/communication-aligned.jsonl').open('w') as stream:
        for row in aligned:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (out/'exports/communication-control-calls.jsonl').open('w') as stream:
        for row in control_calls:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    if runner_results:
        from .performance import summarize
        if set(rank_identities) != {0, 1}:
            raise ValueError('group runner attachment requires both Store workload identities')
        result['runner_summary']=summarize(runner_results,limits=limits,rank_identities=rank_identities)
    from flagprof.monitor.diagnostic_report import panel
    result['diagnosis']=panel(manifest.parent)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    from .product import write_report
    write_report(out,result)
    return dict(output=str(out), status=result['status'], issues=issues)
