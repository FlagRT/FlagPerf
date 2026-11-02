# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Generation trajectories and inclusive first-divergence plans from Store events."""
from . import Result


def trajectories(work):
    requests = {}
    for _, _, _, event in work.events(('span',)):
        if event.get('name') != 'generation_step_complete':
            continue
        attrs = event['attrs']
        key = attrs['request_id']
        row = requests.setdefault(key, dict(request_id=key, samples=attrs['samples'], mode=attrs['mode'], steps=[]))
        if row['samples'] != attrs['samples'] or row['mode'] != attrs['mode']:
            raise ValueError('generation request identity changed')
        if attrs['generation_step'] != len(row['steps']):
            raise ValueError('generation steps missing, duplicated or out of order')
        row['steps'].append(attrs)
    return requests


def prefix_plan(left, right):
    """The differing selection was made from common input, so include that step."""
    common = min(len(left), len(right))
    first = next((i for i in range(common) if left[i] != right[i]), None)
    return dict(first_token_divergence=first, compare_steps=first+1 if first is not None else common,
                lengths_equal=len(left)==len(right), ref_steps=len(left), cmp_steps=len(right),
                rule='inclusive_first_token_divergence')


def comparison(ref, cmp):
    def samples(work):
        result = {}
        for request in trajectories(work).values():
            if request['mode'] != 'accuracy':
                raise ValueError('numerical generation pairing requires accuracy mode')
            if not request['steps'] or not request['steps'][-1]['done']:
                raise ValueError('incomplete generation trajectory')
            for i, sample in enumerate(request['samples']):
                if sample in result:
                    raise ValueError('generation pairing requires one request per sample')
                result[sample] = dict(tokens=[step['tokens'][i] for step in request['steps']],
                                      stop_reason=request['steps'][-1]['stop_reasons'][i])
        return result
    left, right = samples(ref), samples(cmp)
    if set(left) != set(right) or not left:
        raise ValueError('generation sample trajectories missing')
    limits = {}
    for sample in left:
        plan = dict(sample=sample, **prefix_plan(left[sample]['tokens'], right[sample]['tokens']),
                    ref=left[sample], cmp=right[sample])
        ref.row('oracle.generation', sample, plan)
        limits[sample] = plan['compare_steps']
    return limits


def analyze(work):
    requests = trajectories(work)
    if not requests:
        return []
    for key, row in requests.items():
        work.row('generation', key, dict(request_id=key, samples=row['samples'], mode=row['mode'],
                 steps=len(row['steps']), complete=bool(row['steps'] and row['steps'][-1]['done']),
                 tokens=[s['tokens'] for s in row['steps']],
                 stop_reasons=row['steps'][-1]['stop_reasons'] if row['steps'] else []))
    for _, _, seq, event in work.events(('span',)):
        if event.get('cat') == 'step' and event.get('attrs',{}).get('kind') == 'generation_step':
            work.row('generation.steps', str(seq).zfill(20), dict(**event['attrs'], host_duration_ns=event['dur'],
                     duration_scope='instrumented complete advance; backbone/head/greedy/next-input'))
        elif event.get('name') in ('kv_created','kv_observation','kv_released'):
            work.row('generation.kv', str(seq).zfill(20), dict(event['attrs'], event=event['name']))
    return [Result('generation', 'ok' if all(r['steps'][-1]['done'] for r in requests.values()) else 'partial',
                   work.table('generation'), dict(steps=work.table('generation.steps'),kv=work.table('generation.kv')),
                   ['采集请求的完整生成步；无诊断性能基线单独计时。'])]


def communication(work):
    from .workspace import encode
    import json
    counts = {}
    context=work.view.device_analysis_context()
    for _, _, seq, event in work.events(('comm',)):
        ext = event.get('ext', {}).get('communication', {})
        if not ext:
            continue
        key = (event['group'], event['op'], event['lib'])
        row = counts.setdefault(key, dict(group=key[0], op=key[1], backend=key[2], role=ext['role'],
                                         calls=0, generation_calls=0, logical_input_bytes=0, logical_output_bytes=0,
                                         unknown_payload_calls=0,device_linked_calls=0,device_sum_ns=0,device_representation_records=0))
        row['calls'] += 1
        row['generation_calls'] += bool(ext.get('generation'))
        row['logical_input_bytes'] += ext['logical_input_bytes']
        row['logical_output_bytes'] += ext['logical_output_bytes']
        row['unknown_payload_calls'] += ext.get('payload_status')!='tensor'
        devicekey=encode((event['group'],event['group_seq'],event.get('rank')))
        found=work.db.execute('select payload from rows where panel=? and key=?',('communication.device_calls',devicekey)).fetchone()
        device=json.loads(found[0]) if found else None
        if device:
            row['device_linked_calls']+=1;row['device_sum_ns']+=device['device_sum_ns'];row['device_representation_records']+=device['device_records']
        output=dict(event,device_analysis=device or dict(device_link_status='explicit_link_unavailable',device_union_ns=None,device_sum_ns=None,reason='no_explicit_device_match_in_capture'))
        work.row('communication.calls', str(seq).zfill(20), output)
    if not counts:
        return []
    for key, row in counts.items():
        row['device_unlinked_calls']=row['calls']-row['device_linked_calls']
        row['device_link_ratio']=row['device_linked_calls']/row['calls'] if row['calls'] else None
        if not row['device_linked_calls']:row['device_sum_ns']=None
        if row['unknown_payload_calls']:
            row.update(logical_input_bytes=None,logical_output_bytes=None,payload_status='object_or_control_unavailable')
        else:
            row['payload_status']='tensor_logical_bytes'
        work.row('communication', '/'.join(key), dict(row, runtime_padding_bytes=None, device_wire_bytes=None,
                                                    device_link_status='linked' if row['device_linked_calls']==row['calls'] else 'partial' if row['device_linked_calls'] else 'explicit_link_unavailable',clock=context.get('clock')))
    return [Result('communication', 'ok' if all(r['device_linked_calls']==r['calls'] for r in counts.values()) else 'partial', work.table('communication'), dict(clock=context.get('clock'),device_calls=work.table('communication.device_calls')),
                   ['模型与控制通信按组分列；主机提交时长、逻辑调用数和设备表示记录数分别统计。设备占用仅来自明确关联。'])]
