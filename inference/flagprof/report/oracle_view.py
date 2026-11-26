# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Reader-facing decisions from paired tensor observations."""
from collections import Counter

from .fields import display


def identity_priority(row):
    if row.get('equal') is False:
        return 0 if row.get('requirement') == 'must_equal' or not row.get('declared') else 1
    if row.get('equal') is None:
        return 4
    return 2 if row.get('requirement') == 'must_equal' else 3


def identity_summary(rows):
    counts = Counter((r.get('requirement'), r.get('equal')) for r in rows)
    parts = []
    for key, label in [('must_equal', '必须一致'), ('declarable', '允许声明差异')]:
        total = sum(v for (req, _), v in counts.items() if req == key)
        if total:
            parts.append(f'{label}的条件中，{counts[key, True]}/{total} 项相同')
    missing = [display(r['field'], 'field') for r in rows if r.get('requirement') == 'not_recorded']
    if missing:
        parts.append('未记录的条件为' + '、'.join(missing))
    return '。'.join(parts) + '。' if parts else ''


def identity_row(row):
    item = dict(row, review_priority=identity_priority(row))
    value = row.get('ref')
    if row.get('equal') is True and isinstance(value, (dict, list)):
        units = {'boundaries':'个采集位置', 'model_files':'份文件摘要', 'batches':'个输入批次',
                 'software':'项软件信息', 'numerics':'项数值设置'}
        if row['field'] in units:
            item['paired_preview'] = f"{len(value)} {units[row['field']]}"
    return item


def route_note(identities):
    gems = next((r for r in identities if r.get('field') == 'flaggems'), {})
    def disabled(side):
        value = gems.get(side)
        enabled = value.get('enabled') if isinstance(value, dict) else value
        return enabled is False or enabled == 'off'

    if disabled('cmp'):
        return '两侧均未启用 FlagGems。' if disabled('ref') else '候选侧未启用 FlagGems。'
    return ''


def observed_equal(row):
    return row.get('numeric_equal', row.get('digest_equal'))


def operator_row(original, note):
    row = dict(original)
    if row.get('introduced'):
        verdict = '输入数值一致，输出出现差异' if 'numeric_equal' in row else '同输入下出现数值差异'
        priority = 0
    elif row.get('input_status') == 'different':
        verdict, priority = '进入算子前输入已不同', 1
    elif row.get('call_config_status') == 'equal' and row.get('input_status') == 'equal' and observed_equal(row) is True:
        verdict, priority = '观测输入与输出均一致', 3
    else:
        verdict, priority = '需核对配置或张量证据', 2
    row['op_verdict'] = verdict
    row['review_priority'] = priority
    route = row.get('expected_route') or {}
    if not note:
        row['candidate_rule'] = route.get('impl') or ('未取得该调用的路由规则' if route.get('status') in
            (None, 'not_registered') else display(route['status']))
    return row


def changes(record, samples):
    # Older producers ranked 0 -> 0 and listed some transitions twice.
    transitions = {}
    for entry in list(record.get('largest_jumps', ())) + list(record.get('zero_to_nonzero', ())):
        ratio = entry.get('ratio')
        if entry.get('change') != '0_to_nonzero' and ratio in (None, 1):
            continue
        transitions[entry['before'], entry['after'], entry.get('sample')] = dict(entry)
    needed = {row[side] for row in transitions.values() for side in ('before', 'after')
              if row.get(side + '_l2') is None}
    values = {}
    if needed:
        for sample in samples:
            if sample.get('call_key') in needed:
                values[sample['call_key']] = sample.get('relative_l2')
    for row in transitions.values():
        for side in ('before', 'after'):
            row.setdefault(side + '_l2', values.get(row[side]))
        ratio = row.get('ratio')
        first = record.get('first_inequal') or {}
        if ratio is None and first.get('call_key') == row['after']:
            row['change_label'] = '首次观测到误差'
        else:
            row['change_label'] = ('由零变为非零' if ratio is None else '误差增大' if ratio > 1 else '误差减小')
        yield row


def first_message(panel, layer_count, *, found=False):
    gaps = panel.get('coverage', {}).get('alignment', {})
    if panel.get('status') == 'ok' and layer_count:
        return f'{layer_count} 个已配对边界中未观测到差异。'
    parts = [f'{label} {gaps[key]} 条' for key, label in [('only_ref', '仅参考侧有记录'),
             ('only_cmp', '仅候选侧有记录'), ('misaligned_shape', '形状无法配对'), ('unscoped', '缺少位置标识'),
             ('misaligned_dtype', '数据类型无法配对'), ('misaligned_semantics', '角色或掩码不同'),
             ('missing_numeric_evidence', '缺少可配对数值'), ('limited_numeric_evidence', '仅有局部数值证据')]
             if gaps.get(key)]
    detail = ('该位置是已配对记录中的首个差异，缺失位置与它的先后关系尚未确定。' if found else
              f'已配对 {layer_count} 个边界，证据不足以确定整个范围的首个差异。')
    return detail + '。'.join(parts)


def overview(model, panels, paired_tensors=None):
    identities = list(model.rows('oracle.identity_diff'))
    if panels.get('oracle.identity_diff', {}).get('status') == 'unavailable':
        blockers = [display(r['field'], 'field') for r in identities if identity_priority(r) == 0]
        detail = '阻断条件为' + ('、'.join(blockers) or '比较身份与缺失记录') + '。实际配对 0 条张量记录。'
        if any(r['field'] == 'inputs_sha256' and r.get('equal') is False for r in identities):
            detail += '请核对两次运行使用的输入文件，使用同一份输入重新采集后比较。'
        if any(r['field'] == 'dtype' and r.get('equal') is False and r.get('requirement') == 'must_equal'
               and not r.get('declared') for r in identities):
            detail += '若目的是比较不同浮点精度，请在 accuracy 工作流的 compare.declare 中显式列出 dtype。'
        return dict(title='比较条件未满足，无法判断精度差异', detail=detail, section='oracle.identity_diff')

    layers = model.rows('oracle.layer')
    final = None
    outputs = 0
    for row in layers:
        if row.get('scope') == '<root>' and row.get('boundary') == 'output':
            outputs += 1
            if final is None or row.get('order', 0) >= final.get('order', 0):
                final = row
    title = '已配对边界的精度对照'
    details = []
    if paired_tensors is not None:
        details.append(f'实际配对 {paired_tensors:,} 条张量记录。')
    if final:
        full = final.get('levels') == ['full']
        same = observed_equal(final)
        scope = '最终输出' if outputs == 1 else '最后一次观测的模型输出'
        if not full:
            scope += '的已观测位置'
        maximum = final.get('max_abs')
        if isinstance(maximum, dict):
            maximum = maximum.get('value')
        if same is True:
            title = scope + '一致'
        elif same is False:
            title = scope + ('存在数值差异' if 'numeric_equal' in final or (maximum or 0) > 0 or final.get('nonfinite_pair_mismatches')
                             else '的张量摘要不同')
        if final.get('samples') is not None:
            details.append(f"比较了 {final['samples']} 个样本。")
        if final.get('valid_elements') is not None and final.get('differing_elements') is not None:
            details.append(f"有效元素 {final['valid_elements']:,} 个，其中 {final['differing_elements']:,} 个不同。")
        if maximum is not None:
            details.append('最大绝对误差为 ' + display(maximum) + '。')
    if len(layers):
        equal = sum(observed_equal(r) is True for r in layers)
        if any('numeric_equal' in r for r in layers):
            different = sum(observed_equal(r) is False for r in layers)
            details.append(f'已配对 {len(layers)} 个边界，{equal} 个数值一致，{different} 个观测到差异，'
                           f'{len(layers)-equal-different} 个证据不足以判定。')
        elif equal == len(layers):
            details.append(f'{equal} 个已配对边界的张量摘要全部一致。')
        else:
            details.append(f'已检查 {len(layers)} 个配对边界，其中 {equal} 个张量摘要一致。')
    else:
        details.append('没有可配对的逐层输出。')
    if panels.get('oracle.layer', {}).get('status') == 'partial':
        details.append('部分观测缺失，结论仅覆盖已配对的位置。')

    calls = Counter()
    scopes = set()
    for row in model.rows('oracle.op_sensitivity'):
        calls[row['op_verdict']] += 1
        if row.get('scope') and len(scopes) < 2:
            scopes.add(row['scope'])
    if calls:
        scope = next(iter(scopes)) if len(scopes) == 1 else ''
        if scope.startswith('<root>.layers.') and scope.removeprefix('<root>.layers.').isdigit():
            scope = '第 '+scope.removeprefix('<root>.layers.')+' 层'
        prefix = '在 '+scope+' ' if scope and scope!='<root>' else ''
        counts = [f'{n} 次{verdict}' for verdict,n in calls.items()]
        if len(calls)==1 and '观测输入与输出均一致' in calls:
            details.append(f"{prefix}检查的 {sum(calls.values())} 次算子调用，两侧输入与输出均一致。")
        else:
            details.append(f"{prefix}检查了 {sum(calls.values())} 次算子调用，其中"+'，'.join(counts)+'。')
    elif 'oracle.op_sensitivity' in panels:
        details.append('本次没有可比较的算子调用。')
    replays = list(model.rows('oracle.replay'))
    section = 'oracle.layer' if len(layers) else 'oracle.first_divergence'
    if replays:
        valid = [r for r in replays if r.get('valid')]
        same = sum(r.get('metrics', {}).get('digest_equal') is True for r in valid)
        if same == len(replays):
            title += '，同输入回放的两侧输出一致'
        elif any(r.get('metrics', {}).get('digest_equal') is False for r in valid):
            title += '，同输入回放也观测到输出差异'
        details.append(f'固定输入交叉回放 {len(replays)} 组，{len(valid)} 组有效，其中 {same} 组两侧输出一致。')
        section = 'oracle.replay'
    return dict(title=title, detail=''.join(details), section=section)
