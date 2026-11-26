# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Present diagnostic findings in the context of the monitored runs."""
from .fields import display


def timeline_axis(tasks, findings, session_ns):
    """Choose one origin for the complete report, before any interactive filtering."""
    tasks = list(tasks)
    if not tasks or any(not task.get('task_label') for task in tasks):
        return dict(origin='unmapped', duration_ns=None)
    if len(tasks) > 1 and session_ns is not None:
        return dict(origin='node', duration_ns=session_ns)
    if len(tasks) == 1:
        origin = 'task'
        starts = {tasks[0]['task_label']: 0}
    else:
        # A detached multi-run result may retain a common host clock but no node session.
        clocks = {t.get('start_clock') for t in tasks}
        if len(clocks) != 1 or not all(t.get('start_clock') and t.get('started_ns') is not None for t in tasks):
            return dict(origin='unmapped', duration_ns=None)
        origin = 'first_task'
        first = min(t['started_ns'] for t in tasks)
        starts = {t['task_label']: t['started_ns'] - first for t in tasks}
    end = max((starts[t['task_label']] for t in tasks if t.get('started_ns') is not None), default=None)
    for records, field in ((tasks, 'receipt_relative_ns'), (findings, 'diagnostic_time_ns')):
        for row in records:
            when = row.get(field)
            if when is not None and when >= 0 and row.get('task_label') in starts:
                end = max(end or 0, starts[row['task_label']] + when)
    return dict(origin=origin, duration_ns=end, starts=starts)


def node_axis(document, evidence, tasks, findings):
    """Project raw timestamps onto the full session, including on a --last report."""
    reference = document.get('coverage', {}).get('reference', {})
    window = reference.get('window')
    if not window or len(window) != 2 or window[0] is None:
        return None
    start, end = window
    if end is None and document.get('window'):
        end = document['window'][1]
    if end is None or end <= start:
        return None
    clock = document.get('session_clock')
    if not clock:
        # Older exports identify the node clock on monitor-authored records.
        clocks = {r['clk'] for r in evidence if r.get('clk') and
                  (r.get('detail', {}).get('scope') == 'business_input' or
                   r.get('kind') == 'process_exit' and r.get('detail', {}).get('source') == 'subprocess.wait')}
        clocks.update(t['exit_clock'] for t in tasks if t.get('exit_clock') and t.get('returncode') is not None)
        if len(clocks) != 1:
            return None
        clock = clocks.pop()
    for task in tasks:
        for label, timestamp, source in (('node_start_ns','started_ns','start_clock'),
                                         ('node_exit_ns','exit_ns','exit_clock')):
            if task.get(source) == clock and task.get(timestamp) is not None:
                task[label] = task[timestamp] - start
    by_run = {t['run_id']: t for t in tasks if t['run_id']}
    for row in findings:
        # A rule update is dated by the monitor. OOM and exit use the
        # originating event/receipt, which can precede the monitor decision.
        if row.get('revision', 1) > 1:
            when = row.get('observed_at')
        elif row.get('rule') in ('hang.no_progress','hang.heartbeat','comm.timeout'):
            when = row.get('first_detected')
        elif row.get('event_time_ns') is not None:
            when = row.get('event_time_ns') if row.get('event_clock') == clock else None
        else:
            # A legacy alert can retain only the offset from its own probe.
            task = by_run.get(row.get('run_id'), {})
            if row.get('rule') == 'proc.exit' and 'node_exit_ns' in task:
                row['node_event_ns'] = task['node_exit_ns']
            elif 'node_start_ns' in task and row.get('task_relative_ns') is not None:
                row['node_event_ns'] = task['node_start_ns'] + row['task_relative_ns']
            continue
        if when is not None:
            row['node_event_ns'] = when - start
    for row in findings:
        task_start = by_run.get(row.get('run_id'), {}).get('node_start_ns')
        if row.get('diagnostic_time_ns') is None and task_start is not None and row.get('node_event_ns') is not None:
            row['diagnostic_time_ns'] = row['node_event_ns'] - task_start
    return end - start


def event_reading(row):
    rule = row.get('rule')
    label = '异常退出' if rule == 'proc.exit' else display(rule)
    when = row.get('occurrence_task_relative_ns', row.get('task_relative_ns'))
    if rule == 'proc.exit':
        code = row.get('event_detail', {}).get('returncode')
        if code is not None:
            label += f'（退出码 {code}）'
    elif row.get('revision', 1) > 1:
        state = {'resolved': '已解除', 'unknown': '状态待确认', 'open': '持续'}.get(row.get('state'), '更新')
        label += ' · ' + state
        origin = row.get('task_relative_ns')
        elapsed = row.get('observed_since_onset_ns')
        when = origin + elapsed if origin is not None and elapsed is not None else None
    elif rule in ('hang.no_progress', 'hang.heartbeat', 'comm.timeout'):
        when = row.get('first_detected_relative_ns')
    return label, when


def by_task(observations, findings):
    tasks = {}
    for row in [*observations, *findings]:
        key = row.get('run_id')
        task = tasks.setdefault(key, dict(run_id=key))
        if row.get('rank') is not None:
            task.setdefault('rank', row['rank'])
    for row in observations:
        tasks[row.get('run_id')].update(row)
    known = [task for key, task in tasks.items() if key]
    # Legacy coverage is yielded in first-observed order. Use actual starts
    # for numbering only when they share a recorded clock.
    clock = known[0].get('start_clock') if known else None
    ordered = clock and all(t.get('start_clock') == clock and t.get('started_ns') is not None for t in known)
    if ordered:
        known.sort(key=lambda t:t['started_ns'])
    for index, task in enumerate(known, 1):
        task['task_label'] = '本次任务' if len(known)==1 else f'任务 {index}'
    if None in tasks:
        tasks[None]['task_label'] = '归属未记录'
    lanes = known + ([tasks[None]] if None in tasks else [])
    alerts = {key: set() for key in tasks}
    for index, row in enumerate(findings):
        key = row.get('run_id')
        task = tasks[key]
        row['task_label'] = task['task_label']
        row['event_label'], row['diagnostic_time_ns'] = event_reading(row)
        alerts[key].add((row.get('rank'), row.get('id') or ('record', index)))
        task.setdefault('pid', (row.get('proc_key') or {}).get('pid'))
    for task in lanes:
        task['alert_count'] = len(alerts[task['run_id']])
        code = task.get('returncode')
        if code is not None:
            outcome = '正常退出' if code == 0 else '异常退出'
            task['exit_summary'] = f'{outcome}（退出码 {code}）'
        if (task.get('start_clock') and task['start_clock'] == task.get('exit_clock')
                and task.get('exit_ns') is not None and task.get('started_ns') is not None):
            task['receipt_relative_ns'] = task['exit_ns'] - task['started_ns']
    timing = ''
    if len(known) > 1 and ordered and all(t.get('receipt_relative_ns', -1) >= 0 for t in known):
        disjoint = all(a['exit_ns'] <= b['started_ns'] for a, b in zip(known, known[1:]))
        timing = '按启动顺序编号，任务观测区间没有重叠。' if disjoint else '按启动顺序编号，部分任务的观测区间重叠。'
    elif len(known) > 1:
        timing = '任务按启动顺序编号。' if ordered else '每个编号对应一个运行标识。'
    # Sort within each task only. Keep equal or unmapped instants in source order.
    positions = {t['run_id']:i for i, t in enumerate(lanes)}
    findings.sort(key=lambda r:(positions[r.get('run_id')], r['diagnostic_time_ns'] is None,
                                r['diagnostic_time_ns'] or 0))
    return lanes, timing


def conclusion(model, tasks, findings, timing):
    known = [t for t in tasks if t['run_id']]
    affected = [t for t in tasks if t['alert_count']]
    total = sum(t['alert_count'] for t in affected)
    resolved = total == 1 and max(findings, key=lambda row:row.get('revision', 0)).get('state') == 'resolved'
    if len(affected) == 1:
        owner = affected[0]
        kinds = list(dict.fromkeys('异常退出' if r.get('rule')=='proc.exit' else display(r.get('rule')) for r in findings))
        subject = '、'.join(kinds) if len(kinds) <= 2 else f'{total} 条告警'
        if not owner['run_id']:
            title = f'{total} 条告警缺少任务标识'
        elif len(known) == 1:
            title = subject + ('告警已解除' if resolved else '')
        else:
            title = subject + ('都属于' if total > 1 else '属于') + owner['task_label']
    elif affected:
        total_known = sum(bool(t['run_id']) for t in affected)
        title = f'{total_known} 个任务收到告警' if total_known else '告警归属待核对'
        title += f'，共 {total} 条'
    else:
        title = '未记录任务告警'
    parts = [f'共观测 {len(known)} 个任务。'] if len(known)>1 else []
    if timing:
        parts.append(timing)
    if len(known) <= 3:
        for t in known:
            facts = [t['exit_summary']] if 'exit_summary' in t else []
            if t['alert_count'] and len(affected) > 1:
                facts.append(f"收到 {t['alert_count']} 条告警")
            elif not t['alert_count'] and len(known)>1:
                facts.append('未记录告警')
            elif not facts and t.get('registered_execution'):
                facts.append(f"有 {t['registered_execution']} 次有效执行观测")
            if facts:
                prefix = t['task_label']+' ' if len(known)>1 else '进程'
                parts.append(prefix+'，'.join(facts)+'。')
    if len(affected) > 1 and any(not t['run_id'] for t in affected):
        parts.append('另有告警缺少任务标识，见事件明细。')
    if len(findings) > total:
        parts.append(f'{total} 条告警共有 {len(findings)} 条状态记录。')
    model.finding(title, ''.join(parts), 'diagnosis' if findings else 'diagnosis.coverage',
                  'fault' if findings else 'result')
