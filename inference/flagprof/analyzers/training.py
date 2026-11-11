# Copyright 2026 Suscom Lab, The Hong Kong University of Science and Technology (Guangzhou)
# Licensed under the Apache License, Version 2.0.
"""Training intervals read from standard spans, without importing torch."""
from itertools import groupby
import json
from . import Result


def analyze(work):
    work.db.execute('create table training_intervals(owner text, step integer, phase text, micro integer, payload text)')
    count = 0
    for run, writer, sequence, event in work.events(('span',)):
        attrs = event.get('attrs', {})
        if attrs.get('training_schema') != 'training/1':
            continue
        index, phase = attrs.get('step'), attrs.get('phase')
        if type(index) is not int or index < 0 or phase not in ('step', 'zero_grad', 'forward', 'backward', 'optimizer'):
            raise ValueError('invalid training interval identity')
        row = dict(attrs, run_id=run, writer=writer, sequence=sequence,
                   rank=event.get('rank'), pid=event.get('pid'), tid=event.get('tid'),
                   host_duration_ns=event.get('dur'), clock=event.get('clk'),
                   duration_scope='instrumented host interval; asynchronous device completion not implied')
        key = json.dumps([run, writer, event.get('rank'), event.get('pid'), event.get('tid')])
        work.db.execute('insert into training_intervals values(?,?,?,?,?)',
                        (key, index, phase, attrs.get('microstep'), json.dumps(row)))
        work.row('training.phases', json.dumps([run, writer, sequence]), row)
        count += 1
        if count % 1000 == 0:
            work.check_budget()
    if not count:
        return []
    partial = False
    cursor = work.db.execute('select owner,step,phase,micro,payload from training_intervals order by owner,step,phase,micro')
    for (owner, index), intervals in groupby(cursor, key=lambda r: (r[0], r[1])):
        counts = {}; micros = {}; durations = {}; issues = []; outer = None
        for _, _, phase, micro, payload in intervals:
            row = json.loads(payload)
            counts[phase] = counts.get(phase, 0) + 1
            if row.get('outcome', {}).get('status') != 'complete': issues.append(phase + ':not_complete')
            duration = row['host_duration_ns']
            if type(duration) is not int or duration < 0: issues.append(phase + ':missing_duration')
            else: durations[phase] = durations.get(phase, 0) + duration
            if phase == 'step': outer = row
            if phase in ('forward', 'backward'):
                values = micros.setdefault(phase, [])
                if type(micro) is not int or micro < 0 or micro in values: issues.append(phase + ':invalid_microstep')
                values.append(micro)
        for phase in ('step', 'zero_grad', 'optimizer'):
            if counts.get(phase) != 1: issues.append(phase + ':missing_or_duplicate')
        expected = (outer or {}).get('accumulation')
        if type(expected) is not int or expected < 1:
            issues.append('accumulation_not_declared')
        else:
            for phase in ('forward', 'backward'):
                values = micros.get(phase, [])
                if len(values) != expected or any(v != i for i, v in enumerate(sorted(v for v in values if type(v) is int))):
                    issues.append(phase + ':incomplete_microsteps')
        issues = sorted(set(issues)); partial |= bool(issues)
        work.row('training', json.dumps([owner, index]), dict(step=index, owner=json.loads(owner),
                 status='partial' if issues else 'ok', issues=issues, phase_counts=counts,
                 host_duration_ns=(outer or {}).get('host_duration_ns'), phase_host_ns=durations))
    return [Result('training', 'partial' if partial else 'ok', work.table('training'),
                   dict(phases=work.table('training.phases'), intervals=count),
                   ['训练阶段主机区间；不能相加为设备耗时，也不代表训练精度或收敛通过。',
                    '既有模块与路由窗口以 forward 为边界；不将其当作完整 backward 或 DDP 通信覆盖。'])]
