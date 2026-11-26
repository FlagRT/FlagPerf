# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Read the native report's small identity and status fields without materializing its event tables."""
from collections import Counter
import re
from flagprof.offline.streaming import JSONStream

# Find delimiters outside and inside strings while skipping unselected containers.
STRUCTURE = re.compile(r'[\[\]{}"]')
QUOTED = re.compile(r'["\\]')


def skip(stream):
    stream.space()
    if stream.peek() not in ('{','['):
        stream.value()
        return
    stack=[stream.take(1)]
    quoted=escape=False
    while stack:
        if not stream.peek():
            raise ValueError('incomplete report JSON')
        if escape:
            stream.take(1)
            escape=False
            continue
        match=(QUOTED if quoted else STRUCTURE).search(stream.text,stream.pos)
        if not match:
            stream.take(len(stream.text)-stream.pos)
            continue
        stream.take(match.start()-stream.pos)
        token=stream.take(1)
        if token=='"':
            quoted=not quoted
        elif quoted:
            escape=True
        elif token in ('{','['):
            stack.append(token)
        elif stack.pop()!=('{' if token=='}' else '['):
            raise ValueError('unbalanced report JSON')


def fields(stream):
    stream.expect('{')
    stream.space()
    if stream.peek()!='}':
        while True:
            key=stream.value()[0]
            stream.expect(':')
            stream.space()
            yield key
            stream.space()
            if stream.peek()=='}':
                break
            stream.expect(',')
    stream.expect('}')


def array(stream):
    stream.expect('[')
    stream.space()
    if stream.peek()!=']':
        while True:
            yield None
            stream.space()
            if stream.peek()==']':
                break
            stream.expect(',')
    stream.expect(']')


def selected(stream, keys):
    result={}
    for key in fields(stream):
        if key in keys:
            result[key]=stream.value()[0]
        else:
            skip(stream)
    return result


def report_info(path, *, record_bytes=16<<20, details=False):
    stream=JSONStream(path,limit=record_bytes)
    result=dict(results={},runner_summary={},comparisons=[],metric_status_counts=Counter(),missing_deltas=Counter())
    try:
        for field in fields(stream):
            if field in ('schema','status','issues','pair_statuses','comparison_status','blocking_fields','paired_tensors','paired_boundaries'):
                result[field]=stream.value()[0]
            elif field=='results':
                for panel in fields(stream):
                    if details:
                        result['results'][panel]=selected(stream,('status','notes'))
                    else:
                        result['results'][panel]={}
                        skip(stream)
            elif field=='runner_summary' and stream.peek()!='n':
                for key in fields(stream):
                    if key not in ('groups','sources'):
                        skip(stream)
                        continue
                    keys=('rank','scope','phase','status','shape','observations','mean_ns','median_ns','p95_ns') if details else ('rank','observations')
                    if key=='sources':
                        keys=('rank','identity','store_identity','flaggems_source_sha256')
                    result['runner_summary'][key]=[selected(stream,keys) for _ in array(stream)]
            elif field=='comparisons' and details:
                for _ in array(stream):
                    pair=selected(stream,('left','right','status','reasons','differences'))
                    if 'left' in pair and 'right' in pair:
                        result['comparisons'].append(pair)
            elif field=='metric_comparisons' and details:
                for _ in array(stream):
                    row=selected(stream,('comparison_status','delta','relative_change'))
                    result['metric_status_counts'][row['comparison_status']]+=1
                    if row.get('delta') is None and row.get('relative_change') is None:
                        result['missing_deltas'][row['comparison_status']]+=1
            else:
                skip(stream)
        stream.space()
        if stream.peek():
            raise ValueError('trailing report input')
        return result
    finally:
        stream.close()
