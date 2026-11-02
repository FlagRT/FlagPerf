# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Source ledger: enumerate before normalization; reconcile after preparation."""
from __future__ import annotations
from collections import Counter
from typing import Any, Callable, Mapping, Sequence
from flagprof.correlate.linker import identity, ref_sort

class UnrepresentableError(ValueError):
    def __init__(self, fields: Sequence[str]) -> None:
        self.fields = tuple(sorted(fields))
        super().__init__('missing source identity: ' + ', '.join(self.fields))

def enumerate_source(events: Sequence[Mapping[str, Any]], trace_uid: str,
                     classify: Callable[[Mapping[str, Any]], tuple[str | None, str]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for i, e in enumerate(events):
        loc = e.get('_linker_loc',i)
        key = (trace_uid,str(loc))
        if key in seen:
            raise ValueError('duplicate source loc: ' + str(loc))
        seen.add(key)
        category, reason = classify(e)
        rows.append(dict(trace_uid=trace_uid,loc=loc,category=category,
                         disposition='unknown' if category is not None else 'source_only',reason=reason,outputs=[]))
    return rows

def reconcile(rows: Sequence[Mapping[str, Any]], devices: Sequence[Mapping[str, Any]],
              missing: Mapping[tuple[str,str], Sequence[str]], inventory: Sequence[Mapping[str, Any]]) -> tuple[list[dict],dict]:
    outputs: dict[tuple,list] = {}
    for ordinal, e in enumerate(devices):
        outputs.setdefault(identity(e),[]).append(ordinal)
    result = []
    for item in rows:
        row = dict(item)
        key = identity(row)
        if key in missing:
            row.update(disposition='unrepresentable',reason='missing_identity',missing_fields=list(missing[key]))
        elif key in outputs:
            row.update(disposition='imported',reason=None,outputs=outputs[key])
        result.append(row)
    result.sort(key=ref_sort)
    categories = {}
    for cat in ('compute','comm','memcpy','unknown'):
        old = {identity(r) for r in inventory if r['category']==cat}
        current = {identity(r) for r in result if r['category']==cat}
        if (cat!='unknown' and old!=current) or (cat=='unknown' and not old<=current):
            raise ValueError('source inventory conservation failed: ' + cat)
        counts = Counter(r['disposition'] for r in result if r['category']==cat)
        if counts['source_only'] or len(current)!=sum(counts.values()):
            raise ValueError('device work cannot be source_only')
        emitted = sum(1 for e in outputs if e in current)
        if emitted != counts['imported'] or len(current)-emitted != counts['unrepresentable']+counts['unknown']:
            raise ValueError('disposition conservation failed: '+cat)
        categories[cat] = dict(R=len(current),legacy_R=len(old),E=emitted,dispositions=dict(counts),
                              added=[dict(trace_uid=u,loc=next(r['loc'] for r in result if identity(r)==(u,l))) for u,l in sorted(current-old)])
    summary = dict(records=len(result),dispositions=dict(Counter(r['disposition'] for r in result)),categories=categories,
                   one_to_many=[dict(trace_uid=r['trace_uid'],loc=r['loc'],outputs=r['outputs']) for r in result if len(r['outputs'])>1])
    assert sum(summary['dispositions'].values())==len(rows)
    return result,summary
