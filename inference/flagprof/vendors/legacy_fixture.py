# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compatibility for historical communication fixtures without project corr."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from flagprof.correlate.linker import NormalizedTrace

def preserve_unbound_communication(trace: NormalizedTrace, calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain the original narrow fixture exception; malformed explicit corr fails."""
    unbound = {c['marker']: c for c in calls if 'corr' not in c and
               {'marker', 'group', 'group_seq', 'op', 'rank'} <= c.keys()}
    result = []
    for record in trace.records:
        if record['t'] == 'op' and record['name'] in unbound:
            record['t'] = 'host'
            record.pop('corr', None)
            result.append(dict(trace_uid=trace.trace_uid, loc=record['loc'],
                               reason='no_project_corr_in_source', call=unbound[record['name']]))
    return result
