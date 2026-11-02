# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CANN native ownership, including the explicit Enqueue/Dequeue bridge."""
from __future__ import annotations
from .format import NATIVE_CALL_CATEGORIES, ANNOTATION_CATEGORIES
from collections import defaultdict
from flagprof.correlate.linker import Hit, NormalizedTrace, identity
from flagprof.correlate.native import ScopeIndex
from flagprof.vendors.native_common import ScopeConfig, classify, nesting_rule, launch_interval
from flagprof.vendors.rules_common import PreparedRule
from .rules import correlation_keys

VERSION = 'cann-native-t8.3-queue-v1'

def build_native_rules(trace: NormalizedTrace, config: ScopeConfig):
    calls, annotations, modules = classify(trace, config.resolved(ANNOTATION_CATEGORIES), NATIVE_CALL_CATEGORIES)
    call_index = ScopeIndex(calls)
    dequeues = ScopeIndex([r for r in trace.records if r['name'].startswith('Dequeue') and r['ext']['chrome.ph']=='X'])
    enqueues = defaultdict(list)
    for r in trace.records:
        if r['name'].startswith('Enqueue'):
            for key in correlation_keys(r):
                enqueues[(r['trace_uid'],r['host_pid'],key)].append(r)
    hits = {}
    for launch in trace.launches:
        found = []
        for dequeue in dequeues.enclosing(launch_interval(launch)):
            for key in correlation_keys(dequeue):
                for enqueue in enqueues.get((dequeue['trace_uid'],dequeue['host_pid'],key),[]):
                    found.extend(Hit(owner,(dequeue,enqueue)) for owner in call_index.enclosing(enqueue))
        hits[identity(launch)] = found
    return ([nesting_rule(trace,calls,'cann.native_nesting'),
             PreparedRule('cann.native_enqueue','host',('trace_uid','host_pid','correlation'),hits)],
            [],calls,annotations,modules)
