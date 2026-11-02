"""CUDA-compatible native host rules; hardware identity is supplied separately."""
from __future__ import annotations
from .format import NATIVE_CALL_CATEGORIES, ANNOTATION_CATEGORIES
from collections import defaultdict
from flagprof.correlate.linker import Hit, NormalizedTrace, identity
from flagprof.correlate.native import ScopeIndex
from flagprof.vendors.native_common import ScopeConfig, classify, nesting_rule
from flagprof.vendors.rules_common import PreparedRule

VERSION = 'cuda-native-t8-v1'

def build_native_rules(trace: NormalizedTrace, config: ScopeConfig):
    calls, annotations, modules = classify(trace, config.resolved(ANNOTATION_CATEGORIES), NATIVE_CALL_CATEGORIES)
    external = defaultdict(list)
    for r in calls:
        key = r['ext']['chrome.args'].get('External id')
        if key is not None:
            external[(r['trace_uid'], r['host_pid'], str(key))].append(r)
    # Flow start records may omit External id; nesting remains independently
    # applicable, and all successful paths are unioned by the core.
    host = {}
    for r in trace.launches:
        key = r['ext']['chrome.args'].get('External id')
        host[identity(r)] = [Hit(n) for n in external.get((r['trace_uid'],r['host_pid'],str(key)), [])] if key is not None else []
    pids = {r['host_pid'] for r in calls + trace.launches}
    direct = {}
    for e in trace.devices:
        key = e['ext']['chrome.args'].get('External id')
        direct[identity(e)] = ([Hit(n) for n in external.get((e['trace_uid'],next(iter(pids)),str(key)), [])]
                               if key is not None and len(pids)==1 else [])
    return ([nesting_rule(trace,calls,'cuda.native_nesting'),
             PreparedRule('cuda.native_external_id','host',('trace_uid','host_pid','External id'),host)],
            [PreparedRule('cuda.native_direct','direct',('trace_uid','host_pid','External id'),direct)],
            calls,annotations,modules)
