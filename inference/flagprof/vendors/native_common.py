"""Chrome host classification shared by vendor adapters (not the core)."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Sequence
from flagprof.correlate.linker import Hit, NormalizedTrace, Record, identity
from flagprof.correlate.native import ScopeIndex
from .rules_common import PreparedRule

@dataclass(frozen=True)
class ScopeConfig:
    module_prefixes: tuple[str, ...] = ()
    module_categories: tuple[str, ...] = ()
    annotation_prefixes: tuple[str, ...] = ('ProfilerStep#',)
    annotation_categories: tuple[str, ...] | None = None

    def resolved(self, categories: tuple[str, ...]) -> ScopeConfig:
        return replace(self,annotation_categories=categories) if self.annotation_categories is None else self

def classify(trace: NormalizedTrace, config: ScopeConfig, native_call_categories: tuple[str, ...]) -> tuple[list[Record], list[Record], set]:
    calls, annotations, modules = [], [], set()
    for r in trace.records:
        ext = r['ext']
        if ext['chrome.ph'] != 'X':
            continue
        is_module = ext['chrome.cat'] in config.module_categories or r['name'].startswith(config.module_prefixes)
        is_annotation = is_module or ext['chrome.cat'] in config.annotation_categories or r['name'].startswith(config.annotation_prefixes)
        if is_annotation:
            annotations.append(r)
            if is_module:
                modules.add(identity(r))
        elif ext['chrome.cat'] in native_call_categories:
            calls.append(r)
    return calls, annotations, modules

def launch_interval(record: Record) -> Record:
    """Chrome flow starts are points; unknown call duration stays unknown."""
    return dict(record,dur=0) if record['ext']['chrome.ph']=='s' and record['dur'] is None else record

def nesting_rule(trace: NormalizedTrace, calls: Sequence[Record], via: str) -> PreparedRule:
    # Chrome flow starts are point events (no dur), not missing-duration
    # calls. Use the vendor source-clock interval convention for launch
    # containment; strict call-to-call ordering remains in the public core.
    index = ScopeIndex(calls)
    return PreparedRule(via, 'host', ('trace_uid','host_pid','tid'),
                        {identity(s): [Hit(c) for c in index.enclosing(launch_interval(s))] for s in trace.launches})
