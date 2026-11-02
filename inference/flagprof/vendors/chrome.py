# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Chrome fragments to normalized records; all source time stays in ext."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from flagprof.correlate.linker import NormalizedTrace, ref_sort
from .format import SourceFormat



def stamp_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = deepcopy(list(events))
    for i, event in enumerate(result):
        event.setdefault("_linker_loc", i)
    return result


def nanoseconds(value: Any) -> Optional[int]:
    return int(Decimal(str(value)) * 1000) if value is not None else None


def adapt(events: Sequence[Dict[str, Any]], calls: Sequence[Dict[str, Any]], *, trace_uid: str,
          taxonomy: SourceFormat, marker_prefix: Optional[str] = None, kind: str = "all",
          inventory: Optional[List[Dict[str, Any]]] = None) -> NormalizedTrace:
    if marker_prefix is not None and (not isinstance(marker_prefix, str) or not marker_prefix):
        raise ValueError('marker_prefix must be None or a nonempty string')
    context = taxonomy.context(events)
    return _adapt(events, _call_map(calls), trace_uid=trace_uid, taxonomy=taxonomy,
                  context=context, marker_prefix=marker_prefix, kind=kind, inventory=inventory)


def _call_map(calls: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One import owns one immutable mapping; later duplicate markers win."""
    return {call["marker"]: call for call in calls}


def _adapt(events: Sequence[Dict[str, Any]], call_map: Mapping[str, Dict[str, Any]], *,
           trace_uid: str, taxonomy: SourceFormat, context: Any,
           marker_prefix: Optional[str] = None, kind: str = "all",
           inventory: Optional[List[Dict[str, Any]]] = None) -> NormalizedTrace:
    """Normalize a bounded batch using the caller's complete source context."""
    records, devices, launches = [], [], []
    for i, event in enumerate(events):
        args = event.get("args", {})
        pid, cat = event.get("pid"), event.get("cat")
        host_pid = taxonomy.host_pid(event, context)
        task = taxonomy.task_type(event)
        name, phase = event.get("name", ""), event.get("ph")
        category = taxonomy.category(event, context) if taxonomy.normalizable(event, context) else None
        record = {"trace_uid": trace_uid, "loc": event.get("_linker_loc", i),
                  "name": name, "pid": pid, "host_pid": host_pid, "tid": event.get("tid"),
                  "ts": nanoseconds(event.get("ts")), "dur": nanoseconds(event.get("dur")),
                  "ext": {"chrome.raw_ts_ns": nanoseconds(event.get("ts")),
                          "chrome.raw_dur_ns": nanoseconds(event.get("dur")),
                          "chrome.clock_domain": trace_uid, "chrome.cat": cat,
                          "chrome.ph": phase, "chrome.id": event.get("id"), "chrome.args": dict(args)}}
        if category is not None:
            record.update(t="kernel" if category in ("compute","unknown") else "comm.device" if category == "comm" else "memop",
                          category=category, task_type=task)
            if kind == "all" or category == ("comm" if kind == "communication" else "compute"):
                devices.append(record)
        elif phase == "X" and cat not in taxonomy.contract['roles']['allowed_non_device_categories'] and marker_prefix is not None and name.startswith(marker_prefix):
            record.update(t="op", corr=call_map.get(name, {}).get("corr", name))
        elif taxonomy.is_launch(event):
            record["t"] = "launch"
            launches.append(record)
        else:
            record["t"] = "host"
        records.append(record)
    return NormalizedTrace(trace_uid, sorted(records, key=ref_sort), sorted(devices, key=ref_sort),
                           sorted(launches, key=ref_sort), inventory or [])
