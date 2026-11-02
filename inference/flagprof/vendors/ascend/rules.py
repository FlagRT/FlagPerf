"""CANN explicit-key rules. Multiplicity always reaches the union core."""
from __future__ import annotations
from .format import queue_key_valid

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from flagprof.correlate.linker import Hit, NormalizedTrace, Record, Rule, identity
from flagprof.vendors.rules_common import Intervals, PreparedRule, flow, same_thread



VERSION = 'cann-project-t8.3-queue-v2'


def correlation_keys(record: Record) -> Sequence[Tuple[str, int]]:
    """OpMarkBean correlation_id is a queue ID, not a wildcard field family."""
    value=record['ext']['chrome.args'].get('correlation_id')
    if queue_key_valid(value):
        return [('torch_npu.queue',value)]
    return []


def build_rules(trace: NormalizedTrace) -> List[Rule]:
    connections: Dict[Tuple[Any, ...], List[Record]] = defaultdict(list)
    enqueues: Dict[Tuple[Any, ...], List[Record]] = defaultdict(list)
    for record in trace.records:
        args = record["ext"]["chrome.args"]
        if record["name"] == "Node@launch" and args.get("connection_id") is not None:
            connections[record["trace_uid"], str(args["connection_id"])].append(record)
        if record["name"].startswith("Enqueue"):
            for key in correlation_keys(record):
                enqueues[record["trace_uid"], record["host_pid"], key].append(record)
    device_hits = {}
    for event in trace.devices:
        key = event["ext"]["chrome.args"].get("connection_id")
        device_hits[identity(event)] = [Hit(source) for source in connections.get((event["trace_uid"], str(key)), [])] if key is not None else []
    owner_index = Intervals([record for record in trace.records if record["t"] == "op"])
    dequeue_index = Intervals([record for record in trace.records
                              if record["name"].startswith("Dequeue") and record["ext"]["chrome.ph"] == "X"])
    host_hits = {}
    for launch in trace.launches:
        hits = []
        for dequeue in dequeue_index.contains(launch):
            for key in correlation_keys(dequeue):
                for enqueue in enqueues.get((dequeue["trace_uid"], dequeue["host_pid"], key), []):
                    # Do not require a unique Enqueue. Both owners survive.
                    for owner in owner_index.contains(enqueue):
                        hits.append(Hit(owner, (dequeue, enqueue)))
        host_hits[identity(launch)] = hits
    return [flow(trace, "cann", "HostToDevice"),
            PreparedRule("cann.connection_id", "device", ("trace_uid", "connection_id"), device_hits),
            same_thread(trace, "cann"),
            PreparedRule("cann.enqueue_corr", "host", ("trace_uid", "host_pid", "enqueue_correlation_id", "source_clock", "tid"), host_hits)]
