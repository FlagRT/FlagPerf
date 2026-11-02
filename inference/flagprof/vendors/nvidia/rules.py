"""CUDA fixture rules; no vendor-specific decisions in the core."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from flagprof.correlate.linker import Hit, NormalizedTrace, Record, Rule, identity
from flagprof.vendors.rules_common import Intervals, PreparedRule, flow, same_thread



def build_rules(trace: NormalizedTrace) -> List[Rule]:
    launches: Dict[Tuple[Any, ...], List[Record]] = defaultdict(list)
    external: Dict[Tuple[Any, ...], List[Record]] = defaultdict(list)
    for record in trace.records:
        args, cat = record["ext"]["chrome.args"], record["ext"]["chrome.cat"]
        if cat in ("cuda_runtime", "cuda_driver") and args.get("correlation") is not None:
            launches[record["trace_uid"], str(args["correlation"])].append(record)
        # T9.0 predicate-inventory: project-anchor candidates, not host-track categories.
        if cat in ("cpu_op", "user_annotation") and args.get("External id") is not None:
            external[record["trace_uid"], record["host_pid"], str(args["External id"])].append(record)
    device_hits = {}
    for event in trace.devices:
        key = event["ext"]["chrome.args"].get("correlation")
        device_hits[identity(event)] = [Hit(source) for source in launches.get((event["trace_uid"], str(key)), [])] if key is not None else []
    owners = Intervals([record for record in trace.records if record["t"] == "op"])
    host_hits = {}
    for launch in trace.launches:
        key = launch["ext"]["chrome.args"].get("External id")
        hits = []
        if key is not None:
            for source in external.get((launch["trace_uid"], launch["host_pid"], str(key)), []):
                candidates = [source] if source["t"] == "op" else owners.contains(source)
                hits.extend(Hit(owner, (source,)) for owner in candidates)
        host_hits[identity(launch)] = hits
    return [flow(trace, "cuda", "ac2g"),
            PreparedRule("cuda.correlation_id", "device", ("trace_uid", "correlation_id"), device_hits),
            same_thread(trace, "cuda"),
            PreparedRule("cuda.external_id", "host", ("trace_uid", "host_pid", "external_id"), host_hits)]
