"""Source-clock structural indexes shared by vendor rule implementations."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from flagprof.correlate.linker import Hit, NormalizedTrace, Record, identity, ref_sort



def raw_interval(record: Record) -> Tuple[Any, Any]:
    start = record["ext"]["chrome.raw_ts_ns"]
    duration = record["ext"]["chrome.raw_dur_ns"] or 0
    return start, start + duration if start is not None else None


def source_thread(record: Record) -> Tuple[Any, ...]:
    return record["trace_uid"], record["host_pid"], record["tid"], record["ext"]["chrome.clock_domain"]


class Intervals:
    def __init__(self, records: Sequence[Record]) -> None:
        groups: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
        for record in records:
            start, end = raw_interval(record)
            if start is not None:
                groups[source_thread(record)].append((start, end, ref_sort(record), record))
        self.groups = {}
        for key, values in groups.items():
            values.sort(key=lambda v: (v[0], v[1], v[2]))
            maximum, maxima = -1, []
            for start, end, _, record in values:
                maximum = max(maximum, end)
                maxima.append(maximum)
            self.groups[key] = ([v[0] for v in values], values, maxima)

    def contains(self, record: Record) -> List[Record]:
        start, end = raw_interval(record)
        if start is None:
            return []
        starts, values, maxima = self.groups.get(source_thread(record), ([], [], []))
        position = bisect_right(starts, start) - 1
        selected = []
        while position >= 0 and maxima[position] > start:
            a, b, _, candidate = values[position]
            if a <= start < b and end <= b:
                selected.append(candidate)
            position -= 1
        return sorted(selected, key=ref_sort)


class PreparedRule:
    """One immutable hit table; every match is returned, including duplicates."""
    def __init__(self, via: str, side: str, namespaces: Tuple[str, ...],
                 hits: Dict[Tuple[str, str], List[Hit]]) -> None:
        self.via, self.side, self.namespaces = via, side, namespaces
        self.hits = {key: tuple(sorted(values, key=lambda h: (ref_sort(h.target),
                                                            tuple(ref_sort(e) for e in h.evidence))))
                     for key, values in hits.items()}

    def matches(self, target: Record) -> Sequence[Hit]:
        return self.hits.get(identity(target), ())


def same_thread(trace: NormalizedTrace, prefix: str) -> PreparedRule:
    index = Intervals([record for record in trace.records if record["t"] == "op"])
    return PreparedRule(prefix + ".same_thread", "host", ("trace_uid", "host_pid", "tid", "source_clock"),
                        {identity(launch): [Hit(owner) for owner in index.contains(launch)]
                         for launch in trace.launches})


def flow(trace: NormalizedTrace, prefix: str, category: str) -> PreparedRule:
    groups: Dict[Tuple[Any, ...], Dict[str, List[Record]]] = defaultdict(lambda: {"s": [], "f": []})
    endpoints: Dict[Tuple[Any, ...], List[Any]] = defaultdict(list)
    for record in trace.records:
        ext = record["ext"]
        if ext["chrome.cat"] == category and ext["chrome.ph"] in ("s", "f") and ext["chrome.id"] is not None:
            key = record["trace_uid"], category, str(ext["chrome.id"])
            groups[key][ext["chrome.ph"]].append(record)
    for phases in groups.values():
        for end in phases["f"]:
            key = end["trace_uid"], end["pid"], end["tid"], end["ext"]["chrome.raw_ts_ns"]
            for start in phases["s"]:
                if start["t"] == "launch":
                    endpoints[key].append((start, end))
    return PreparedRule(prefix + ".flow", "device", ("trace_uid", "flow_category", "flow_id", "pid", "tid", "source_clock"),
                        {identity(event): [Hit(start, (end,)) for start, end in endpoints.get(
                            (event["trace_uid"], event["pid"], event["tid"], event["ext"]["chrome.raw_ts_ns"]), [])]
                         for event in trace.devices})
