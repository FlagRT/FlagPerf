# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Vendor-neutral candidate-union core. No clock proximity or SDK imports."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import fnmatch
import json
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

Record = Dict[str, Any]
Reference = Dict[str, Any]


def ref(record: Mapping[str, Any]) -> Reference:
    return {"trace_uid": record["trace_uid"], "loc": record["loc"]}


def identity(record: Mapping[str, Any]) -> Tuple[str, str]:
    return str(record["trace_uid"]), str(record["loc"])


def ref_sort(reference: Mapping[str, Any]) -> Tuple[str, int, str]:
    value = str(reference["loc"])
    stem = value.split("+dup", 1)[0]
    return str(reference["trace_uid"]), int(stem), value[len(stem):]


def unique_refs(values: Sequence[Mapping[str, Any]]) -> List[Reference]:
    return sorted({identity(v): ref(v) for v in values}.values(), key=ref_sort)


def corr_sort(value: Any) -> Tuple[int, Any]:
    return (0, value) if isinstance(value, int) else (1, str(value))


@dataclass(frozen=True)
class Hit:
    target: Record
    evidence: Tuple[Record, ...] = ()


class Rule(Protocol):
    via: str
    side: str
    namespaces: Tuple[str, ...]

    def matches(self, target: Record) -> Sequence[Hit]: ...


@dataclass
class NormalizedTrace:
    trace_uid: str
    records: List[Record]
    devices: List[Record]
    launches: List[Record]
    inventory: List[Record]


class Linker(Protocol):
    def __call__(self, trace: NormalizedTrace, rules: Sequence[Rule],
                 registry: Mapping[str, Any]) -> Dict[str, Any]: ...


def registry_match(event: Record, registry: Mapping[str, Any]) -> bool:
    """Only explicit registry entries may exclude a device record."""
    if event.get("ext", {}).get("graph_replay"):
        return False
    for entry in registry.get("entries", []):
        match = entry.get("match", {})
        if not match:
            continue
        if all(fnmatch.fnmatchcase(str(event.get(field, "")), str(pattern))
               for field, pattern in match.items()):
            return True
    return False


def ratio(numerator: int, denominator: int) -> Any:
    return numerator / denominator if denominator else None


def summarize(trace: NormalizedTrace, devices: List[Record], launches: List[Record]) -> Dict[str, Any]:
    categories: Dict[str, Any] = {}
    for category in ("compute", "comm", "memcpy", "unknown"):
        raw = [r for r in trace.inventory if r["category"] == category]
        imported = [e for e in devices if e["category"] == category]
        excluded = [e for e in imported if e["link"]["status"] == "not_applicable"]
        attributable = [e for e in imported if e["link"]["status"] != "not_applicable"]
        linked = [e for e in attributable if e["link"]["status"] == "linked"]
        available = [e for e in attributable if e.get("dur") is not None]
        duration = sum(e["dur"] for e in available)
        distribution = {}
        for level in ("A", "B+", "C"):
            selected = [e for e in attributable if e["link"]["level"] == level]
            distribution[level] = {"count": len(selected),
                                   "duration_ns": sum(e["dur"] for e in selected if e.get("dur") is not None)}
        categories[category] = {
            "R": len(raw), "E": len(imported), "X": len(excluded),
            "A": len(attributable), "L": len(linked),
            "linked_count_ratio": ratio(len(linked), len(attributable)),
            "linked_time_ratio": ratio(sum(e["dur"] for e in linked if e.get("dur") is not None), duration),
            "import_completeness": ratio(len(imported), len(raw)),
            "dur_missing": sum(e.get("dur") is None for e in attributable),
            "level_distribution": distribution,
        }
    host_distribution = Counter((e["segments"]["host_attr"], e["segments"]["device_link"])
                                for e in launches)
    return {"categories": categories,
            "level_distribution": {k: v["level_distribution"] for k, v in categories.items()},
            "launch_attr_ratio": ratio(sum(e["host_attr"] == "ok" for e in launches), len(launches)),
            "launch_distribution": {a + "/" + b: host_distribution[a, b]
                                    for a in ("ok", "broken") for b in ("ok", "broken")},
            "reason_counts": dict(sorted(Counter(e["link"]["reason"] for e in devices
                                                if "reason" in e["link"]).items())),
            "time_exclusions": 0,
            "time_exclusion_policy": "No time exclusion rules; source-clock containment and flow equality only."}


def link(trace: NormalizedTrace, rules: Sequence[Rule], registry: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply every applicable rule, union all launches and all owner candidates."""
    device_rules = [rule for rule in rules if rule.side == "device"]
    host_rules = [rule for rule in rules if rule.side == "host"]
    owner_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for launch in trace.launches:
        owners: Dict[Any, Record] = {}
        evidence: List[Record] = [launch]
        via = set()
        for rule in host_rules:
            hits = rule.matches(launch)
            if hits:
                via.add(rule.via)
            for hit in hits:
                owners[hit.target["corr"]] = hit.target
                evidence.extend((hit.target,) + hit.evidence)
        owner_info[identity(launch)] = {"candidates": sorted(owners, key=corr_sort),
                                        "via": via, "evidence": evidence}
    matched = set()
    devices: List[Record] = []
    for event in sorted(trace.devices, key=ref_sort):
        if registry_match(event, registry):
            devices.append(dict(event, link={"status": "not_applicable", "reason": "excluded",
                                              "via": [], "evidence": [ref(event)]}))
            continue
        sources: Dict[Tuple[str, str], Record] = {}
        via = set()
        evidence: List[Record] = [event]
        for rule in device_rules:
            hits = rule.matches(event)
            if hits:
                via.add(rule.via)
            for hit in hits:
                sources[identity(hit.target)] = hit.target
                evidence.extend((hit.target,) + hit.evidence)
        candidates = set()
        unknown = False
        for key, launch in sources.items():
            matched.add(key)
            info = owner_info[key]
            candidates.update(info["candidates"])
            unknown |= not info["candidates"]
            via.update(info["via"])
            evidence.extend(info["evidence"])
        ordered = sorted(candidates, key=corr_sort)
        result: Dict[str, Any] = {"via": sorted(via), "evidence": unique_refs(evidence)}
        if len(ordered) >= 2 or (ordered and unknown):
            result.update(status="ambiguous", reason="conflict", candidates=ordered + ([None] if unknown else []))
        elif len(ordered) == 1:
            result.update(status="linked", corr=ordered[0])
        else:
            reason = "no_bridge" if sources else "no_device_key"
            if event.get("ext", {}).get("source_missing"):
                reason = "source_missing"
            if event.get("ext", {}).get("graph_replay"):
                reason = "graph_replay"
            result.update(status="unlinked", reason=reason)
        result["segments"] = {"host_attr": "ok" if result["status"] == "linked" else "broken",
                              "device_link": "ok" if sources else "broken"}
        result["level"] = "A" if result["status"] == "linked" else "B+" if sources else "C"
        devices.append(dict(event, link=result))
    launches = []
    for launch in sorted(trace.launches, key=ref_sort):
        info = owner_info[identity(launch)]
        owned = len(info["candidates"]) == 1
        seen = identity(launch) in matched
        output = dict(launch, host_attr="ok" if owned else "broken", matched_device=seen,
                      candidates=info["candidates"], via=sorted(info["via"]),
                      evidence=unique_refs(info["evidence"]),
                      segments={"host_attr": "ok" if owned else "broken", "device_link": "ok" if seen else "broken"},
                      level="A" if owned and seen else "B" if owned else "B+" if seen else "C")
        if owned:
            output["corr"] = info["candidates"][0]
        launches.append(output)
    return {"trace_uid": trace.trace_uid, "devices": devices, "launches": launches,
            "summary": summarize(trace, devices, launches)}
