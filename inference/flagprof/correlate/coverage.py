"""Single-trace collection ratios derived from stored fields and raw inventory."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def device_category(event: Mapping[str, Any]) -> str | None:
    if event.get("t") == "kernel":
        kind = event.get("ktype", "compute")
        return kind if kind in ("compute", "comm", "memcpy", "unknown") else "unknown"
    if event.get("t") == "memop":
        return "memcpy"
    if event.get("t") == "comm" and event.get("phase") == "device" and "link" in event:
        return "comm"
    return None


def coverage(events: Sequence[Mapping[str, Any]], inventory: Mapping[str, Any] | None) -> dict[str, Any]:
    """No vendor names, source clocks, or ext reads participate in ratios."""
    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None
    categories = {}
    for category in ("compute", "comm", "memcpy", "unknown"):
        imported = [event for event in events if device_category(event) == category]
        eligible = [event for event in imported if event["link"]["status"] != "not_applicable"]
        linked = [event for event in eligible if event["link"]["status"] == "linked"]
        duration = sum(event["dur"] for event in eligible if event.get("dur") is not None)
        raw = None if inventory is None else inventory["categories"][category]
        levels = {}
        for level in ("A", "B+", "C"):
            def derived_level(event: Mapping[str, Any]) -> str:
                segments = event["link"]["segments"]
                return "A" if segments["host_attr"] == "ok" else "B+" if segments["device_link"] == "ok" else "C"
            selected = [event for event in eligible if derived_level(event) == level]
            levels[level] = {"count": len(selected), "duration_ns": sum(event["dur"] for event in selected if event.get("dur") is not None)}
        categories[category] = {
            "R": raw, "E": len(imported), "X": len(imported) - len(eligible), "A": len(eligible), "L": len(linked),
            "linked_count_ratio": ratio(len(linked), len(eligible)),
            "linked_time_ratio": ratio(sum(event["dur"] for event in linked if event.get("dur") is not None), duration),
            "import_completeness": "unknown" if raw is None else ratio(len(imported), raw),
            "dur_missing": sum(event.get("dur") is None for event in eligible), "level_distribution": levels,
        }
    return {"categories": categories,
            "unrecognized_ktype": sum(e.get("t") == "kernel" and e.get("ktype", "compute") not in
                                      ("compute", "comm", "memcpy", "unknown") for e in events)}
