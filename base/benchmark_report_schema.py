# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence-only rendering for vendor-neutral summary v3 / monitor v2.

Older Ascend summaries remain supported by the existing report renderer.
No provider import, device discovery or host configuration is needed here.
"""
from __future__ import annotations

import html
import json
import math
import statistics
from typing import Any


def cell(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")


def table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows], ""]


def identity(item: dict[str, Any]) -> str:
    if item.get("device_id") is not None:
        return str(item["device_id"])
    if item.get("binding", {}).get("resource_key") is not None:
        return str(item["binding"]["resource_key"])
    if item.get("logical_device_id") is not None:
        return "legacy-logical/" + str(item["logical_device_id"])
    return "unidentified"


def metrics(policy: dict[str, Any]) -> list[dict[str, str]]:
    return policy.get("metric_fields", [])


def selected_samples(samples, windows):
    result = []
    for sample in samples:
        for window in windows:
            if window.get("role") not in (None, "measurement"):
                continue
            if identity(window) != identity(sample):
                # Ascend v1 collectors retain logical IDs; use them only when
                # a sample lacks the canonical device identity.
                if sample.get("device_id") is not None or sample.get("logical_device_id") is None or sample.get("logical_device_id") != window.get("logical_device_id"):
                    continue
            try:
                start, end = float(sample["started_offset_s"]), float(sample["finished_offset_s"])
                lo, hi = float(window["started_offset_s"]), float(window["finished_offset_s"])
                if all(math.isfinite(v) for v in (start, end, lo, hi)) and end >= start and hi > lo and start <= hi and end >= lo:
                    result.append(sample)
                    break
            except (KeyError, ValueError, TypeError):
                continue
    return result


def numeric_samples(samples, key):
    result = []
    for sample in samples:
        value = sample.get("values", {}).get(key)
        if sample.get("valid") is not False and type(value) in (int, float) and math.isfinite(value):
            result.append((sample, value))
    return result


def monitor_svg(samples, monitor):
    policy = monitor.get("policy", {})
    samples = selected_samples(samples, monitor.get("workload_windows", []))
    rows = []
    for metric in metrics(policy):
        for device in sorted({identity(item) for item in samples}):
            points = numeric_samples([item for item in samples if identity(item) == device], metric["key"])
            if points:
                rows.append((device, metric, points))
    if not rows:
        return None
    height = 60 + len(rows) * 110
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<text x="20" y="25" font-size="16">Raw telemetry overlapping rank measurement windows</text>']
    for row, (device, metric, points) in enumerate(rows):
        y = 60 + row * 110
        values = [value for _, value in points]
        times = [float(item["finished_offset_s"]) for item, _ in points]
        lo, hi = min(values), max(values)
        t0, t1 = min(times), max(times)
        label = html.escape(f"{device}: {metric['label']} ({metric['unit']}) [{lo:g}, {hi:g}]")
        output.append(f'<text x="20" y="{y}" font-size="13">{label}</text>')
        for (item, value), timestamp in zip(points, times):
            x = 30 + 930 * (timestamp-t0) / (t1-t0 or 1)
            py = y + 75 - 55 * (value-lo) / (hi-lo or 1)
            output.append(f'<circle cx="{x:.3f}" cy="{py:.3f}" r="3" fill="#2463a4"><title>{html.escape(str(timestamp)+": "+str(value))}</title></circle>')
    output.append('</svg>')
    return "\n".join(output) + "\n"


def render_report(root, summary, result, rank_assets):
    plan = summary.get("resolved_plan") or summary.get("static_plan") or {}
    runtime = summary.get("runtime", {})
    lines = ["# FlagPerf Base Benchmark", "", "## Run", ""]
    if summary.get("status") == "skipped":
        lines += ["启动前跳过：" + cell(summary.get("skip_reason")), ""]
    lines += table(["Field", "Evidence"], [(key, summary.get(key)) for key in (
        "run_id", "vendor", "vendor_display_name", "case", "status", "execution_status", "measurement_status", "monitoring_status", "failure_stage", "error", "skip_reason")])
    lines += ["## Runtime", ""]
    lines += table(["Field", "Evidence"], [("image", runtime.get("image", plan.get("image"))),
        ("image_id", runtime.get("image_id")), ("lock", runtime.get("lock", plan.get("runtime_lock")))])
    lines += ["## Device bindings", ""]
    lines += table(["Local rank", "Framework ID", "Physical ID", "Resource", "Host node", "Container node", "PCI", "UUID/serial"],
        [(b.get("framework_local_rank"), b.get("framework_logical_id"), b.get("host_physical_id"), b.get("resource_key"),
          b.get("host_device_node"), b.get("container_device_node"), b.get("pci_bdf"), b.get("serial_or_uuid")) for b in summary.get("device_bindings", [])])
    lines += ["## Rank metrics", ""]
    lines += table(["Rank", "Metric", "Value", "Unit"], [(m.get("rank"),m.get("metric"),m.get("value"),m.get("unit")) for m in result.get("metrics", [])
        if type(m.get("value")) in (float,int) and math.isfinite(m["value"])])
    lines += [f"Missing ranks: {cell(result.get('missing_ranks', []))}", ""]
    for asset in rank_assets:
        lines += [f"![Rank metrics]({asset['path']})", ""]
    lines += ["## Case configuration", "", "Precedence: generic < vendor < chip < override.", ""]
    configurations = summary.get("case_config", plan.get("case_config", {}))
    lines += table(["Layer", "Source", "SHA256", "Snapshot"], [(key, item.get("path"),item.get("sha256"),item.get("artifact_path")) for key,item in configurations.items()])
    lines += ["## Evidence", "", "[Monitor report](report_monitor.md)", ""]
    for name in ["summary.json", "resolved-plan.json", "case-assets.json", "benchmark-result.json", "runner.log", "container-preflight.json", "benchmark-monitor/summary.json"]:
        if (root/name).is_file():
            lines += [f"- [{name}]({name})"]
    lines += ["", "Host preflight: " + cell(summary.get("host_preflight")), "", "Host postflight: " + cell(summary.get("host_postflight")),
              "", "This report preserves the experiment status. Missing evidence is not evidence of success.", ""]
    return "\n".join(lines)


def render_monitor_report(root, summary, monitor, samples, asset):
    policy = monitor.get("policy", {})
    windows = monitor.get("workload_windows", [])
    selected = selected_samples(samples, windows)
    lines = ["# Benchmark device telemetry", ""]
    lines += table(["Field", "Evidence"], [("vendor",monitor.get("vendor",summary.get("vendor"))),
        ("status",monitor.get("status")), ("collector",monitor.get("collector",policy.get("collector"))),
        ("reasons",monitor.get("reasons")), ("policy",policy), ("rank_device_map",monitor.get("rank_device_map")),
        ("measurement_windows",windows), ("lifecycle_windows",monitor.get("lifecycle_windows")),
        ("raw_samples",monitor.get("raw_samples")), ("parsed_samples",monitor.get("parsed_samples")),
        ("sample_counts",monitor.get("primary_sample_counts_by_target")), ("events",monitor.get("event_artifacts"))])
    rows = []
    for metric in metrics(policy):
        for device in sorted({identity(item) for item in selected}):
            values = [v for _,v in numeric_samples([item for item in selected if identity(item)==device],metric["key"])]
            rows.append((device,metric["label"],metric["unit"],len(values),min(values) if values else None,
                         statistics.median(values) if values else None,max(values) if values else None))
    lines += table(["Device", "Metric", "Unit", "Samples", "Min", "Median", "Max"], rows)
    if asset:
        lines += [f"![Device telemetry]({asset['path']})", ""]
    lines += ["Only command intervals overlapping the recorded rank measurement windows are summarized.",
              "Missing or invalid measurements remain missing. Driver integration windows and theoretical utilization are not inferred.", ""]
    return "\n".join(lines)
