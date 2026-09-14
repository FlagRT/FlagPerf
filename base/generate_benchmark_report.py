#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Generate deterministic, evidence-linked reports for Base Benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = 3
SUPPORTED_SUMMARY_SCHEMA_VERSIONS = (1, 2)
SUPPORTED_RESULT_SCHEMA_VERSIONS = (1,)
SUPPORTED_MONITOR_SCHEMA_VERSIONS = (1,)


class ReportError(RuntimeError):
    """Raised when stored Benchmark evidence cannot be rendered safely."""


def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ReportError(f"required report input does not exist: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"JSON artifact must contain an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"cannot read JSONL artifact {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(
                f"cannot parse JSONL artifact {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ReportError(
                f"JSONL artifact must contain objects: {path}:{line_number}"
            )
        values.append(value)
    return values


def schema_version(
    value: dict[str, Any], name: str, supported: tuple[int, ...],
) -> int:
    version = value.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ReportError(f"invalid {name} schema: {version!r}")
    if version not in supported:
        raise ReportError(f"unsupported {name} schema: {version}")
    return version


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_summary_sha256(summary: dict[str, Any]) -> str:
    stable = dict(summary)
    stable.pop("report_generation", None)
    payload = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def md_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def code(value: Any) -> str:
    rendered = (
        str(value).replace("`", "\\`").replace("|", "\\|").replace("\n", " ")
    )
    return f"`{rendered}`"


def status_text(value: Any) -> str:
    labels = {
        "passed": "通过",
        "partial": "部分完成",
        "failed": "失败",
        "skipped": "跳过",
        "running": "运行中",
        "not_started": "未开始",
        "not_available": "不可用",
        "not-run": "未运行",
        None: "未记录",
    }
    return f"{labels.get(value, '未知')} ({code(value if value is not None else 'missing')})"


def format_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return str(value)
    if number == 0:
        return "0"
    magnitude = abs(number)
    if magnitude >= 1_000_000 or magnitude < 0.0001:
        return f"{number:.{digits}e}"
    return f"{number:,.{digits}f}".rstrip("0").rstrip(".")


def format_duration(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "未记录"
    seconds = float(value)
    if seconds < 60:
        return f"{format_number(seconds, 3)} s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {format_number(remainder, 1)} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {format_number(remainder, 1)} s"


def sequence_text(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return "未记录" if value is None else str(value)


def finite_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = result.get("metrics")
    if not isinstance(metrics, list):
        return []
    return [
        item for item in metrics
        if isinstance(item, dict)
        and isinstance(item.get("value"), (int, float))
        and not isinstance(item.get("value"), bool)
        and math.isfinite(float(item["value"]))
    ]


def metric_groups(
    metrics: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metric in metrics:
        key = (str(metric.get("metric", "unnamed")), str(metric.get("unit", "unitless")))
        groups.setdefault(key, []).append(metric)
    for values in groups.values():
        values.sort(key=lambda item: (int(item.get("rank", -1)), str(item.get("raw", ""))))
    return dict(sorted(groups.items()))


def group_statistics(metrics: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(item["value"]) for item in metrics]
    mean = statistics.fmean(values)
    minimum = min(values)
    maximum = max(values)
    unique_ranks = {item.get("rank") for item in metrics}
    multiple_ranks = len(unique_ranks) > 1
    spread_percent = (
        None if mean == 0 or not multiple_ranks
        else (maximum - minimum) / abs(mean) * 100
    )
    balance_percent = None
    if multiple_ranks and minimum >= 0 and maximum > 0:
        balance_percent = minimum / maximum * 100
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "population_stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "spread_percent": spread_percent,
        "balance_percent": balance_percent,
    }


def svg_document(
    width: int, height: int, title: str, description: str, body: str,
) -> str:
    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{escaped_title}</title>\n"
        f"<desc id=\"desc\">{escaped_description}</desc>\n"
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f'<text x="28" y="36" font-family="sans-serif" font-size="20" '
        f'font-weight="700" fill="#172033">{escaped_title}</text>\n'
        f"{body}\n</svg>\n"
    )


def rank_metric_svg(
    metric_name: str, unit: str, metrics: list[dict[str, Any]],
) -> str | None:
    values = [float(item["value"]) for item in metrics]
    if not values or min(values) < 0 or max(values) <= 0:
        return None
    width = 980
    row_height = 46
    height = 92 + row_height * len(metrics)
    label_width = 115
    chart_left = 145
    chart_width = 660
    maximum = max(values)
    body = [
        f'<text x="28" y="62" font-family="sans-serif" font-size="12" '
        f'fill="#5b6474">同一次运行内各 rank 原始值；横轴以本组最大值归一化，非理论峰值利用率。</text>'
    ]
    for index, item in enumerate(metrics):
        y = 82 + index * row_height
        rank = html.escape(str(item.get("rank", "?")))
        value = float(item["value"])
        bar_width = value / maximum * chart_width
        body.extend([
            f'<text x="{label_width}" y="{y + 22}" text-anchor="end" '
            f'font-family="monospace" font-size="13" fill="#263247">Rank {rank}</text>',
            f'<rect x="{chart_left}" y="{y + 7}" width="{chart_width}" height="22" '
            f'rx="4" fill="#edf2f8"/>',
            f'<rect x="{chart_left}" y="{y + 7}" width="{bar_width:.2f}" height="22" '
            f'rx="4" fill="#3977d4"/>',
            f'<text x="{chart_left + chart_width + 14}" y="{y + 23}" '
            f'font-family="monospace" font-size="13" fill="#172033">'
            f'{html.escape(format_number(value))} {html.escape(unit)}</text>',
        ])
    return svg_document(
        width,
        height,
        f"{metric_name} / Rank 分布",
        f"Benchmark metric {metric_name} in {unit}, grouped by rank.",
        "\n".join(body),
    )


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()
    return slug[:80] or "metric"


def write_asset(root: Path, name: str, content: str | None) -> dict[str, Any] | None:
    if content is None:
        return None
    path = root / "report-assets" / name
    atomic_write(path, content)
    return artifact_ref(path, root)


MONITOR_FIELDS = (
    ("aicore_usage_rate_pct", "AICore", "Aicore Usage Rate(%)"),
    ("aivector_usage_rate_pct", "AIVector", "Aivector Usage Rate(%)"),
    ("hbm_usage_rate_pct", "HBM 占用", "HBM Usage Rate(%)"),
    (
        "hbm_bandwidth_usage_rate_pct", "HBM 带宽",
        "HBM Bandwidth Usage Rate(%)",
    ),
    ("npu_utilization_pct", "NPU", "NPU Utilization(%)"),
)
MAX_VISIBLE_MONITOR_SAMPLES_PER_TARGET = 64


def monitor_target(sample: dict[str, Any]) -> tuple[int, int, int] | None:
    try:
        return (
            int(sample["npu_id"]), int(sample["chip_id"]),
            int(sample["logical_device_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def monitor_source_value(
    sample: dict[str, Any], normalized: str, source: str,
) -> str:
    source_fields = sample.get("source_fields")
    if isinstance(source_fields, dict) and source_fields.get(source) is not None:
        return str(source_fields[source])
    values = sample.get("values")
    value = values.get(normalized) if isinstance(values, dict) else None
    return str(value) if isinstance(value, (int, float)) else "—"


def monitor_windows(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    values = monitor.get("workload_windows")
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def sample_overlaps_measurement(
    sample: dict[str, Any], windows: list[dict[str, Any]],
) -> bool:
    start = sample.get("started_offset_s")
    finish = sample.get("finished_offset_s")
    logic_id = sample.get("logical_device_id")
    if not isinstance(start, (int, float)) or not isinstance(finish, (int, float)):
        return False
    for window in windows:
        if window.get("role") != "measurement":
            continue
        if window.get("logical_device_id") not in (None, logic_id):
            continue
        window_start = window.get("started_offset_s")
        window_finish = window.get("finished_offset_s")
        if (
            isinstance(window_start, (int, float))
            and isinstance(window_finish, (int, float))
            and float(start) <= float(window_finish)
            and float(finish) >= float(window_start)
        ):
            return True
    return False


def monitor_statistics(
    samples: list[dict[str, Any]], windows: list[dict[str, Any]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    targets = sorted({target for item in samples if (target := monitor_target(item))})
    for target in targets:
        valid = [
            item for item in samples
            if monitor_target(item) == target
            and item.get("valid") is True
            and sample_overlaps_measurement(item, windows)
        ]
        for normalized, label, _source in MONITOR_FIELDS:
            values = [
                float(item["values"][normalized])
                for item in valid
                if isinstance(item.get("values"), dict)
                and isinstance(item["values"].get(normalized), (int, float))
                and not isinstance(item["values"].get(normalized), bool)
            ]
            rows.append([
                *target, label, len(values),
                min(values) if values else None,
                statistics.median(values) if values else None,
                statistics.fmean(values) if values else None,
                max(values) if values else None,
            ])
    return rows


def visible_monitor_samples(
    samples: list[dict[str, Any]], windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Select a deterministic readable projection; JSONL stays authoritative."""
    in_window = [
        item for item in samples if sample_overlaps_measurement(item, windows)
    ]
    source = in_window or samples
    selected: list[dict[str, Any]] = []
    targets = sorted({
        target for item in source if (target := monitor_target(item))
    })
    for target in targets:
        group = sorted(
            (item for item in source if monitor_target(item) == target),
            key=lambda item: (
                float(item.get("started_offset_s", math.inf)),
                int(item.get("sequence", sys.maxsize)),
            ),
        )
        if len(group) <= MAX_VISIBLE_MONITOR_SAMPLES_PER_TARGET:
            selected.extend(group)
            continue
        # Include both endpoints and evenly spaced interior samples.  This is
        # a display projection only; statistics still use every valid sample.
        last = len(group) - 1
        indices = {
            round(index * last / (MAX_VISIBLE_MONITOR_SAMPLES_PER_TARGET - 1))
            for index in range(MAX_VISIBLE_MONITOR_SAMPLES_PER_TARGET)
        }
        selected.extend(group[index] for index in sorted(indices))
    selected.sort(key=lambda item: (
        float(item.get("started_offset_s", math.inf)),
        monitor_target(item) or (sys.maxsize, sys.maxsize, sys.maxsize),
    ))
    return selected, len(source), len(samples)


def monitor_usage_svg(
    samples: list[dict[str, Any]], windows: list[dict[str, Any]],
) -> str | None:
    usable = [
        item for item in samples
        if monitor_target(item) is not None
        and isinstance(item.get("started_offset_s"), (int, float))
        and isinstance(item.get("finished_offset_s"), (int, float))
    ]
    if not usable:
        return None
    targets = sorted({monitor_target(item) for item in usable})
    maximum_time = max(
        [float(item["finished_offset_s"]) for item in usable]
        + [
            float(item["finished_offset_s"])
            for item in windows
            if isinstance(item.get("finished_offset_s"), (int, float))
        ]
        + [1.0]
    )
    minimum_time = min(
        [float(item["started_offset_s"]) for item in usable]
        + [
            float(item["started_offset_s"])
            for item in windows
            if isinstance(item.get("started_offset_s"), (int, float))
        ]
        + [0.0]
    )
    span = max(1e-9, maximum_time - minimum_time)
    maximum_samples = max(
        sum(monitor_target(item) == target for item in usable)
        for target in targets
    )
    left, right, chart_top = 240, 34, 108
    chart_width = max(900, maximum_samples * 64)
    row_height, header_height, target_gap = 29, 24, 14
    target_height = header_height + len(MONITOR_FIELDS) * row_height + target_gap
    chart_bottom = chart_top + len(targets) * target_height - target_gap
    width, height = left + chart_width + right, chart_bottom + 48

    def xpos(value: float) -> float:
        return left + (value - minimum_time) / span * chart_width

    def color(value: float) -> str:
        ratio = max(0.0, min(value, 100.0)) / 100.0
        start, end = (236, 243, 251), (24, 93, 164)
        rgb = tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))
        return "#" + "".join(f"{item:02X}" for item in rgb)

    body = [
        '<text x="28" y="64" font-family="sans-serif" font-size="12" '
        'fill="#5b6474">色块数字为 npu-smi 原值；浅蓝背景为各 Device 的精确 rank 测量窗口。</text>'
    ]
    for target_index, target in enumerate(targets):
        assert target is not None
        npu_id, chip_id, logic_id = target
        group_top = chart_top + target_index * target_height
        body.append(
            f'<text x="28" y="{group_top + 17}" font-family="sans-serif" '
            f'font-size="13" font-weight="700" fill="#172033">NPU {npu_id} / '
            f'Chip {chip_id} / Device {logic_id}</text>'
        )
        for window in windows:
            if (
                window.get("role") == "measurement"
                and window.get("logical_device_id") == logic_id
                and isinstance(window.get("started_offset_s"), (int, float))
                and isinstance(window.get("finished_offset_s"), (int, float))
            ):
                start_x = xpos(float(window["started_offset_s"]))
                finish_x = xpos(float(window["finished_offset_s"]))
                body.append(
                    f'<rect x="{start_x:.2f}" y="{group_top + header_height}" '
                    f'width="{max(1.0, finish_x - start_x):.2f}" '
                    f'height="{len(MONITOR_FIELDS) * row_height}" '
                    'fill="#dceafa" opacity="0.55"/>'
                )
        target_samples = [item for item in usable if monitor_target(item) == target]
        for field_index, (normalized, label, source) in enumerate(MONITOR_FIELDS):
            row_top = group_top + header_height + field_index * row_height
            body.extend([
                f'<text x="{left - 12}" y="{row_top + 19}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11" fill="#263247">'
                f'{html.escape(label)}</text>',
                f'<line x1="{left}" y1="{row_top + row_height}" '
                f'x2="{left + chart_width}" y2="{row_top + row_height}" '
                'stroke="#d8dee8"/>',
            ])
            for sample in target_samples:
                start_x = xpos(float(sample["started_offset_s"])) + 1
                finish_x = xpos(float(sample["finished_offset_s"])) - 1
                value = sample.get("values", {}).get(normalized) if isinstance(
                    sample.get("values"), dict
                ) else None
                raw = monitor_source_value(sample, normalized, source)
                valid = isinstance(value, (int, float)) and sample.get("valid") is True
                fill = color(float(value)) if valid else "#cfd6df"
                text_fill = "#ffffff" if valid and float(value) >= 58 else "#172033"
                body.extend([
                    f'<rect x="{start_x:.2f}" y="{row_top + 3}" '
                    f'width="{max(2.0, finish_x - start_x):.2f}" '
                    f'height="{row_height - 6}" rx="3" fill="{fill}" '
                    'stroke="#ffffff" stroke-width="0.8"/>',
                    f'<text x="{(start_x + finish_x) / 2:.2f}" y="{row_top + 19}" '
                    f'text-anchor="middle" font-family="monospace" font-size="9" '
                    f'fill="{text_fill}">{html.escape(raw)}</text>',
                ])
    body.extend([
        f'<line x1="{left}" y1="{chart_bottom + 4}" x2="{left + chart_width}" '
        f'y2="{chart_bottom + 4}" stroke="#697386"/>',
        f'<text x="{left}" y="{chart_bottom + 22}" font-family="monospace" '
        f'font-size="10" fill="#5b6474">{format_number(minimum_time, 3)} s</text>',
        f'<text x="{left + chart_width}" y="{chart_bottom + 22}" text-anchor="end" '
        f'font-family="monospace" font-size="10" fill="#5b6474">'
        f'{format_number(maximum_time, 3)} s</text>',
    ])
    return svg_document(
        width, height, "Benchmark NPU 同窗监控",
        "Five raw npu-smi usage fields aligned to rank-local measurement windows.",
        "\n".join(body),
    )


def result_relative_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def markdown_link(label: str, path: str) -> str:
    target = path.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{md_escape(label)}]({target})"


def config_artifacts(
    root: Path, summary: dict[str, Any],
) -> list[tuple[str, dict[str, Any], Path | None]]:
    records = summary.get("case_config")
    if not isinstance(records, dict):
        static_plan = summary.get("static_plan")
        if isinstance(static_plan, dict):
            records = static_plan.get("case_config")
    if not isinstance(records, dict):
        return []
    values = []
    order = {"generic": 0, "ascend": 1, "override": 2}
    for label, record in sorted(records.items(), key=lambda item: (order.get(item[0], 9), item[0])):
        if not isinstance(record, dict):
            continue
        path = result_relative_file(root, record.get("artifact_path"))
        values.append((label, record, path))
    return values


def parse_top_level_yaml_scalars(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$", line)
        if not match or not match.group(2):
            continue
        raw = match.group(2)
        if " #" in raw:
            raw = raw.split(" #", 1)[0].rstrip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            raw = raw[1:-1]
        values[match.group(1)] = raw
    return values


def resolved_config_rows(
    root: Path, summary: dict[str, Any],
) -> list[tuple[str, str, str]]:
    merged: dict[str, tuple[str, str]] = {}
    for label, _, artifact in config_artifacts(root, summary):
        if artifact is None:
            continue
        try:
            values = parse_top_level_yaml_scalars(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
        for key, value in values.items():
            merged[key] = (value, label)
    return [(key, *merged[key]) for key in sorted(merged)]


def host_result(summary: dict[str, Any], key: str) -> dict[str, Any]:
    record = summary.get(key)
    if not isinstance(record, dict):
        return {}
    result = record.get("result")
    return result if isinstance(result, dict) else {}


def postflight_status(summary: dict[str, Any]) -> Any:
    record = summary.get("host_postflight")
    if not isinstance(record, dict):
        return "not_available"
    return record.get("status") or host_result(summary, "host_postflight").get("status")


def conclusion_reasons(summary: dict[str, Any], result: dict[str, Any]) -> list[str]:
    if summary.get("status") == "skipped":
        return [str(summary.get("skip_reason", "Case 不适用"))]
    reasons: list[str] = []
    execution = summary.get("execution_status")
    measurement = summary.get("measurement_status")
    if execution != "passed":
        reasons.append(f"执行层为 {status_text(execution)}")
    if measurement != "passed":
        reasons.append(f"测量层为 {status_text(measurement)}")
    if summary.get("timed_out") or result.get("timed_out"):
        reasons.append("运行达到 timeout 并被终止")
    returncode = summary.get("container_returncode", result.get("container_returncode"))
    if returncode not in (None, 0):
        reasons.append(f"容器/worker 返回码为 {returncode}")
    missing = result.get("missing_ranks")
    if isinstance(missing, list) and missing:
        reasons.append(f"缺少 Rank {sequence_text(missing)} 的语义结果")
    fallback_count = result.get("fallback_count")
    if isinstance(fallback_count, int) and fallback_count > 0:
        reasons.append(f"日志检测到 {fallback_count} 次 CPU fallback")
    if postflight_status(summary) == "failed":
        reasons.append("宿主 postflight 失败，运行后健康状态未闭环")
    monitoring_status = summary.get("monitoring_status")
    if monitoring_status in ("partial", "failed"):
        reasons.append(f"同窗监控为 {status_text(monitoring_status)}")
        monitoring = summary.get("monitoring")
        if isinstance(monitoring, dict):
            monitor_reasons = monitoring.get("reasons")
            if isinstance(monitor_reasons, list):
                reasons.extend(f"监控：{item}" for item in monitor_reasons)
    if summary.get("failure_stage"):
        reasons.append(f"失败阶段：{summary['failure_stage']}")
    if summary.get("error"):
        reasons.append(f"错误：{summary['error']}")
    if not finite_metrics(result):
        reasons.append("没有可展示的有限数值指标")
    return list(dict.fromkeys(reasons))


def evidence_records(
    root: Path, summary: dict[str, Any], result: dict[str, Any],
) -> list[tuple[str, str, int | None, str | None]]:
    candidates: list[tuple[str, Any]] = [
        ("实验外层事实（生成报告后会更新 report_generation）", "summary.json"),
        ("Benchmark 语义结果", "benchmark-result.json"),
        ("最终执行计划", "resolved-plan.json"),
        ("宿主 runner 合并日志", "runner.log"),
        ("Benchmark 原始日志", result.get("benchmark_log")),
        ("宿主 preflight", (summary.get("host_preflight") or {}).get("path") if isinstance(summary.get("host_preflight"), dict) else None),
        ("宿主 postflight", (summary.get("host_postflight") or {}).get("path") if isinstance(summary.get("host_postflight"), dict) else None),
        ("Benchmark 监控 summary", "benchmark-monitor/summary.json"),
        ("Benchmark 解析监控样本", "benchmark-monitor/samples.jsonl"),
        ("Benchmark 原始监控命令", "benchmark-monitor/samples.raw.jsonl"),
    ]
    for label, record, artifact in config_artifacts(root, summary):
        candidates.append((f"Case 配置快照：{label}", record.get("artifact_path")))
    records = []
    seen: set[str] = set()
    for label, value in candidates:
        path = result_relative_file(root, value)
        if path is None:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        if relative == "summary.json":
            records.append((label, relative, None, None))
        else:
            records.append((label, relative, path.stat().st_size, sha256(path)))
    return records


def render_monitor_report(
    root: Path,
    summary: dict[str, Any],
    monitor: dict[str, Any],
    samples: list[dict[str, Any]],
    monitor_asset: dict[str, Any] | None,
) -> str:
    run_id = summary.get("run_id") or root.name
    case_name = summary.get("case", "unknown")
    status = monitor.get("status", summary.get("monitoring_status", "not_available"))
    policy = monitor.get("policy") if isinstance(monitor.get("policy"), dict) else {}
    windows = monitor_windows(monitor)
    visible_samples, projected_count, total_count = visible_monitor_samples(
        samples, windows
    )
    sample_projection_note = (
        f"报告展示 {len(visible_samples)}/{projected_count} 个"
        + ("测量窗重叠样本" if projected_count != total_count else "已保存样本")
        + "；如超过每目标 64 个则做确定性等距抽样。所有 "
        + f"{total_count} 个样本仍完整保存在下方 JSONL 证据中。"
        if total_count
        else "本次结果没有保存监控样本；请结合监控状态和证据缺口判断是显式关闭、早期失败还是旧结果。"
    )
    reasons = monitor.get("reasons") if isinstance(monitor.get("reasons"), list) else []
    if status == "passed":
        conclusion = (
            "所选设备均获得满足证据策略的有效样本，且每个 rank 的精确测量窗口完整可校验。"
        )
    elif status == "not-run":
        conclusion = "本次运行显式关闭监控；Benchmark 测量状态不受影响。"
    elif status == "partial":
        conclusion = "已保存可用监控证据，但时间窗、设备覆盖或样本数量至少一项不完整。"
    elif status == "failed":
        conclusion = "监控器未能形成完整证据；Benchmark 性能测量仍按独立状态解释。"
    else:
        conclusion = "该结果没有可用的 Benchmark 同窗监控证据。"

    lines = [
        f"# Ascend Base Benchmark 监控报告：{run_id}",
        "",
        f"> **结论：** {conclusion}",
        ">",
        "> **解释边界：** 监控通过只表示采样范围与窗口覆盖完整；这些整数百分比不等于理论峰值利用率，也不是算子或 kernel profiler。",
        "",
        "[返回 Benchmark 主报告](report.md)",
        "",
        "## 1. 状态与采集策略",
        "",
        "| 项目 | 本次事实 |",
        "|---|---|",
        f"| Case | {code(case_name)} |",
        f"| 监控状态 | {status_text(status)} |",
        f"| Collector | {code(monitor.get('collector', policy.get('collector', '未记录')))} |",
        f"| 目标采样间隔 | {format_duration(monitor.get('target_interval_s', policy.get('target_interval_s')))} |",
        f"| 命令超时 | {format_duration(monitor.get('command_timeout_s', policy.get('command_timeout_s')))} |",
        f"| 每目标最低有效样本 | {code(monitor.get('required_samples_per_target', monitor.get('required_samples_per_chip', policy.get('required_samples_per_target', '未记录'))))} |",
        f"| 覆盖窗口 | {md_escape(policy.get('coverage_window', 'rank-local exact measurement window'))} |",
        f"| 自动追加 workload | {code(monitor.get('automatic_workload_extension', False))} |",
        f"| 时钟域 | {md_escape(monitor.get('clock_domain', '未记录'))} |",
        "",
    ]
    if reasons:
        lines.extend([
            "### 证据缺口",
            "",
            *[f"- {md_escape(reason)}" for reason in reasons],
            "",
        ])

    lines.extend([
        "## 2. Rank、设备与精确时间窗",
        "",
        "| Rank | Local Rank | 物理 NPU | Chip | 逻辑 Device |",
        "|---:|---:|---:|---:|---:|",
    ])
    rank_map = monitor.get("rank_device_map")
    if isinstance(rank_map, list) and rank_map:
        for item in rank_map:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {md_escape(item.get('rank', '—'))} | "
                f"{md_escape(item.get('local_rank', '—'))} | "
                f"{md_escape(item.get('npu_id', '—'))} | "
                f"{md_escape(item.get('chip_id', '—'))} | "
                f"{md_escape(item.get('logic_id', '—'))} |"
            )
    else:
        lines.append("| — | — | — | — | 未记录精确 rank/device 映射 |")

    lines.extend([
        "",
        "| 窗口 | Rank / Device | UTC 开始 → 结束 | Monitor offset (s) |",
        "|---|---|---|---|",
    ])
    all_windows = []
    lifecycle = monitor.get("lifecycle_windows")
    if isinstance(lifecycle, list):
        all_windows.extend(item for item in lifecycle if isinstance(item, dict))
    all_windows.extend(windows)
    if all_windows:
        for window in sorted(
            all_windows,
            key=lambda item: float(item.get("started_offset_s", math.inf)),
        ):
            identity = (
                f"rank={window.get('rank')} / Device={window.get('logical_device_id')}"
                if window.get("role") == "measurement" else "全进程"
            )
            lines.append(
                f"| {code(window.get('role', 'unknown'))} | {code(identity)} | "
                f"{code(window.get('started_at', '—'))} → {code(window.get('finished_at', '—'))} | "
                f"{format_number(window.get('started_offset_s'), 6)} → "
                f"{format_number(window.get('finished_offset_s'), 6)} |"
            )
    else:
        lines.append("| — | — | 未记录 | — |")

    counts = monitor.get("primary_sample_counts_by_target")
    lines.extend([
        "",
        "### 精确测量窗覆盖",
        "",
        "| NPU/Chip/Device | 有效重叠样本 | 要求 |",
        "|---|---:|---:|",
    ])
    if isinstance(counts, dict) and counts:
        required = monitor.get(
            "required_samples_per_target",
            monitor.get("required_samples_per_chip", "—"),
        )
        for target, count in sorted(counts.items()):
            lines.append(f"| {code(target)} | {md_escape(count)} | {md_escape(required)} |")
    else:
        lines.append("| — | 0 | 未记录 |")

    lines.extend([
        "",
        "## 3. 测量窗内原始样本统计",
        "",
        "下表只统计有效且与其自身逻辑 Device 精确测量窗口重叠的样本。统计值是本次离散采样的阅读辅助，不是厂商阈值。",
        "",
        "| NPU | Chip | Device | 指标 | 样本数 | 最小 | 中位数 | 均值 | 最大 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    stats = monitor_statistics(samples, windows)
    if stats:
        for npu_id, chip_id, logic_id, label, count, minimum, median, mean, maximum in stats:
            lines.append(
                f"| {npu_id} | {chip_id} | {logic_id} | {md_escape(label)} (%) | "
                f"{count} | {format_number(minimum)} | {format_number(median)} | "
                f"{format_number(mean)} | {format_number(maximum)} |"
            )
    else:
        lines.append("| — | — | — | 无有效测量窗样本 | 0 | — | — | — | — |")
    if monitor_asset is not None:
        lines.extend([
            "",
            "### 可视化",
            "",
            f"![Benchmark NPU 同窗监控]({monitor_asset['path']})",
        ])

    lines.extend([
        "",
        "## 4. 逐样本原值",
        "",
        sample_projection_note,
        "",
        "| 样本 | NPU/Chip/Device | UTC 区间 | Offset (s) | 窗口 | AICore | AIVector | HBM 占用 | HBM 带宽 | NPU | 有效性 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    if visible_samples:
        for sample in visible_samples:
            target = monitor_target(sample)
            target_text = (
                f"{target[0]}/{target[1]}/{target[2]}" if target else "?/?/?"
            )
            values = [
                monitor_source_value(sample, normalized, source)
                for normalized, _label, source in MONITOR_FIELDS
            ]
            missing = sample.get("missing_or_invalid_fields")
            validity = "有效" if sample.get("valid") is True else (
                "无效：" + "、".join(map(str, missing))
                if isinstance(missing, list) and missing else "无效"
            )
            lines.append(
                f"| {code('S' + str(sample.get('sequence', '?')).zfill(2))} | "
                f"{code(target_text)} | {code(sample.get('sample_started_at', '—'))} → "
                f"{code(sample.get('sample_finished_at', '—'))} | "
                f"{format_number(sample.get('started_offset_s'), 6)} → "
                f"{format_number(sample.get('finished_offset_s'), 6)} | "
                f"{'测量窗' if sample_overlaps_measurement(sample, windows) else '窗口外'} | "
                + " | ".join(md_escape(value) for value in values)
                + f" | {md_escape(validity)} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — | — | — | — | 没有样本 |")

    lines.extend([
        "",
        "## 5. 原始证据",
        "",
        "| 证据 | 路径 | 字节数 | SHA-256 |",
        "|---|---|---:|---|",
    ])
    evidence: list[tuple[str, Any]] = [
        (
            "监控 summary",
            monitor.get("summary") or (
                artifact_ref(root / "benchmark-monitor" / "summary.json", root)
                if (root / "benchmark-monitor" / "summary.json").is_file()
                else None
            ),
        ),
        ("解析样本 JSONL", monitor.get("parsed_samples")),
        ("原始命令 JSONL", monitor.get("raw_samples")),
    ]
    event_artifacts = monitor.get("event_artifacts")
    if isinstance(event_artifacts, dict):
        evidence.extend([
            ("容器窗口", event_artifacts.get("container")),
            ("torchrun 窗口", event_artifacts.get("torchrun")),
        ])
        measurements = event_artifacts.get("measurements")
        if isinstance(measurements, list):
            evidence.extend(("Rank 测量窗口", item) for item in measurements)
    emitted = False
    for label, reference in evidence:
        if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
            continue
        path = result_relative_file(root, reference["path"])
        if path is None:
            continue
        emitted = True
        relative = path.relative_to(root).as_posix()
        lines.append(
            f"| {label} | {markdown_link(relative, relative)} | "
            f"{reference.get('bytes', path.stat().st_size)} | "
            f"{code(reference.get('sha256', sha256(path)))} |"
        )
    if not emitted:
        lines.append("| — | 未保存监控证据 | — | — |")

    lines.extend([
        "",
        "## 6. 解释限制",
        "",
        "- `npu-smi` 返回的是驱动统计窗口内的整数百分比；其内部积分窗口和理论分母不由本报告猜测。",
        "- AICore 高只说明统计窗口内持续活动，不证明每个 active cycle 的指令效率；AIVector 低也不能单独判定 GEMM 异常。",
        "- HBM 占用率与 HBM 带宽利用率是不同物理量，报告不会混合二者。",
        "- 监控本身可能带来扰动；正式基线发布前仍需在同一配置和选卡下做 monitor off/on 重复 A/B。",
        "- 本报告没有 HCCS、功耗、温度、CPU/RAM 或 operator/kernel timeline；需要深度归因时应使用独立 profiling 流程。",
        "",
        "---",
        "",
        f"Benchmark 监控报告生成器 schema：{code(REPORT_SCHEMA_VERSION)}。原始 JSON/JSONL 与事件文件是事实源。",
        "",
    ])
    return "\n".join(lines)


def render_report(
    root: Path,
    summary: dict[str, Any],
    result: dict[str, Any],
    assets: list[dict[str, Any]],
) -> str:
    run_id = summary.get("run_id") or root.name
    case_name = summary.get("case") or result.get("case") or "unknown"
    status = summary.get("status")
    metrics = finite_metrics(result)
    groups = metric_groups(metrics)
    expected = result.get("expected_ranks") if isinstance(result.get("expected_ranks"), list) else []
    observed = result.get("observed_ranks") if isinstance(result.get("observed_ranks"), list) else []
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    runtime = summary.get("runtime") if isinstance(summary.get("runtime"), dict) else {}
    resolved_plan = summary.get("resolved_plan") if isinstance(summary.get("resolved_plan"), dict) else {}
    permissions = resolved_plan.get("permissions") if isinstance(resolved_plan.get("permissions"), dict) else {}
    preflight = host_result(summary, "host_preflight")
    reasons = conclusion_reasons(summary, result)

    complete_semantic_evidence = bool(metrics) and (
        not expected or observed == expected
    ) and result.get("fallback_count", 0) == 0
    if status == "skipped":
        conclusion = (
            f"本次 `{case_name}` 在启动前跳过：{summary.get('skip_reason', 'Case 不适用')}。"
            "未启动容器、访问设备或产生性能测量；此状态不代表测试通过。"
        )
    elif status == "passed" and complete_semantic_evidence:
        conclusion = (
            f"本次 `{case_name}` 的容器执行和语义结果解析均通过，"
            f"已获得 {len(observed)}/{len(expected) if expected else len(observed)} 个预期 rank 的结果。"
        )
    elif status == "passed":
        conclusion = (
            f"外层状态把本次 `{case_name}` 标记为通过，但当前结果证据不完整；"
            "应先核对缺失 artifact、Rank 或 fallback 字段，不能直接采用性能结论。"
        )
    elif status == "partial":
        conclusion = (
            f"本次 `{case_name}` 仅部分完成；已获得 "
            f"{len(observed)}/{len(expected) if expected else len(observed)} 个预期 rank 的结果，"
            "并保存可用测量及缺口证据，不能按完整通过结果使用。"
        )
    else:
        conclusion = f"本次 `{case_name}` 未形成可接受的完整结果；应先处理失败原因，再讨论性能数值。"

    lines = [
        f"# Ascend Base Benchmark 报告：{run_id}",
        "",
        f"> **结论：** {conclusion}",
        ">",
        "> **解释边界：** 本报告只描述本次运行；它不自动代表稳定性能基线、理论峰值利用率或跨机器可比结果。",
        "",
        "## 1. 状态总览",
        "",
        "| 层次 | 状态 | 人类含义 |",
        "|---|---|---|",
        f"| 实验总状态 | {status_text(status)} | 由执行、测量、请求的监控证据与 postflight 共同决定 |",
        f"| 容器/worker 执行 | {status_text(summary.get('execution_status'))} | 是否正常结束；不能替代语义结果检查 |",
        f"| Benchmark 测量 | {status_text(summary.get('measurement_status'))} | 是否解析到所有预期 rank 的 FlagPerf 结果且无 CPU fallback |",
        f"| Benchmark 同窗监控 | {status_text(summary.get('monitoring_status', 'not_available'))} | 是否覆盖所有目标 Device 的精确 rank 测量窗口；不判定理论利用率 |",
        f"| 宿主 postflight | {status_text(postflight_status(summary))} | 运行后设备枚举/占用检查是否闭环 |",
        f"| 报告生成 | 通过 ({code('passed')}) | 本 Markdown 与所列 SVG 已生成；不反写实验总状态 |",
        "",
    ]
    if reasons:
        lines.extend([
            "### 需要关注的原因",
            "",
            *[f"- {md_escape(reason)}" for reason in reasons],
            "",
        ])

    lines.extend([
        "监控采样、精确窗口、逐 Device 原值和解释边界见 "
        "[report_monitor.md](report_monitor.md)。",
        "",
    ])

    lines.extend([
        "## 2. 测试范围与运行身份",
        "",
        "| 项目 | 本次事实 |",
        "|---|---|",
        f"| Case | {code(case_name)} |",
        f"| Run ID | {code(run_id)} |",
        f"| 宿主 | {code(preflight.get('host', '未记录'))} |",
        f"| 选择来源 | {code(selection.get('source', '未记录'))} |",
        f"| 物理 NPU | {code(sequence_text(selection.get('selected_npu_ids')))} |",
        f"| 逻辑 Device | {code(sequence_text(selection.get('selected_device_ids')))} |",
        f"| 预期 / 已观测 Rank | {code(sequence_text(expected))} / {code(sequence_text(observed))} |",
        f"| 缺失 Rank | {code(sequence_text(result.get('missing_ranks', [])))} |",
        f"| 容器返回码 | {code(summary.get('container_returncode', result.get('container_returncode', '未记录')))} |",
        f"| Timeout | {code(summary.get('timed_out', result.get('timed_out', '未记录')))} |",
        f"| 开始 / 结束 | {code(summary.get('started_at', '未记录'))} / {code(summary.get('finished_at', '未记录'))} |",
        f"| 外层墙钟时长 | {format_duration(summary.get('wall_clock_duration_s'))} |",
        f"| CPU fallback 次数 | {code(result.get('fallback_count', '未记录'))} |",
        "",
        "## 3. 性能指标",
        "",
        "以下为 `benchmark-result.json` 保存的语义结果，不用图表像素替代原值。",
        "",
        "| Rank | 指标 | 原值 | 单位 | 原始结果行 |",
        "|---:|---|---:|---|---|",
    ])
    if metrics:
        for metric in metrics:
            lines.append(
                f"| {md_escape(metric.get('rank', '—'))} | {md_escape(metric.get('metric', '—'))} | "
                f"{format_number(metric.get('value'))} | {md_escape(metric.get('unit', '—'))} | "
                f"{code(metric.get('raw', '未记录'))} |"
            )
    else:
        lines.append("| — | 未解析到有限数值指标 | — | — | — |")

    lines.extend([
        "",
        "### 同次运行 Rank 统计",
        "",
        "| 指标 | 结果行 / 唯一 Rank | 最小值 | 最大值 | 均值 | 总体标准差 | Rank 差异率 | 最小/最大平衡度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    if groups:
        for (metric_name, unit), values in groups.items():
            stats = group_statistics(values)
            lines.append(
                f"| {md_escape(metric_name)} ({md_escape(unit)}) | "
                f"{len(values)} / {len({item.get('rank') for item in values})} | "
                f"{format_number(stats['minimum'])} | {format_number(stats['maximum'])} | "
                f"{format_number(stats['mean'])} | {format_number(stats['population_stddev'])} | "
                f"{format_number(stats['spread_percent']) + '%' if stats['spread_percent'] is not None else '—'} | "
                f"{format_number(stats['balance_percent']) + '%' if stats['balance_percent'] is not None else '—'} |"
            )
    else:
        lines.append("| — | 0 | — | — | — | — | — | — |")
    lines.extend([
        "",
        "`Rank 差异率 = (max - min) / |mean| × 100%`；`平衡度 = min / max × 100%`。"
        "二者是本报告的同次运行派生统计，不是厂商阈值，也不能替代多次独立运行的方差分析。",
        "",
    ])
    if assets:
        lines.extend(["### 可视化", ""])
        for asset in assets:
            lines.append(f"![{md_escape(case_name)} Rank 分布]({asset['path']})")
            lines.append("")

    lines.extend([
        "## 4. Case 配置与可复现性",
        "",
        "配置优先级为 `generic < ascend < override`。当前 BenchmarkExecutor 会把各层 YAML 快照写入结果目录；"
        "旧结果若只有 path/hash 而没有快照，本报告不会从已变化的工作区文件猜测当时参数。",
        "",
        "| 层 | 原始来源 | 结果内快照 | 字节数 | SHA-256 |",
        "|---|---|---|---:|---|",
    ])
    configs = config_artifacts(root, summary)
    if configs:
        for label, record, artifact in configs:
            artifact_link = (
                markdown_link(artifact.relative_to(root).as_posix(), artifact.relative_to(root).as_posix())
                if artifact is not None else "未保存（旧结果）"
            )
            lines.append(
                f"| {code(label)} | {code(record.get('path', '未记录'))} | {artifact_link} | "
                f"{md_escape(record.get('bytes', '—'))} | {code(record.get('sha256', '未记录'))} |"
            )
    else:
        lines.append("| — | 未记录 Case 配置来源 | — | — | — |")

    resolved_rows = resolved_config_rows(root, summary)
    if resolved_rows:
        lines.extend([
            "",
            "### 顶层标量的最终覆盖结果",
            "",
            "| 参数 | 最终值 | 最后写入层 |",
            "|---|---|---|",
            *[
                f"| {code(key)} | {code(value)} | {code(source)} |"
                for key, value, source in resolved_rows
            ],
        ])

    lock = runtime.get("lock") if isinstance(runtime.get("lock"), dict) else {}
    image_manifest = lock.get("image_manifest") if isinstance(lock.get("image_manifest"), dict) else {}
    stack_lock = lock.get("stack_lock") if isinstance(lock.get("stack_lock"), dict) else {}
    lease = summary.get("device_lease") if isinstance(summary.get("device_lease"), dict) else {}
    lines.extend([
        "",
        "## 5. 运行时、权限与资源生命周期",
        "",
        "| 项目 | 本次事实 |",
        "|---|---|",
        f"| 镜像 | {code(runtime.get('image', resolved_plan.get('image', '未记录')))} |",
        f"| 镜像 ID | {code(runtime.get('image_id', '未记录'))} |",
        f"| 运行时验证状态 | {status_text(image_manifest.get('validation_status'))} |",
        f"| 运行时已验证范围 | {md_escape(image_manifest.get('validation_scope', '未记录'))} |",
        f"| image manifest SHA-256 | {code(image_manifest.get('sha256', '未记录'))} |",
        f"| stack lock SHA-256 | {code(stack_lock.get('sha256', '未记录'))} |",
        f"| Privileged root | {code(permissions.get('privileged_root', '未记录'))} |",
        f"| Network / IPC / PID | {code(permissions.get('network', '未记录'))} / {code(permissions.get('ipc_namespace', '未记录'))} / {code(permissions.get('pid_namespace', '未记录'))} |",
        f"| Device lease | backend={code(lease.get('backend', '未记录'))}, devices={code(sequence_text(lease.get('device_ids')))} |",
        f"| Lease 获取 / 释放 | {code(lease.get('acquired_at', '未记录'))} / {code(lease.get('released_at', '未记录'))} |",
        "",
        "## 6. Preflight / Postflight 健康闭环",
        "",
        "| 检查 | 状态 | 选卡/占用范围 | 证据 |",
        "|---|---|---|---|",
    ])
    for key, label in (("host_preflight", "运行前"), ("host_postflight", "运行后")):
        wrapper = summary.get(key) if isinstance(summary.get(key), dict) else {}
        result_record = host_result(summary, key)
        status_value = wrapper.get("status") or result_record.get("status") or (
            "passed" if key == "host_preflight" and result_record else "not_available"
        )
        occupancy = result_record.get("occupancy") if isinstance(result_record.get("occupancy"), dict) else {}
        scope = occupancy.get("checked_device_ids") or selection.get("selected_device_ids") or "未记录"
        artifact = result_relative_file(root, wrapper.get("path"))
        link = (
            markdown_link(artifact.relative_to(root).as_posix(), artifact.relative_to(root).as_posix())
            if artifact is not None else "未保存"
        )
        lines.append(
            f"| {label} | {status_text(status_value)} | {code(sequence_text(scope))} | {link} |"
        )

    lines.extend([
        "",
        "## 7. 证据索引",
        "",
        "| 证据 | 相对路径 | 字节数 | SHA-256 |",
        "|---|---|---:|---|",
    ])
    for label, relative, byte_count, digest in evidence_records(root, summary, result):
        lines.append(
            f"| {md_escape(label)} | {markdown_link(relative, relative)} | "
            f"{byte_count if byte_count is not None else '动态'} | "
            f"{code(digest) if digest is not None else '见 canonical summary hash'} |"
        )

    lines.extend([
        "",
        f"Canonical summary SHA-256（排除会被报告生成更新的 `report_generation`）："
        f"{code(canonical_summary_sha256(summary))}",
        "",
        "## 8. 解释限制与下一步",
        "",
        "- `passed` 只表示本次执行、结果解析、fallback 门禁和当前 postflight 形成闭环；不等于稳定性能基线。",
        "- Rank 统计是同一进程组内的分布，不能替代多次独立进程、不同时间窗口和置信区间。",
        "- 本报告不把 TFLOPS、TOPS、GB/s、GB 等不同物理量合并，也不从 Toolkit/DMI 结果推导利用率。",
        "- `npu-smi` 同窗采样是低频设备遥测，不是 operator/kernel profiler；报告不会从整数百分比推导理论利用率。",
        "- 当前监控不包含 HCCS、功耗、温度或宿主 CPU/RAM；这些字段缺失时报告会明确留空，而不是推断。",
        "- 若要发布正式结论，还需固定 branch/commit、CPU/NUMA、驱动/固件及重复运行方案，并保留全部原始证据。",
        "",
        "---",
        "",
        f"Benchmark 报告生成器 schema：{code(REPORT_SCHEMA_VERSION)}。报告是 JSON、配置快照和日志证据的确定性阅读视图；发生冲突时以原始证据及 SHA-256 为准。",
        "",
    ])
    return "\n".join(lines)


def generate_benchmark_report(result_dir: Path) -> dict[str, Any]:
    root = result_dir.expanduser().resolve()
    summary = read_json(root / "summary.json")
    assert summary is not None
    summary_version = schema_version(
        summary, "Benchmark summary", SUPPORTED_SUMMARY_SCHEMA_VERSIONS,
    )
    if summary.get("kind") not in (None, "benchmark"):
        raise ReportError(f"result is not a Benchmark run: kind={summary.get('kind')!r}")
    result = read_json(root / "benchmark-result.json", required=False) or {}
    result_version = None
    if result:
        result_version = schema_version(
            result, "Benchmark result", SUPPORTED_RESULT_SCHEMA_VERSIONS,
        )

    monitor_path = root / "benchmark-monitor" / "summary.json"
    monitor = read_json(monitor_path, required=False)
    monitor_version = None
    if monitor is not None:
        monitor_version = schema_version(
            monitor, "Benchmark monitor", SUPPORTED_MONITOR_SCHEMA_VERSIONS,
        )
    else:
        embedded_monitor = summary.get("monitoring")
        monitor = embedded_monitor if isinstance(embedded_monitor, dict) else {}
    parsed_reference = monitor.get("parsed_samples") if isinstance(
        monitor.get("parsed_samples"), dict
    ) else {}
    parsed_path = result_relative_file(root, parsed_reference.get("path"))
    samples = read_jsonl(parsed_path) if parsed_path is not None else []

    rank_assets: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for (metric_name, unit), metrics in metric_groups(finite_metrics(result)).items():
        base_name = f"benchmark-rank-{safe_slug(metric_name)}-{safe_slug(unit)}"
        name = base_name + ".svg"
        suffix = 2
        while name in used_names:
            name = f"{base_name}-{suffix}.svg"
            suffix += 1
        used_names.add(name)
        reference = write_asset(root, name, rank_metric_svg(metric_name, unit, metrics))
        if reference:
            rank_assets.append(reference)

    monitor_asset = write_asset(
        root,
        "benchmark-monitor-usage.svg",
        monitor_usage_svg(
            visible_monitor_samples(samples, monitor_windows(monitor))[0],
            monitor_windows(monitor),
        ),
    )
    assets = [*rank_assets, *([monitor_asset] if monitor_asset is not None else [])]

    report_path = root / "report.md"
    monitor_report_path = root / "report_monitor.md"
    atomic_write(report_path, render_report(root, summary, result, rank_assets))
    atomic_write(
        monitor_report_path,
        render_monitor_report(root, summary, monitor, samples, monitor_asset),
    )
    result_path = root / "benchmark-result.json"
    metadata = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "path": report_path.relative_to(root).as_posix(),
        "bytes": report_path.stat().st_size,
        "sha256": sha256(report_path),
        "monitor_path": monitor_report_path.relative_to(root).as_posix(),
        "monitor_bytes": monitor_report_path.stat().st_size,
        "monitor_sha256": sha256(monitor_report_path),
        "outputs": {
            "main": artifact_ref(report_path, root),
            "monitor": artifact_ref(monitor_report_path, root),
        },
        "assets": assets,
        "input_schemas": {
            "summary": summary_version,
            "benchmark_result": result_version,
            "benchmark_monitor": monitor_version,
        },
        "input_facts": {
            "summary_canonical_sha256": canonical_summary_sha256(summary),
            "benchmark_result": (
                artifact_ref(result_path, root) if result_path.is_file() else None
            ),
            "benchmark_monitor": (
                artifact_ref(monitor_path, root) if monitor_path.is_file() else None
            ),
        },
    }
    return metadata


def generate_and_record(result_dir: Path) -> dict[str, Any]:
    root = result_dir.expanduser().resolve()
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    assert summary is not None
    original_status = summary.get("status")
    try:
        metadata = generate_benchmark_report(root)
    except Exception as exc:
        summary["report_generation"] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        summary["status"] = original_status
        write_json(summary_path, summary)
        raise
    summary = read_json(summary_path)
    assert summary is not None
    summary["report_generation"] = metadata
    summary["status"] = original_status
    write_json(summary_path, summary)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = generate_and_record(args.result_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
