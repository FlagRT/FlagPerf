#!/usr/bin/env python3
"""Generate a deterministic, human-readable report for an Ascend toolkit run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable


REPORT_SCHEMA_VERSION = 6
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = (1, 2, 3)
ERROR_GROUP_LIMIT = 6
ERROR_CONTEXT_EXAMPLE_LIMIT = 3
ERROR_INLINE_BUDGET = 900
CASE_ORDER = (
    "computation-BF16",
    "computation-FP16",
    "computation-FP32",
    "computation-INT8",
    "main_memory-bandwidth",
    "main_memory-capacity",
    "interconnect-h2d",
    "interconnect-d2h",
    "interconnect-h2d-latency",
    "interconnect-d2h-latency",
    "interconnect-P2P_intraserver",
    "interconnect-P2P_intraserver-latency",
    "interconnect-MPI_intraserver",
)
CASE_LABELS = {
    "computation-BF16": "BF16 算力",
    "computation-FP16": "FP16 算力",
    "computation-FP32": "FP32 算力",
    "computation-INT8": "INT8 算力",
    "main_memory-bandwidth": "D2D 设备内存带宽",
    "main_memory-capacity": "HBM 容量",
    "interconnect-h2d": "H2D 带宽",
    "interconnect-d2h": "D2H 带宽",
    "interconnect-h2d-latency": "H2D 时延",
    "interconnect-d2h-latency": "D2H 时延",
    "interconnect-P2P_intraserver": "单机 P2P 带宽",
    "interconnect-P2P_intraserver-latency": "单机 P2P 时延",
    "interconnect-MPI_intraserver": "单机 MPI/HCCL AllReduce",
}
STATUS_TEXT = {
    "passed": "通过",
    "partial": "部分完成",
    "failed": "失败",
    "unsupported": "不支持",
    "running": "运行中",
    "missing": "缺失",
    "unknown": "未知",
    "complete": "完整",
    "idle": "空闲",
    "not-run": "未执行",
}


class ReportError(RuntimeError):
    """The result directory cannot be rendered safely."""


def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ReportError(f"required JSON artifact is missing: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"JSON artifact must contain an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReportError(f"cannot read JSONL artifact {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ReportError(f"JSONL artifact must contain objects: {path}:{number}")
        values.append(value)
    return values


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


def artifact_ref(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def finite_values(metrics: Iterable[dict[str, Any]]) -> list[float]:
    values = []
    for metric in metrics:
        value = metric.get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None or not isinstance(value, (int, float)):
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    rendered = f"{number:,.{digits}f}"
    if digits:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered if rendered != "-0" else "0"


def raw_metric_value(metric: dict[str, Any]) -> str:
    """Render a vendor metric without rounding or deriving a replacement value."""
    raw = metric.get("value_raw")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    value = metric.get("value")
    return str(value) if isinstance(value, (int, float)) else "—"


def format_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or not math.isfinite(float(seconds)):
        return "—"
    value = float(seconds)
    if value < 60:
        return f"{value:.2f} s"
    minutes, remainder = divmod(value, 60)
    if minutes < 60:
        return f"{int(minutes)} min {remainder:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min"


def human_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")


def md_escape(value: Any) -> str:
    text = str(value if value is not None else "—")
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def status_text(value: Any) -> str:
    key = str(value or "missing").lower()
    return STATUS_TEXT.get(key, key)


def _error_group(message: str) -> tuple[str, str | None]:
    """Return a stable error signature and its varying execution context."""
    match = re.search(r"(DMI error\s+[^:;]+:\s+.*)$", message, flags=re.IGNORECASE)
    if not match:
        return message, None
    context = message[:match.start()].rstrip(" :")
    return match.group(1), context or None


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rstrip() + "…"


def human_error(value: Any, *, evidence_path: str | None = None) -> str:
    """Render actionable error groups without copying every raw failure inline."""
    text = " ".join(str(value or "").split())
    if text.startswith("vendor diagnosis ") and " is unsupported or lacks a threshold" in text:
        item = text.split()[2]
        return f"厂商 {item} 诊断不支持或缺少当前设备阈值"
    if not text:
        return "—"

    messages = [item.strip() for item in text.split(";") if item.strip()]
    groups: dict[str, dict[str, Any]] = {}
    for message in messages:
        signature, context = _error_group(message)
        group = groups.setdefault(signature, {"count": 0, "contexts": []})
        group["count"] += 1
        if context and context not in group["contexts"]:
            group["contexts"].append(context)

    rendered: list[str] = []
    included_messages = 0
    for signature, group in list(groups.items())[:ERROR_GROUP_LIMIT]:
        count = int(group["count"])
        candidate = _shorten(signature, 420)
        if count > 1:
            candidate += f"（重复 {count} 次"
            examples = group["contexts"][:ERROR_CONTEXT_EXAMPLE_LIMIT]
            if examples:
                candidate += "；示例范围：" + "、".join(
                    _shorten(str(item), 100) for item in examples
                )
            candidate += "）"
        separator_size = 1 if rendered else 0
        if rendered and sum(len(item) for item in rendered) + separator_size + len(candidate) > ERROR_INLINE_BUDGET:
            break
        if not rendered and len(candidate) > ERROR_INLINE_BUDGET:
            candidate = _shorten(candidate, ERROR_INLINE_BUDGET)
        rendered.append(candidate)
        included_messages += count

    omitted = len(messages) - included_messages
    if omitted > 0:
        rendered.append(f"另有 {omitted} 条错误未内联")
    summary = "；".join(rendered)
    if evidence_path and (len(messages) > 1 or omitted > 0 or summary != text):
        summary += "；" + markdown_link("完整错误证据", evidence_path)
    return summary


def markdown_link(label: str, path: str) -> str:
    safe_label = label.replace("[", "\\[").replace("]", "\\]")
    safe_path = path.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{safe_label}]({safe_path})"


def case_sort_key(name: str) -> tuple[int, str]:
    try:
        return CASE_ORDER.index(name), name
    except ValueError:
        return len(CASE_ORDER), name


def command_records(case: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    command = case.get("command")
    if isinstance(command, dict):
        records.append(command)
    commands = case.get("commands")
    if isinstance(commands, list):
        records.extend(item for item in commands if isinstance(item, dict))
    return records


def inferred_measurement(case: dict[str, Any]) -> tuple[str, str | None, bool]:
    explicit = case.get("measurement_status")
    if explicit in ("passed", "partial", "failed"):
        return str(explicit), case.get("measurement_error"), False

    records = command_records(case)
    commands_ok = bool(records) and all(
        record.get("returncode") == 0 and not record.get("timed_out", False)
        for record in records
    )
    has_metrics = bool(finite_values(case.get("metrics", [])))
    diagnosis = case.get("diagnosis_status")
    final = case.get("status")
    error = case.get("error")
    if commands_ok and has_metrics and diagnosis in ("passed", "unsupported"):
        return "passed", None, True
    if final == "partial" and has_metrics:
        return "partial", str(error) if error else None, True
    if final == "passed" and has_metrics:
        return "passed", None, True
    return "failed", str(error) if error else "measurement evidence is incomplete", True


def metric_unit(case: dict[str, Any]) -> str:
    units = {str(item.get("unit")) for item in case.get("metrics", []) if item.get("unit")}
    return ", ".join(sorted(units)) if units else "—"


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None,
                "max": None, "stdev": None, "cv_pct": None}
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "stdev": stdev,
        "cv_pct": stdev / mean * 100 if mean else None,
    }


def case_metric_summary(name: str, case: dict[str, Any]) -> str:
    metrics = case.get("metrics", [])
    values = finite_values(metrics)
    unit = metric_unit(case)
    if not values:
        return "无有效指标"
    if name.startswith("computation-") or name in ("interconnect-h2d", "interconnect-d2h"):
        if len(values) == 1:
            if name.startswith("computation-"):
                return f"{raw_metric_value(metrics[0])} {unit}"
            return f"{format_number(values[0], 6)} {unit}"
    stats = numeric_summary(values)
    return (
        f"n={stats['count']}，{format_number(stats['min'])}–"
        f"{format_number(stats['max'])} {unit}，"
        f"中位数 {format_number(stats['median'])}"
    )


def case_scope(name: str, case: dict[str, Any]) -> str:
    metrics = case.get("metrics", [])
    if name.startswith("computation-"):
        devices = {str(item.get("device")) for item in metrics if item.get("device") is not None}
        return (
            "DMI 全设备聚合" if {item.lower() for item in devices} == {"all"}
            else f"{len(devices)} 个厂商执行目标"
        )
    if name == "main_memory-bandwidth":
        devices = {str(item.get("device")) for item in metrics if item.get("device") is not None}
        return f"{len(devices)} 个逻辑 Device"
    if name == "main_memory-capacity":
        targets = {(item.get("card"), item.get("chip")) for item in metrics}
        scopes = {str(item.get("scope", "chip")) for item in metrics}
        scope = "chip" if scopes == {"chip"} else "/".join(sorted(scopes))
        return f"{len(targets)} 个 {scope}，逐 {scope} 原值，未聚合"
    if name in ("interconnect-h2d", "interconnect-d2h"):
        devices = {str(item.get("device")) for item in metrics if item.get("device") is not None}
        return "DMI 全设备聚合" if any(item.lower() == "all" for item in devices) else f"{len(devices)} 个 Device"
    if name in ("interconnect-h2d-latency", "interconnect-d2h-latency"):
        devices = {str(item.get("device")) for item in metrics if item.get("device") is not None}
        points = {(str(item.get("device")), item.get("size_bytes")) for item in metrics}
        return f"{len(devices)} 个 Device，{len(points)} 个扫点"
    if name == "interconnect-P2P_intraserver-latency":
        pairs = {(str(item.get("source_device")), str(item.get("destination_device")))
                 for item in metrics}
        return f"{len(pairs)} 个有向 pair，{len(metrics)} 个扫点"
    if name == "interconnect-MPI_intraserver":
        scope = case.get("hccl_scope", {})
        return f"{scope.get('rank_count', '—')} ranks，{len(metrics)} 个消息大小"
    if name == "interconnect-P2P_intraserver":
        participants = {
            str(item.get(key)) for item in metrics
            for key in ("source_device", "destination_device") if item.get(key) is not None
        }
        p2p_scope = case.get("p2p_scope", {})
        if isinstance(p2p_scope, dict) and p2p_scope.get("mode") == "selected-combinations":
            return (
                f"{len(participants)} 个逻辑 Device，"
                f"{p2p_scope.get('expected_pair_count', 0)} 个单次组合"
            )
        return f"{len(participants)} 张物理 NPU，{len(metrics)} 个有向指标"
    return f"{len(metrics)} 条指标"


def parse_toolbox_info(text: Any) -> dict[str, str]:
    if not isinstance(text, str):
        return {}
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def svg_document(width: int, height: int, title: str, description: str,
                 body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">\n'
        f'  <title id="title">{html.escape(title)}</title>\n'
        f'  <desc id="desc">{html.escape(description)}</desc>\n'
        '  <style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#202936}'
        '.mono{font-family:"DejaVu Sans Mono",monospace}.muted{fill:#667085}'
        '.axis{stroke:#98A2B3;stroke-width:1}.grid{stroke:#E4E7EC;stroke-width:1}'
        '.bar{fill:#2F6FB3;stroke:#194D80;stroke-width:1}.open{fill:#EAF2FB;stroke:#2F6FB3;stroke-width:2}'
        '</style>\n'
        f'  <rect width="{width}" height="{height}" fill="#FCFCFD"/>\n'
        f'{body}\n</svg>\n'
    )


def compute_svg(cases: dict[str, Any]) -> str | None:
    floating = []
    for name in ("computation-BF16", "computation-FP16", "computation-FP32"):
        for metric in cases.get(name, {}).get("metrics", []):
            value = metric.get("value")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                label = name.removeprefix("computation-")
                if metric.get("device") is not None:
                    label += f" / {metric['device']}"
                floating.append((label, float(value), raw_metric_value(metric)))
    int_values = []
    for metric in cases.get("computation-INT8", {}).get("metrics", []):
        value = metric.get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            label = "INT8"
            if metric.get("device") is not None:
                label += f" / {metric['device']}"
            int_values.append((label, float(value), raw_metric_value(metric)))
    if not floating and not int_values:
        return None
    width = 900
    height = 170 + 46 * (len(floating) + len(int_values)) + (45 if int_values else 0)
    left, chart_width = 135, 610
    parts = [
        '  <text x="36" y="36" font-size="22" font-weight="700">算力测量概览</text>',
        '  <text x="36" y="59" font-size="13" class="muted">浮点与 INT8 使用独立坐标，不表示跨单位可比性</text>',
    ]
    y = 92
    if floating:
        maximum = max(value for _, value, _ in floating)
        parts.append(f'  <text x="36" y="{y}" font-size="15" font-weight="700">浮点（TFLOPS）</text>')
        y += 22
        for label, value, raw_value in floating:
            bar_width = chart_width * value / maximum if maximum else 0
            parts.extend([
                f'  <text x="{left - 12}" y="{y + 19}" text-anchor="end" font-size="14">{html.escape(label)}</text>',
                f'  <line x1="{left}" y1="{y + 13}" x2="{left + chart_width}" y2="{y + 13}" class="grid"/>',
                f'  <rect x="{left}" y="{y}" width="{bar_width:.2f}" height="26" rx="3" class="bar"/>',
                f'  <text x="{left + bar_width + 9:.2f}" y="{y + 19}" font-size="13" class="mono">{html.escape(raw_value)}</text>',
            ])
            y += 46
    if int_values:
        y += 12
        parts.append(f'  <text x="36" y="{y}" font-size="15" font-weight="700">整数（TOPS）</text>')
        y += 22
        maximum = max(value for _, value, _ in int_values)
        for label, value, raw_value in int_values:
            bar_width = chart_width * value / maximum if maximum else 0
            parts.extend([
                f'  <text x="{left - 12}" y="{y + 19}" text-anchor="end" font-size="14">{html.escape(label)}</text>',
                f'  <line x1="{left}" y1="{y + 13}" x2="{left + chart_width}" y2="{y + 13}" class="grid"/>',
                f'  <rect x="{left}" y="{y}" width="{bar_width:.2f}" height="26" rx="3" class="bar"/>',
                f'  <text x="{left + bar_width + 9:.2f}" y="{y + 19}" font-size="13" class="mono">{html.escape(raw_value)}</text>',
            ])
            y += 46
    return svg_document(
        width, height, "算力测量概览",
        "BF16、FP16、FP32 和 INT8 的本轮 DMI 原始指标；每个厂商执行目标独立绘制。",
        "\n".join(parts),
    )


def grouped_device_stats(metrics: Iterable[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for metric in metrics:
        device = metric.get("device")
        value = metric.get("value")
        if device is not None and isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[str(device)].append(float(value))
    def device_key(item: tuple[str, Any]) -> tuple[int, str]:
        return (int(item[0]), item[0]) if item[0].isdigit() else (10**9, item[0])
    return [(device, numeric_summary(values)) for device, values in sorted(grouped.items(), key=device_key)]


def d2d_svg(case: dict[str, Any]) -> str | None:
    rows = grouped_device_stats(case.get("metrics", []))
    if len(rows) < 2:
        return None
    minima = [float(stats["min"]) for _, stats in rows if stats["min"] is not None]
    maxima = [float(stats["max"]) for _, stats in rows if stats["max"] is not None]
    lower, upper = min(minima), max(maxima)
    padding = max((upper - lower) * 0.08, 0.5)
    axis_min, axis_max = lower - padding, upper + padding
    width, row_height = 980, 32
    height = 118 + row_height * len(rows)
    left, chart_width = 105, 720
    def x(value: float) -> float:
        return left + (value - axis_min) / (axis_max - axis_min) * chart_width
    parts = [
        '  <text x="32" y="34" font-size="22" font-weight="700">D2D 逐设备分布</text>',
        f'  <text x="32" y="57" font-size="13" class="muted">每行显示 min—median—max；聚焦坐标 {format_number(axis_min)}–{format_number(axis_max)} GB/s，不从零开始</text>',
        f'  <line x1="{left}" y1="82" x2="{left + chart_width}" y2="82" class="axis"/>',
        f'  <text x="{left}" y="76" font-size="11" class="mono muted">{format_number(axis_min)}</text>',
        f'  <text x="{left + chart_width}" y="76" text-anchor="end" font-size="11" class="mono muted">{format_number(axis_max)} GB/s</text>',
    ]
    for index, (device, stats) in enumerate(rows):
        y = 101 + index * row_height
        minimum = float(stats["min"])
        median = float(stats["median"])
        maximum = float(stats["max"])
        parts.extend([
            f'  <text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="13">Device {html.escape(device)}</text>',
            f'  <line x1="{left}" y1="{y}" x2="{left + chart_width}" y2="{y}" class="grid"/>',
            f'  <line x1="{x(minimum):.2f}" y1="{y}" x2="{x(maximum):.2f}" y2="{y}" stroke="#2F6FB3" stroke-width="4"/>',
            f'  <circle cx="{x(minimum):.2f}" cy="{y}" r="4" class="open"/>',
            f'  <circle cx="{x(median):.2f}" cy="{y}" r="5" fill="#194D80"/>',
            f'  <circle cx="{x(maximum):.2f}" cy="{y}" r="4" class="open"/>',
            f'  <text x="{left + chart_width + 12}" y="{y + 5}" font-size="11" class="mono">{format_number(minimum)}/{format_number(median)}/{format_number(maximum)}</text>',
        ])
    return svg_document(width, height, "D2D 逐设备分布", "每个逻辑 Device 的最小值、中位数和最大值。", "\n".join(parts))


def color_for(value: float, minimum: float, maximum: float) -> tuple[str, str]:
    ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    start = (234, 242, 251)
    end = (47, 111, 179)
    rgb = tuple(round(start[index] + ratio * (end[index] - start[index])) for index in range(3))
    fill = "#" + "".join(f"{component:02X}" for component in rgb)
    return fill, "#FFFFFF" if ratio > 0.58 else "#202936"


def p2p_svg(case: dict[str, Any], direction: str) -> str | None:
    metrics = [item for item in case.get("metrics", []) if item.get("direction") == direction]
    participants = sorted({
        str(item.get(key)) for item in metrics
        for key in ("source_device", "destination_device") if item.get(key) is not None
    }, key=lambda item: (int(item), item) if item.isdigit() else (10**9, item))
    if len(participants) < 2:
        return None
    lookup = {(str(item.get("source_device")), str(item.get("destination_device"))): float(item["value"])
              for item in metrics if isinstance(item.get("value"), (int, float))}
    values = list(lookup.values())
    if not values:
        return None
    minimum, maximum = min(values), max(values)
    cell, left, top = 78, 116, 102
    width, height = left + cell * len(participants) + 34, top + cell * len(participants) + 72
    label = "单向" if direction == "unidirectional" else "双向"
    selected_mode = isinstance(case.get("p2p_scope"), dict) and (
        case["p2p_scope"].get("mode") == "selected-combinations"
    )
    participant_label = "逻辑 Device" if selected_mode else "物理 NPU"
    title_suffix = "所选组合" if selected_mode else "完整矩阵"
    parts = [
        f'  <text x="30" y="34" font-size="22" font-weight="700">P2P {label}{title_suffix}</text>',
        f'  <text x="30" y="58" font-size="13" class="muted">单位 GB/s；色标按本方向独立缩放 {format_number(minimum)}–{format_number(maximum)}</text>',
        f'  <text x="{left + cell * len(participants) / 2}" y="83" text-anchor="middle" font-size="13">目标 {participant_label}</text>',
        f'  <text x="25" y="{top + cell * len(participants) / 2}" transform="rotate(-90 25 {top + cell * len(participants) / 2})" text-anchor="middle" font-size="13">源 {participant_label}</text>',
    ]
    for index, participant in enumerate(participants):
        center = left + index * cell + cell / 2
        parts.append(f'  <text x="{center}" y="{top - 14}" text-anchor="middle" font-size="13">{html.escape(participant)}</text>')
        center_y = top + index * cell + cell / 2 + 5
        parts.append(f'  <text x="{left - 14}" y="{center_y}" text-anchor="end" font-size="13">{html.escape(participant)}</text>')
    for row, source in enumerate(participants):
        for column, destination in enumerate(participants):
            x_pos, y_pos = left + column * cell, top + row * cell
            value = lookup.get((source, destination))
            if value is None:
                fill, text_color, text = "#F2F4F7", "#98A2B3", "N/A"
            else:
                fill, text_color = color_for(value, minimum, maximum)
                text = format_number(value, 2)
            parts.extend([
                f'  <rect x="{x_pos}" y="{y_pos}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" stroke="#FFFFFF"/>',
                f'  <text x="{x_pos + (cell - 2) / 2}" y="{y_pos + cell / 2 + 5}" text-anchor="middle" font-size="12" class="mono" style="fill:{text_color}">{text}</text>',
            ])
    return svg_document(
        width, height, f"P2P {label}{title_suffix}",
        f"{len(participants)} 个{participant_label}的 {direction} P2P 结果。",
        "\n".join(parts),
    )


def sweep_svg(cases: dict[str, dict[str, Any]], *, x_key: str, title: str,
              y_label: str) -> str | None:
    """Render deterministic log-size curves for latency or HCCL bandwidth."""
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for case_name, case in sorted(cases.items(), key=lambda item: case_sort_key(item[0])):
        for metric in case.get("metrics", []):
            x_value, y_value = metric.get(x_key), metric.get("value")
            if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                continue
            if not (x_value > 0 and math.isfinite(float(y_value))):
                continue
            identity = metric.get("device")
            if metric.get("source_device") is not None:
                identity = f"{metric.get('source_device')}→{metric.get('destination_device')}"
            label = CASE_LABELS.get(case_name, case_name) + (f" / {identity}" if identity is not None else "")
            series[label].append((float(x_value), float(y_value)))
    points = [point for values in series.values() for point in values]
    if not points:
        return None
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    width, height = 980, 460
    left, top, chart_width, chart_height = 90, 70, 700, 300
    def xpos(value: float) -> float:
        if xmax == xmin:
            return left + chart_width / 2
        return left + (math.log2(value) - math.log2(xmin)) / (math.log2(xmax) - math.log2(xmin)) * chart_width
    def ypos(value: float) -> float:
        if ymax == ymin:
            return top + chart_height / 2
        return top + chart_height - (value - ymin) / (ymax - ymin) * chart_height
    colors = ("#2F6FB3", "#B54708", "#027A48", "#7A5AF8", "#C11574", "#475467")
    parts = [
        f'  <text x="30" y="34" font-size="22" font-weight="700">{html.escape(title)}</text>',
        f'  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>',
        f'  <text x="{left + chart_width / 2}" y="{height - 28}" text-anchor="middle" font-size="13">消息/传输大小（Byte，对数轴）</text>',
        f'  <text x="22" y="{top + chart_height / 2}" transform="rotate(-90 22 {top + chart_height / 2})" text-anchor="middle" font-size="13">{html.escape(y_label)}</text>',
    ]
    for index, (label, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        ordered = sorted(values)
        path = " ".join(("M" if point_index == 0 else "L") +
                        f" {xpos(x):.2f} {ypos(y):.2f}"
                        for point_index, (x, y) in enumerate(ordered))
        parts.append(f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in ordered:
            parts.append(f'  <circle cx="{xpos(x):.2f}" cy="{ypos(y):.2f}" r="3" fill="{color}"/>')
        legend_y = 78 + index * 20
        parts.extend([
            f'  <line x1="810" y1="{legend_y}" x2="830" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'  <text x="838" y="{legend_y + 4}" font-size="11">{html.escape(label)}</text>',
        ])
    parts.extend([
        f'  <text x="{left}" y="{top + chart_height + 18}" font-size="11" class="mono muted">{format_number(xmin, 0)}</text>',
        f'  <text x="{left + chart_width}" y="{top + chart_height + 18}" text-anchor="end" font-size="11" class="mono muted">{format_number(xmax, 0)}</text>',
        f'  <text x="{left - 8}" y="{top + chart_height}" text-anchor="end" font-size="11" class="mono muted">{format_number(ymin)}</text>',
        f'  <text x="{left - 8}" y="{top + 5}" text-anchor="end" font-size="11" class="mono muted">{format_number(ymax)}</text>',
    ])
    return svg_document(width, height, title, f"{len(series)} 条扫点曲线；精确值见报告表格。", "\n".join(parts))


def write_asset(root: Path, name: str, content: str | None) -> dict[str, Any] | None:
    if content is None:
        return None
    path = root / "report-assets" / name
    atomic_write(path, content)
    return artifact_ref(path, root)


MONITOR_FIELDS = (
    ("aicore_usage_rate_pct", "AICore Usage Rate (%)", "AIC", "Aicore Usage Rate(%)"),
    ("aivector_usage_rate_pct", "AIVector Usage Rate (%)", "AIV", "Aivector Usage Rate(%)"),
    (
        "hbm_bandwidth_usage_rate_pct", "HBM Bandwidth Usage Rate (%)",
        "HBM BW", "HBM Bandwidth Usage Rate(%)",
    ),
    ("npu_utilization_pct", "NPU Utilization (%)", "NPU", "NPU Utilization(%)"),
)


def monitor_unit_samples(root: Path, unit: dict[str, Any]) -> list[dict[str, Any]]:
    reference = unit.get("parsed_samples")
    path = reference.get("path") if isinstance(reference, dict) else None
    if not isinstance(path, str):
        return []
    return read_jsonl(root / "toolkit-evidence" / path)


def monitor_target(sample: dict[str, Any]) -> tuple[int, int, int] | None:
    try:
        return (
            int(sample["npu_id"]), int(sample["chip_id"]),
            int(sample["logical_device_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def monitor_targets(samples: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    return sorted({target for sample in samples if (target := monitor_target(sample))})


def monitor_sample_id(sample: dict[str, Any]) -> str:
    sequence = sample.get("sequence")
    if isinstance(sequence, int):
        return f"S{sequence:02d}"
    if isinstance(sequence, float) and sequence.is_integer():
        return f"S{int(sequence):02d}"
    return "S?" if sequence is None else f"S{sequence}"


def monitor_sample_sort_key(sample: dict[str, Any]) -> tuple[float, float, int, int, int]:
    target = monitor_target(sample) or (sys.maxsize, sys.maxsize, sys.maxsize)
    start = sample.get("started_offset_s")
    finish = sample.get("finished_offset_s")
    return (
        float(start) if isinstance(start, (int, float)) else math.inf,
        float(finish) if isinstance(finish, (int, float)) else math.inf,
        *target,
    )


def monitor_window_roles(
    sample: dict[str, Any], windows: list[dict[str, Any]],
) -> list[str]:
    start = sample.get("started_offset_s")
    finish = sample.get("finished_offset_s")
    if not isinstance(start, (int, float)) or not isinstance(finish, (int, float)):
        return ["时间未知"]
    roles: list[str] = []
    for window in windows:
        window_start = window.get("started_offset_s")
        window_finish = window.get("finished_offset_s")
        if not isinstance(window_start, (int, float)) or not isinstance(
            window_finish, (int, float)
        ):
            continue
        if float(start) <= float(window_finish) and float(finish) >= float(window_start):
            roles.append(str(window.get("role", "未命名窗口")))
    return roles or ["窗口外"]


def monitor_source_value(
    sample: dict[str, Any], normalized_field: str, source_field: str,
) -> str:
    source_fields = sample.get("source_fields")
    if isinstance(source_fields, dict):
        raw = source_fields.get(source_field)
        if raw is not None:
            return str(raw)
    values = sample.get("values")
    value = values.get(normalized_field) if isinstance(values, dict) else None
    return str(value) if isinstance(value, (int, float)) else "—"


def monitor_sample_validity(sample: dict[str, Any]) -> str:
    if sample.get("valid") is True:
        return "有效"
    missing = sample.get("missing_or_invalid_fields")
    if isinstance(missing, list) and missing:
        return "无效：" + "、".join(str(item) for item in missing)
    return "无效"


def monitor_anchor(
    case_name: str | None, section: str, unit_index: int | None = None,
) -> str:
    parts = ["monitor"]
    if case_name:
        parts.append(case_name.removeprefix("computation-").lower())
    if unit_index is not None:
        parts.extend(("unit", str(unit_index + 1)))
    parts.append(section)
    return "-".join(parts)


def monitor_sample_rows(
    samples: list[dict[str, Any]], windows: list[dict[str, Any]],
    target: tuple[int, int, int],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for sample in sorted(samples, key=monitor_sample_sort_key):
        if monitor_target(sample) != target:
            continue
        started_at = sample.get("sample_started_at", "—")
        finished_at = sample.get("sample_finished_at", "—")
        start_offset = sample.get("started_offset_s", "—")
        finish_offset = sample.get("finished_offset_s", "—")
        rows.append([
            monitor_sample_id(sample), " + ".join(monitor_window_roles(sample, windows)),
            f"{started_at}\n→ {finished_at}",
            f"{start_offset} → {finish_offset}",
            *(
                monitor_source_value(sample, field, source_field)
                for field, _label, _short_label, source_field in MONITOR_FIELDS
            ),
            monitor_sample_validity(sample),
        ])
    return rows


def monitor_heatmap_svg(
    title: str, samples: list[dict[str, Any]], windows: list[dict[str, Any]],
) -> str | None:
    samples = [
        item for item in samples
        if monitor_target(item) is not None
        and isinstance(item.get("started_offset_s"), (int, float))
        and isinstance(item.get("finished_offset_s"), (int, float))
    ]
    if not samples:
        return None
    samples.sort(key=monitor_sample_sort_key)
    targets = monitor_targets(samples)
    if not targets:
        return None
    maximum_time = max(
        [float(item["finished_offset_s"]) for item in samples]
        + [float(window.get("finished_offset_s", 0)) for window in windows]
        + [1.0]
    )
    header_samples: dict[str, dict[str, Any]] = {}
    for sample in samples:
        header_samples.setdefault(monitor_sample_id(sample), sample)
    ordered_headers = sorted(header_samples.values(), key=monitor_sample_sort_key)
    left = 230
    chart_width = max(930, 44 * len(ordered_headers))
    right = 34
    row_height = 28
    target_header_height = 24
    target_gap = 12
    chart_top = 128
    target_height = target_header_height + row_height * len(MONITOR_FIELDS) + target_gap
    chart_bottom = chart_top + target_height * len(targets) - target_gap
    width, height = left + chart_width + right, chart_bottom + 52

    def xpos(value: float) -> float:
        return left + max(0.0, min(value, maximum_time)) / maximum_time * chart_width

    def sample_color(value: float) -> str:
        ratio = max(0.0, min(float(value), 100.0)) / 100.0
        start, end = (238, 244, 252), (20, 82, 145)
        rgb = tuple(round(start[i] + ratio * (end[i] - start[i])) for i in range(3))
        return "#" + "".join(f"{item:02X}" for item in rgb)

    parts = [
        f'  <text x="28" y="34" font-size="22" font-weight="700">{html.escape(title)}</text>',
        '  <text x="28" y="56" font-size="12" class="muted">每个色块直接显示 npu-smi 原值；Sxx 可在报告逐样本表中定位 UTC、设备和 workload 窗口</text>',
        '  <rect x="28" y="70" width="18" height="11" fill="#DCEBFA"/>',
        '  <text x="52" y="80" font-size="11">primary DMI</text>',
        '  <rect x="146" y="70" width="18" height="11" fill="#F4E8C8"/>',
        '  <text x="170" y="80" font-size="11">extension</text>',
        '  <rect x="258" y="70" width="18" height="11" fill="#EEF4FC" stroke="#98A2B3"/>',
        '  <text x="282" y="80" font-size="11">色深辅助表示 0–100；数字是原值</text>',
    ]
    for window in windows:
        start = window.get("started_offset_s")
        finish = window.get("finished_offset_s")
        if not isinstance(start, (int, float)) or not isinstance(finish, (int, float)):
            continue
        fill = "#DCEBFA" if window.get("role") == "primary" else "#F4E8C8"
        parts.append(
            f'  <rect x="{xpos(float(start)):.2f}" y="{chart_top}" '
            f'width="{max(1.0, xpos(float(finish)) - xpos(float(start))):.2f}" '
            f'height="{chart_bottom - chart_top}" fill="{fill}" opacity="0.48"/>'
        )
    for sample in ordered_headers:
        center = xpos(
            (float(sample["started_offset_s"]) + float(sample["finished_offset_s"])) / 2
        )
        parts.extend([
            f'  <text x="{center:.2f}" y="108" text-anchor="middle" font-size="10" class="mono sample-id">{html.escape(monitor_sample_id(sample))}</text>',
            f'  <line x1="{center:.2f}" y1="114" x2="{center:.2f}" y2="{chart_bottom}" class="grid" opacity="0.55"/>',
        ])
    for target_index, target in enumerate(targets):
        group_top = chart_top + target_index * target_height
        npu_id, chip_id, logic_id = target
        parts.append(
            f'  <text x="28" y="{group_top + 16}" font-size="13" font-weight="700">NPU {npu_id} / Chip {chip_id} / Device {logic_id}</text>'
        )
        target_samples = [sample for sample in samples if monitor_target(sample) == target]
        for field_index, (field, _label, short_label, source_field) in enumerate(MONITOR_FIELDS):
            row_top = group_top + target_header_height + field_index * row_height
            center_y = row_top + row_height / 2
            parts.extend([
                f'  <text x="{left - 12}" y="{center_y + 4:.2f}" text-anchor="end" font-size="11">{html.escape(short_label)}</text>',
                f'  <line x1="{left}" y1="{row_top + row_height:.2f}" x2="{left + chart_width}" y2="{row_top + row_height:.2f}" class="grid"/>',
            ])
            for sample in target_samples:
                start_x = xpos(float(sample["started_offset_s"])) + 1
                finish_x = xpos(float(sample["finished_offset_s"])) - 1
                cell_width = max(1.0, finish_x - start_x)
                raw_value = monitor_source_value(sample, field, source_field)
                normalized = sample.get("values", {}).get(field) if isinstance(
                    sample.get("values"), dict
                ) else None
                valid_value = isinstance(normalized, (int, float)) and sample.get("valid") is True
                fill = sample_color(float(normalized)) if valid_value else "#D0D5DD"
                text_fill = "#FFFFFF" if valid_value and float(normalized) >= 58 else "#202936"
                parts.extend([
                    f'  <rect x="{start_x:.2f}" y="{row_top + 3:.2f}" width="{cell_width:.2f}" height="{row_height - 6}" rx="3" class="sample-cell" fill="{fill}" stroke="#FFFFFF" stroke-width="0.8"/>',
                    f'  <text x="{start_x + cell_width / 2:.2f}" y="{center_y + 4:.2f}" text-anchor="middle" font-size="10" class="mono sample-value" fill="{text_fill}" style="fill:{text_fill}">{html.escape(raw_value)}</text>',
                ])
    parts.extend([
        f'  <line x1="{left}" y1="{chart_bottom + 4}" x2="{left + chart_width}" y2="{chart_bottom + 4}" class="axis"/>',
        f'  <text x="{left}" y="{chart_bottom + 21}" font-size="10" class="mono muted">0 s</text>',
        f'  <text x="{left + chart_width}" y="{chart_bottom + 21}" text-anchor="end" font-size="10" class="mono muted">{format_number(maximum_time, 2)} s</text>',
    ])
    return svg_document(
        width, height, title,
        "每个 NPU/Chip/逻辑 Device 的四类 npu-smi 原始监控区间和可见原值；蓝色背景为主 DMI，黄色为扩展负载。",
        "\n".join(parts),
    )


def hccs_timeline_svg(
    title: str, samples: list[dict[str, Any]], windows: list[dict[str, Any]],
) -> str | None:
    """Render raw HCCS Rx/Tx values as a non-interactive, auditable timeline."""
    samples = [item for item in samples if monitor_target(item) is not None]
    if not samples:
        return None
    samples.sort(key=monitor_sample_sort_key)
    sequences = sorted({int(item["sequence"]) for item in samples if isinstance(item.get("sequence"), int)})
    rows = sorted({
        (*monitor_target(item), str(item.get("link", "?")), direction)
        for item in samples for direction in ("rx", "tx")
        if monitor_target(item) is not None
    })
    left, cell_width, row_height, top = 300, 68, 26, 104
    chart_width = max(760, cell_width * max(1, len(sequences)))
    width, height = left + chart_width + 30, top + row_height * len(rows) + 48
    by_cell = {
        ((*monitor_target(item), str(item.get("link", "?")), direction), int(item["sequence"])):
        monitor_source_value(item, f"{direction}_gb_s", f"{direction}_bandwidth(GB/S)")
        for item in samples for direction in ("rx", "tx")
        if monitor_target(item) is not None and isinstance(item.get("sequence"), int)
    }
    parts = [
        f'  <text x="28" y="34" font-size="22" font-weight="700">{html.escape(title)}</text>',
        '  <text x="28" y="57" font-size="12" class="muted">每格为 npu-smi hccs-bw 原始 GB/S；Sxx 对应报告中的 UTC 与 workload 窗口</text>',
    ]
    for index, sequence in enumerate(sequences):
        x = left + index * chart_width / max(1, len(sequences))
        parts.append(f'  <text x="{x + chart_width / max(1, len(sequences)) / 2:.2f}" y="88" text-anchor="middle" font-size="10" class="mono">S{sequence:02d}</text>')
    for row_index, row in enumerate(rows):
        npu, chip, logic, link, direction = row
        y = top + row_index * row_height
        label = f"NPU {npu}/Chip {chip}/Dev {logic} · Link {link} · {direction.upper()}"
        parts.append(f'  <text x="{left - 10}" y="{y + 17}" text-anchor="end" font-size="10">{html.escape(label)}</text>')
        for index, sequence in enumerate(sequences):
            x = left + index * chart_width / max(1, len(sequences))
            width_cell = chart_width / max(1, len(sequences))
            value = by_cell.get((row, sequence), "—")
            parts.extend([
                f'  <rect x="{x + 1:.2f}" y="{y + 2}" width="{max(1, width_cell - 2):.2f}" height="{row_height - 4}" rx="3" fill="#E8F1FB" stroke="#B8CCE4"/>',
                f'  <text x="{x + width_cell / 2:.2f}" y="{y + 17}" text-anchor="middle" font-size="9" class="mono">{html.escape(value)}</text>',
            ])
    return svg_document(
        width, height, title,
        "静态 HCCS 时间线，每个单元格直接显示厂商 Rx/Tx 原值，无需任何交互或 JavaScript。",
        "\n".join(parts),
    )


def render_monitor_report(
    root: Path, summary: dict[str, Any], manifest: dict[str, Any] | None,
    monitor_assets: dict[tuple[str, int], str],
) -> str:
    cases = manifest.get("cases", {}) if manifest else {}
    compute_cases = {
        name: case for name, case in cases.items() if name.startswith("computation-")
    }
    lines = [
        "# Ascend 910C 作证监控报告",
        "",
        "## 快速导航",
        "",
        f"- [结论边界](#{monitor_anchor(None, 'boundary')})",
    ]
    for case_name, case in sorted(compute_cases.items(), key=lambda item: case_sort_key(item[0])):
        label = CASE_LABELS.get(case_name, case_name)
        monitor = case.get("monitor") if isinstance(case.get("monitor"), dict) else {}
        units = monitor.get("units", []) if isinstance(monitor.get("units"), list) else []
        links = [
            f"[{label}](#{monitor_anchor(case_name, 'case')})",
            f"[DMI 原值](#{monitor_anchor(case_name, 'dmi')})",
        ]
        for unit_index, _unit in enumerate(units):
            links.extend([
                f"[Unit {unit_index + 1} timeline](#{monitor_anchor(case_name, 'timeline', unit_index)})",
                f"[Unit {unit_index + 1} 逐样本原值](#{monitor_anchor(case_name, 'samples', unit_index)})",
            ])
        lines.append("- " + " · ".join(links))
    for case_name in sorted(
        (name for name in cases if name in {
            "main_memory-bandwidth", "main_memory-capacity", "interconnect-h2d",
            "interconnect-d2h", "interconnect-P2P_intraserver",
            "interconnect-MPI_intraserver",
        }), key=case_sort_key,
    ):
        lines.append(
            f"- [{CASE_LABELS.get(case_name, case_name)}]"
            f"(#{monitor_anchor(case_name, 'movement')})"
        )
    lines.extend([
        f"- [健康与源证据](#{monitor_anchor(None, 'evidence')})",
        "",
        f'<a id="{monitor_anchor(None, "boundary")}"></a>',
        "## 结论边界",
        "",
        "- 本报告并列展示 DMI 厂商原值和同一 workload 窗口内的 `npu-smi` 原始时序，不计算理论峰值兑现率。",
        "- `monitoring_status=passed` 只表示该 Case 要求的厂商监控字段达到采样覆盖门槛；它不自动判定硬件达到理论峰值。",
        "- AICore 是本轮 GEMM 的主要活动证据；AIVector、HBM Bandwidth 和 NPU Utilization 作为同期旁证，任何字段都不设置性能阈值。",
        "- 时间对齐使用采样命令区间与 DMI 进程墙钟区间的重叠；DMI 内部矩阵执行 `duration` 仍作为独立厂商原字段展示，不能把两者混写成同一个精确窗口。",
        "- timeline 的每个色块直接显示原值；样本编号 `Sxx` 与下方逐样本表一一对应，无需任何交互。",
        "",
        table(["项目", "值"], [
            ["运行 ID", summary.get("run_id", root.name)],
            ["实验状态", status_text(manifest.get("status") if manifest else summary.get("status"))],
            ["监控范围", "FP16 / FP32 / BF16 / INT8 / D2D / H2D / D2H / P2P / 单机 HCCL / HBM 容量"],
            ["数据策略", "只展示厂商原值、采样窗口和路由覆盖；不计算利用率、兑现率或平均值"],
        ]),
    ])
    if not compute_cases:
        lines.extend(["", "本轮没有计算类 Case，因此没有可展示的监控数据。"])
    for case_name, case in sorted(compute_cases.items(), key=lambda item: case_sort_key(item[0])):
        monitor = case.get("monitor") if isinstance(case.get("monitor"), dict) else {}
        lines.extend([
            "", f'<a id="{monitor_anchor(case_name, "case")}"></a>',
            f"## {CASE_LABELS.get(case_name, case_name)}", "",
            table(["测量层", "监控层", "诊断层", "最终状态"], [[
                status_text(inferred_measurement(case)[0]),
                status_text(case.get("monitoring_status", monitor.get("status", "missing"))),
                status_text(case.get("diagnosis_status", "missing")),
                status_text(case.get("status", "missing")),
            ]]),
            "", f'<a id="{monitor_anchor(case_name, "dmi")}"></a>',
            "### DMI 主结果（原值）", "",
        ])
        primary_rows = [[
            item.get("device", "—"), raw_metric_value(item), item.get("unit", "—"),
            item.get("dmi_context", {}).get("execute_times", "—"),
            item.get("dmi_context", {}).get("duration", "—"),
            item.get("dmi_context", {}).get("power", "—"),
            item.get("field", item.get("source", "—")),
        ] for item in case.get("metrics", [])]
        lines.append(table([
            "厂商执行目标", "原值", "单位", "Execute Times", "Duration", "Power", "源字段"
        ], primary_rows))
        units = monitor.get("units", []) if isinstance(monitor.get("units"), list) else []
        if not units:
            lines.extend(["", "未形成监控单元；请结合 Case 的 `monitoring_status` 和错误字段解释。"])
            continue
        for unit_index, unit in enumerate(units):
            targets = unit.get("targets", [])
            counts = unit.get("sample_counts_by_target", {})
            primary_counts = unit.get("primary_sample_counts_by_target", {})
            samples = monitor_unit_samples(root, unit)
            sample_targets = monitor_targets(samples)
            lines.extend([
                "", f'<a id="{monitor_anchor(case_name, "unit", unit_index)}"></a>',
                f"### 监控单元 {unit_index + 1}", "",
                table(["NPU / Chip / Device", "workload 内完整样本", "主 DMI 内样本"], [
                    [
                        f"{target.get('npu_id')}/{target.get('chip_id')}/{target.get('logic_id')}",
                        counts.get(f"{target.get('npu_id')}/{target.get('chip_id')}/{target.get('logic_id')}", 0),
                        primary_counts.get(f"{target.get('npu_id')}/{target.get('chip_id')}/{target.get('logic_id')}", 0),
                    ] for target in targets
                ]),
            ])
            asset = monitor_assets.get((case_name, unit_index))
            lines.extend([
                "", f'<a id="{monitor_anchor(case_name, "timeline", unit_index)}"></a>',
                "#### Timeline 与可见原值", "",
                "横轴使用真实采样命令区间；色块内数字为 `npu-smi` 原值，`Sxx` 用于关联下方明细表。",
            ])
            if asset:
                lines.append(f"\n![{CASE_LABELS.get(case_name, case_name)} 监控时间线]({asset})")
            else:
                lines.extend(["", "未形成可视化 timeline；请查看下方逐样本表和原始 JSONL。"])
            lines.extend([
                "", f'<a id="{monitor_anchor(case_name, "samples", unit_index)}"></a>',
                "#### 逐样本原值", "",
                "表中四个百分比直接来自结构化样本的 `source_fields`；不做平均、舍入或峰值换算。",
            ])
            if not samples:
                lines.extend(["", "未读取到结构化监控样本。"])
            else:
                for npu_id, chip_id, logic_id in sample_targets:
                    lines.extend([
                        "", f"##### NPU {npu_id} / Chip {chip_id} / Device {logic_id}", "",
                        table([
                            "样本", "workload 窗口", "UTC 采样区间", "相对区间 (s)",
                            "AIC (%)", "AIV (%)", "HBM BW (%)", "NPU (%)", "有效性",
                        ], monitor_sample_rows(
                            samples, unit.get("workload_windows", []),
                            (npu_id, chip_id, logic_id),
                        )),
                    ])
            extensions = unit.get("extensions", [])
            if extensions:
                extension_rows = []
                for extension_index, extension in enumerate(extensions, 1):
                    metrics = extension.get("raw_dmi_metrics", [])
                    if metrics:
                        for item in metrics:
                            stdout_path = (
                                extension.get("stdout", {}).get("path")
                                if isinstance(extension.get("stdout"), dict) else None
                            )
                            extension_rows.append([
                                extension_index, item.get("device", "—"),
                                raw_metric_value(item), item.get("unit", "—"),
                                item.get("dmi_context", {}).get("duration", "—"),
                                (
                                    markdown_link("stdout", "toolkit-evidence/" + stdout_path)
                                    if stdout_path else "缺失"
                                ),
                            ])
                    else:
                        extension_rows.append([
                            extension_index, "—", "未形成指标", extension.get("returncode", "—"),
                            "—", "缺失",
                        ])
                lines.extend([
                    "", "追加负载仅延长监控窗口；下列 DMI 原值不参与主性能结果，也不做聚合。", "",
                    table([
                        "追加序号", "厂商执行目标", "DMI 原值", "单位/rc", "Duration", "原文"
                    ], extension_rows),
                ])
            raw_ref = unit.get("raw_samples", {}).get("path") if isinstance(unit.get("raw_samples"), dict) else None
            parsed_ref = unit.get("parsed_samples", {}).get("path") if isinstance(unit.get("parsed_samples"), dict) else None
            links = []
            if raw_ref:
                links.append(markdown_link("npu-smi 原始 JSONL", "toolkit-evidence/" + raw_ref))
            if parsed_ref:
                links.append(markdown_link("逐 Chip 结构化 JSONL", "toolkit-evidence/" + parsed_ref))
            if links:
                lines.extend(["", "原始证据：" + " · ".join(links)])
            reasons = unit.get("reasons", [])
            if reasons:
                lines.extend(["", "证据不完整原因：", ""])
                lines.extend(f"- {md_escape(reason)}" for reason in reasons)
    movement_names = {
        "main_memory-bandwidth", "main_memory-capacity", "interconnect-h2d",
        "interconnect-d2h", "interconnect-P2P_intraserver",
        "interconnect-MPI_intraserver",
    }
    movement_cases = {
        name: case for name, case in cases.items() if name in movement_names
    }
    if movement_cases:
        lines.extend([
            "", "## 数据搬运与 HBM 静态作证数据", "",
            "D2D 以 HBM Bandwidth Usage Rate(%) 时间线作证；H2D/D2H/P2P/HCCL 以 HCCS 逐链路 Rx/Tx 原值作证。若拓扑含 SIO 而当前栈没有动态 SIO 计数，Case 明确为 `partial`，不会用 HBM 指标替代链路指标。HBM 容量属于静态属性，因此不伪造 timeline。",
        ])
    for case_name, case in sorted(movement_cases.items(), key=lambda item: case_sort_key(item[0])):
        monitor = case.get("monitor") if isinstance(case.get("monitor"), dict) else {}
        lines.extend([
            "", f'<a id="{monitor_anchor(case_name, "movement")}"></a>',
            f"### {CASE_LABELS.get(case_name, case_name)}", "",
            table(["DMI 测量", "监控", "最终状态", "监控类型"], [[
                status_text(inferred_measurement(case)[0]),
                status_text(case.get("monitoring_status", monitor.get("status", "missing"))),
                status_text(case.get("status", "missing")),
                monitor.get("evidence_kind", "—"),
            ]]),
        ])
        metric_rows = [[
            item.get("device", item.get("source_device", "—")),
            item.get("destination_device", "—"), item.get("direction", "—"),
            raw_metric_value(item), item.get("unit", "—"),
        ] for item in case.get("metrics", [])]
        lines.extend(["", "DMI/HCCL 主结果仍按厂商原值展示：", "", table([
            "Device/源", "目的", "方向", "原值", "单位"
        ], metric_rows)])
        if case_name == "main_memory-capacity":
            samples = monitor.get("samples", []) if isinstance(monitor.get("samples"), list) else []
            rows = []
            for sample in samples:
                fields = sample.get("source_fields", {})
                ecc = sample.get("ecc_guard", {})
                ecc_fields = ecc.get("source_fields", {}) if isinstance(ecc, dict) else {}
                isolated = ", ".join(
                    f"{key}={value}" for key, value in ecc_fields.items()
                    if "Isolated Pages Count" in key
                ) or "—"
                rows.append([
                    f"{sample.get('npu_id')}/{sample.get('chip_id')}/{sample.get('logical_device_id')}",
                    fields.get("HBM Capacity(MB)", "—"),
                    fields.get("HBM Clock Speed(MHz)", "—"),
                    fields.get("HBM Temperature(C)", "—"),
                    isolated, ecc.get("returncode", "—") if isinstance(ecc, dict) else "—",
                ])
            lines.extend(["", table([
                "NPU/Chip/Device", "HBM Capacity(MB)", "HBM Clock(MHz)",
                "HBM Temperature(C)", "ECC 隔离页原字段", "ECC rc",
            ], rows)])
            continue
        units = monitor.get("units", []) if isinstance(monitor.get("units"), list) else []
        route_coverage = monitor.get("route_coverage", {})
        if route_coverage:
            lines.extend(["", table([
                "路由总数", "HCCS/HCCS_SW", "SIO", "其他/未知"
            ], [[
                route_coverage.get("selected_route_count", 0),
                route_coverage.get("hccs_or_switch_count", 0),
                route_coverage.get("sio_count", 0),
                route_coverage.get("unknown_count", 0),
            ]])])
            routes = monitor.get("routes", []) if isinstance(monitor.get("routes"), list) else []
            if routes:
                lines.extend(["", table([
                    "源 Device (Phy)", "目的 Device (Phy)", "topo relation", "动态作证覆盖"
                ], [[
                    f"{route.get('source_device', '—')} ({route.get('source_phy_id', '—')})",
                    f"{route.get('destination_device', '—')} ({route.get('destination_phy_id', '—')})",
                    route.get("relation", "—"),
                    ("hccs-bw" if route.get("relation") in ("HCCS", "HCCS_SW") else "未覆盖"),
                ] for route in routes])])
        for unit_index, unit in enumerate(units):
            samples = monitor_unit_samples(root, unit)
            asset = monitor_assets.get((case_name, unit_index))
            lines.extend(["", f"#### 监控单元 {unit_index + 1}：timeline 与原值", ""])
            if asset:
                lines.append(f"![{CASE_LABELS.get(case_name, case_name)} 监控时间线]({asset})")
            rows = []
            for sample in sorted(samples, key=monitor_sample_sort_key):
                roles = " + ".join(monitor_window_roles(sample, unit.get("workload_windows", [])))
                if unit.get("target_resource") == "hbm-bandwidth":
                    rows.append([
                        monitor_sample_id(sample), roles,
                        f"{sample.get('sample_started_at', '—')} → {sample.get('sample_finished_at', '—')}",
                        f"{sample.get('npu_id')}/{sample.get('chip_id')}/{sample.get('logical_device_id')}",
                        monitor_source_value(sample, "hbm_bandwidth_usage_rate_pct", "HBM Bandwidth Usage Rate(%)"),
                        monitor_source_value(sample, "npu_utilization_pct", "NPU Utilization(%)"),
                    ])
                else:
                    rows.append([
                        monitor_sample_id(sample), roles,
                        f"{sample.get('sample_started_at', '—')} → {sample.get('sample_finished_at', '—')}",
                        f"{sample.get('npu_id')}/{sample.get('chip_id')}/{sample.get('logical_device_id')}",
                        sample.get("link", "—"),
                        monitor_source_value(sample, "rx_gb_s", "rx_bandwidth(GB/S)"),
                        monitor_source_value(sample, "tx_gb_s", "tx_bandwidth(GB/S)"),
                    ])
            headers = (["样本", "窗口", "UTC 区间", "NPU/Chip/Device", "HBM BW (%)", "NPU (%)"]
                       if unit.get("target_resource") == "hbm-bandwidth" else
                       ["样本", "窗口", "UTC 区间", "NPU/Chip/Device", "Link", "Rx (GB/S)", "Tx (GB/S)"])
            lines.extend(["", table(headers, rows)])
            refs = []
            for key, label in (("raw_samples", "原始 JSONL"), ("parsed_samples", "结构化 JSONL")):
                ref = unit.get(key, {})
                if isinstance(ref, dict) and ref.get("path"):
                    refs.append(markdown_link(label, "toolkit-evidence/" + ref["path"]))
            if refs:
                lines.extend(["", "证据：" + " · ".join(refs)])
        reasons = monitor.get("reasons", [])
        if reasons:
            lines.extend(["", "证据不完整原因：", ""])
            lines.extend(f"- {md_escape(reason)}" for reason in reasons)
    lines.extend([
        "", f'<a id="{monitor_anchor(None, "evidence")}"></a>',
        "## 健康与源证据", "",
        "- " + evidence_link(root, "Toolkit manifest", "toolkit-evidence/manifest.json"),
        "- " + evidence_link(root, "DMI pre-health", "toolkit-evidence/health/pre/health.stdout"),
        "- " + evidence_link(root, "DMI post-health", "toolkit-evidence/health/post/health.stdout"),
        "", "---", "",
        f"监控报告生成器 schema：`{REPORT_SCHEMA_VERSION}`。发生冲突时以 DMI/npu-smi 原始输出、manifest 和 SHA-256 为准。", "",
    ])
    return "\n".join(lines)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(md_escape(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def selection_info(
    summary: dict[str, Any], manifest: dict[str, Any] | None
) -> dict[str, Any]:
    preflight = summary.get("host_preflight", {}).get("result", {}) if isinstance(
        summary.get("host_preflight"), dict
    ) else {}
    selection = preflight.get("selection") if isinstance(preflight, dict) else None
    if not isinstance(selection, dict) and manifest:
        selection = manifest.get("selection")
    if not isinstance(selection, dict):
        discovered = (
            manifest.get("discovered_device_ids", []) if manifest
            else summary.get("device_nodes_exposed", [])
        )
        selection = {
            "source": "legacy-result",
            "selected_device_ids": discovered,
            "excluded_device_ids": [],
        }
    inventory = (
        manifest.get("discovered_device_ids", []) if manifest
        else preflight.get("actual_device_node_ids", [])
    )
    return {
        "source": selection.get("source", "unknown"),
        "requested_ids": selection.get("requested_ids", []),
        "inventory_device_ids": inventory,
        "selected_npu_ids": selection.get("selected_npu_ids", []),
        "selected_device_ids": selection.get("selected_device_ids", []),
        "excluded_device_ids": selection.get("excluded_device_ids", []),
    }


def report_environment(summary: dict[str, Any], manifest: dict[str, Any] | None) -> str:
    labels = summary.get("image_labels") if isinstance(summary.get("image_labels"), dict) else {}
    namespaces = summary.get("container_namespaces") if isinstance(
        summary.get("container_namespaces"), dict
    ) else {}
    toolbox = parse_toolbox_info(summary.get("toolbox_install_info"))
    preflight = summary.get("host_preflight", {}).get("result", {}) if isinstance(summary.get("host_preflight"), dict) else {}
    selection = selection_info(summary, manifest)
    discovered = selection["inventory_device_ids"]
    rows = [
        ["运行 ID", summary.get("run_id", "—")],
        ["主机", preflight.get("host", "—")],
        ["镜像", summary.get("image", "—")],
        ["镜像 ID", summary.get("image_id", "—")],
        ["CANN", labels.get("io.flagrt.cann.version", "—")],
        ["MindCluster ToolBox", toolbox.get("version", "—")],
        ["Python", labels.get("io.flagrt.python.version", "—")],
        ["PyTorch", labels.get("io.flagrt.torch.version", "—")],
        ["triton_ascend", labels.get("io.flagrt.triton_ascend.version", "—")],
        ["Torch-FL commit", labels.get("io.flagrt.torch_fl.commit", "—")],
        ["FlagGems commit", labels.get("io.flagrt.flag_gems.commit", "—")],
        ["库存逻辑 Device", f"{len(discovered)}：{discovered}"],
        ["选卡来源", selection["source"]],
        ["参与物理 NPU", selection["selected_npu_ids"] or "未记录"],
        ["选中逻辑 Device", selection["selected_device_ids"]],
        ["排除逻辑 Device", selection["excluded_device_ids"] or "无"],
        ["容器权限", "privileged root" if summary.get("privileged_root") else "非 privileged"],
        ["IPC namespace", namespaces.get("ipc", "未记录")],
        ["PID namespace", namespaces.get("pid", "未记录")],
    ]
    return table(["项目", "本轮事实"], rows)


def case_rows(root: Path, cases: dict[str, Any]) -> tuple[list[list[Any]], bool]:
    rows = []
    inferred_present = False
    for name, case in sorted(cases.items(), key=lambda item: case_sort_key(item[0])):
        measurement, measurement_error, inferred = inferred_measurement(case)
        inferred_present = inferred_present or inferred
        metrics_path = f"toolkit-evidence/cases/{name}/metrics.json"
        evidence_path = metrics_path if (root / metrics_path).is_file() else "toolkit-evidence/manifest.json"
        error = human_error(
            measurement_error or case.get("error"), evidence_path=evidence_path
        )
        rows.append([
            CASE_LABELS.get(name, name),
            status_text(measurement) + ("*" if inferred else ""),
            status_text(case.get("monitoring_status", "not-run")),
            status_text(case.get("diagnosis_status", "missing")),
            status_text(case.get("status", "missing")),
            case_metric_summary(name, case),
            case_scope(name, case),
            format_duration(case.get("duration_s")),
            error,
        ])
    return rows, inferred_present


def evidence_link(root: Path, label: str, relative: str) -> str:
    return markdown_link(label, relative) if (root / relative).is_file() else "缺失"


def render_report(root: Path, summary: dict[str, Any], manifest: dict[str, Any] | None,
                  assets: list[dict[str, Any]]) -> str:
    cases = manifest.get("cases", {}) if manifest and isinstance(manifest.get("cases"), dict) else {}
    statuses = Counter(str(case.get("status", "missing")) for case in cases.values())
    measurements = Counter(inferred_measurement(case)[0] for case in cases.values())
    monitoring = Counter(str(case.get("monitoring_status", "not-run")) for case in cases.values())
    diagnoses = Counter(str(case.get("diagnosis_status", "missing")) for case in cases.values())
    overall = manifest.get("status") if manifest else summary.get("status", "failed")
    started = manifest.get("started_at") if manifest else summary.get("started_at")
    finished = manifest.get("finished_at") if manifest else summary.get("finished_at")
    asset_paths = {item["path"] for item in assets}
    selection = selection_info(summary, manifest)

    lines = [
        "# Ascend 910C Toolkit 实验报告",
        "",
        "## 技术摘要",
        "",
        f"- **总体状态：{status_text(overall)}（`{overall}`）。** 本报告只整理已保存证据，不重新判定硬件结果。",
    ]
    if cases:
        lines.extend([
            f"- **测量层：** {len(cases)} 个 Case 中，通过 {measurements['passed']}、部分完成 {measurements['partial']}、失败 {measurements['failed']}。",
            f"- **作证监控层：** 完整 {monitoring['passed']}、部分完成 {monitoring['partial']}、未执行 {monitoring['not-run']}。",
            f"- **厂商诊断层：** 通过 {diagnoses['passed']}、不支持 {diagnoses['unsupported']}、未执行 {diagnoses['not-run']}、失败 {diagnoses['failed']}、缺失 {diagnoses['missing']}。",
            f"- **最终 Case 状态：** 通过 {statuses['passed']}、部分完成 {statuses['partial']}、失败 {statuses['failed']}。",
        ])
        if measurements["passed"] == len(cases) and statuses["partial"] and diagnoses["failed"] == 0:
            lines.append("- **关键解释：** 本轮所有性能测量均形成有效指标；总体 `partial` 来自厂商诊断阈值不支持，而不是测量命令失败。")
    else:
        stage = summary.get("failure_stage", "unknown")
        lines.append(f"- **未形成 Case manifest。** 运行在 `{stage}` 阶段终止；已有证据和错误见下文。")
    if summary.get("error"):
        lines.append(f"- **外层错误：** `{md_escape(summary['error'])}`")

    lines.extend([
        "",
        "## 环境与运行边界",
        "",
        report_environment(summary, manifest),
        "",
        "计算类与数据搬运类的原始监控 timeline、逐样本原值和路由覆盖见 [report_monitor.md](report_monitor.md)。",
        "",
        f"运行窗口：{human_timestamp(started)} ～ {human_timestamp(finished)}。本报告范围为单机、选中逻辑 Device {selection['selected_device_ids']}；库存与排除集合见上表。"
        + ("本轮包含单节点 HCCL AllReduce；不外推到其他集合通信或多节点能力。"
           if "interconnect-MPI_intraserver" in cases else
           "本轮未测试 HCCL；不外推到 HCCL、跨机 P2P 或多节点能力。"),
    ])

    if cases:
        rows, inferred_present = case_rows(root, cases)
        lines.extend([
            "",
            "## 测量与诊断是两个独立结论",
            "",
            "`measurement_status` 说明性能命令、指标和设备覆盖；`monitoring_status` 说明同期监控覆盖；`diagnosis_status` 说明厂商阈值诊断；最终状态综合三者。",
            "",
            table(["Case", "测量", "监控", "诊断", "最终", "指标摘要", "范围", "耗时", "限制/错误"], rows),
        ])
        if inferred_present:
            lines.append("\n\* 当前结果使用旧 manifest schema，带星号的测量状态由命令退出码、超时、metrics 和诊断状态确定性推导。")

        compute_names = [name for name in cases if name.startswith("computation-")]
        if compute_names:
            lines.extend([
                "",
                "## 算力测量均成功，但阈值诊断需单独看待",
                "",
                "图中浮点算力使用 TFLOPS，INT8 使用 TOPS 并置于独立分面；每个柱对应一条 DMI 原值，不进行平均、中位数或峰值兑现率计算。",
            ])
            if "report-assets/compute-overview.svg" in asset_paths:
                lines.append("\n![算力测量概览](report-assets/compute-overview.svg)")
            compute_rows = []
            for name in sorted(compute_names, key=case_sort_key):
                case = cases[name]
                for metric in case.get("metrics", []):
                    compute_rows.append([
                        CASE_LABELS.get(name, name), metric.get("device", "—"),
                        raw_metric_value(metric), metric.get("unit", "—"),
                        case_scope(name, case),
                        status_text(case.get("monitoring_status", "not-run")),
                        status_text(case.get("diagnosis_status", "missing")),
                    ])
            lines.extend(["", table(
                ["类型", "厂商执行目标", "DMI 原值", "单位", "设备口径", "监控", "aiflops 诊断"],
                compute_rows,
            )])

        h2d = cases.get("interconnect-h2d")
        d2h = cases.get("interconnect-d2h")
        d2d = cases.get("main_memory-bandwidth")
        if h2d or d2h or d2d:
            lines.extend([
                "",
                "## H2D、D2H 与 D2D 回答不同的数据搬运问题",
                "",
                "H2D 是 Host→Device，D2H 是 Device→Host，D2D 是同一 Device 内存内部搬运。三个方向不能相除或合并成一个“内存带宽分数”。",
            ])
        if h2d:
            values = finite_values(h2d.get("metrics", []))
            h2d_stats = numeric_summary(values)
            lines.extend([
                "",
                "### H2D：512 MiB × 50 次",
                "",
                table(["指标", "n", "min", "median", "max", "单位", "设备口径", "诊断"], [[
                    "Host→Device 带宽",
                    h2d_stats["count"], format_number(h2d_stats["min"], 6),
                    format_number(h2d_stats["median"], 6),
                    format_number(h2d_stats["max"], 6), metric_unit(h2d),
                    case_scope("interconnect-h2d", h2d),
                    status_text(h2d.get("diagnosis_status", "missing")),
                ]]),
            ])
            h2d_rows = [[
                item.get("device", "—"), format_number(item.get("value"), 6),
                item.get("unit", "—"),
            ] for item in h2d.get("metrics", [])]
            if h2d_rows:
                lines.extend(["", table(["Device", "值", "单位"], h2d_rows)])
        if d2h:
            values = finite_values(d2h.get("metrics", []))
            stats = numeric_summary(values)
            detail = [[item.get("device", "—"), format_number(item.get("value"), 6),
                       item.get("unit", "—")] for item in d2h.get("metrics", [])]
            lines.extend([
                "", "### D2H：512 MiB × 50 次", "",
                table(["指标", "n", "min", "median", "max", "单位", "设备口径", "诊断"], [[
                    "Device→Host 带宽", stats["count"], format_number(stats["min"], 6),
                    format_number(stats["median"], 6), format_number(stats["max"], 6),
                    metric_unit(d2h), case_scope("interconnect-d2h", d2h),
                    status_text(d2h.get("diagnosis_status", "missing")),
                ]]),
            ])
            if detail:
                lines.extend(["", table(["Device", "值", "单位"], detail)])
        if d2d:
            device_stats = grouped_device_stats(d2d.get("metrics", []))
            lines.extend([
                "",
                "### D2D：逐逻辑设备保留完整重复序列",
                "",
                "区间图关注设备间和设备内波动，因此使用明确标注的聚焦坐标；精确值仍以表格和原始 metrics 为准。标准结果逐逻辑 Device 原样保留 DMI GB/s，不再执行 `×2`。",
            ])
            if "report-assets/d2d-device-distribution.svg" in asset_paths:
                lines.append("\n![D2D 逐设备分布](report-assets/d2d-device-distribution.svg)")
            stats_rows = []
            for device, stats in device_stats:
                stats_rows.append([
                    device, stats["count"], format_number(stats["min"]),
                    format_number(stats["median"]), format_number(stats["mean"]),
                    format_number(stats["max"]), format_number(stats["stdev"]),
                    format_number(stats["cv_pct"]),
                ])
            lines.extend(["", table(["Device", "n", "min", "median", "mean", "max", "σ", "CV (%)"], stats_rows)])

        latency_cases = {name: cases[name] for name in (
            "interconnect-h2d-latency", "interconnect-d2h-latency",
            "interconnect-P2P_intraserver-latency",
        ) if name in cases}
        if latency_cases:
            lines.extend([
                "", "## H2D / D2H / P2P 时延扫点", "",
                "时延单位为 ns；每个负载大小和每个有向端点都来自独立命令。P2P 的反向结果不镜像推导。",
            ])
            if "report-assets/transfer-latency.svg" in asset_paths:
                lines.append("\n![传输时延扫点](report-assets/transfer-latency.svg)")
            latency_rows = []
            for name, case in sorted(latency_cases.items(), key=lambda item: case_sort_key(item[0])):
                for metric in sorted(case.get("metrics", []), key=lambda item: (
                    str(item.get("device", "")), str(item.get("source_device", "")),
                    str(item.get("destination_device", "")), item.get("size_bytes", 0))):
                    endpoint = metric.get("device", "—")
                    if metric.get("source_device") is not None:
                        endpoint = f"{metric.get('source_device')}→{metric.get('destination_device')}"
                    latency_rows.append([
                        CASE_LABELS.get(name, name), endpoint, metric.get("size_bytes", "—"),
                        format_number(metric.get("value"), 6), metric.get("unit", "—"),
                        status_text(case.get("measurement_status", case.get("status"))),
                    ])
            lines.extend(["", table(["Case", "Device/pair", "Bytes", "时延", "单位", "测量"], latency_rows)])

        p2p = cases.get("interconnect-P2P_intraserver")
        if p2p:
            selected_pairs = isinstance(p2p.get("p2p_scope"), dict) and (
                p2p["p2p_scope"].get("mode") == "selected-combinations"
            )
            lines.extend([
                "",
                ("## P2P 覆盖所选逻辑 Device 组合" if selected_pairs
                 else "## P2P 矩阵覆盖全部物理 NPU 对"),
                "",
                ("每个无序组合只按升序执行一次 A→B；未执行的反向 B→A 不推断、不镜像。"
                 if selected_pairs else
                 "每个方向独立统计并独立缩放色标；两幅图的颜色深浅不能跨方向直接比较。"),
            ])
            direction_rows = []
            for direction, label in (("unidirectional", "单向"), ("bidirectional", "双向")):
                metrics = [item for item in p2p.get("metrics", []) if item.get("direction") == direction]
                values = finite_values(metrics)
                stats = numeric_summary(values)
                weakest = min(metrics, key=lambda item: float(item["value"])) if values else {}
                strongest = max(metrics, key=lambda item: float(item["value"])) if values else {}
                direction_rows.append([
                    label, stats["count"], format_number(stats["min"]),
                    format_number(stats["median"]), format_number(stats["mean"]),
                    format_number(stats["max"]),
                    f"{weakest.get('source_device', '—')}→{weakest.get('destination_device', '—')}",
                    f"{strongest.get('source_device', '—')}→{strongest.get('destination_device', '—')}",
                ])
            lines.extend(["", table(["方向", "有向 pair", "min", "median", "mean", "max", "最小 pair", "最大 pair"], direction_rows)])
            for direction in ("unidirectional", "bidirectional"):
                asset = f"report-assets/p2p-{direction}-heatmap.svg"
                if asset in asset_paths:
                    label = "单向" if direction == "unidirectional" else "双向"
                    lines.append(f"\n![P2P {label}带宽矩阵]({asset})")

        hccl = cases.get("interconnect-MPI_intraserver")
        if hccl:
            scope = hccl.get("hccl_scope", {}) if isinstance(hccl.get("hccl_scope"), dict) else {}
            lines.extend([
                "", "## 单节点 MPI/HCCL AllReduce", "",
                "MPI 仅负责启动本机 ranks，实际集合通信由 HCCL 执行。权威值为 hccl_test 的算法带宽，不派生 bus bandwidth。",
                "",
                table(["参数", "值"], [
                    ["collective", scope.get("collective", "all_reduce")],
                    ["datatype / op", f"{scope.get('datatype', 'fp32')} / {scope.get('op', 'sum')}"],
                    ["ranks / Device", f"{scope.get('rank_count', '—')} / {scope.get('selected_device_ids', '—')}"],
                    ["warmup / iterations", f"{scope.get('warmup', '—')} / {scope.get('iterations', '—')}"],
                    ["消息范围", f"{scope.get('min_bytes', '—')}–{scope.get('max_bytes', '—')} Byte，×{scope.get('factor', '—')}"],
                ]),
            ])
            if "report-assets/hccl-allreduce.svg" in asset_paths:
                lines.append("\n![HCCL AllReduce 算法带宽](report-assets/hccl-allreduce.svg)")
            hccl_rows = [[
                item.get("message_size_bytes", "—"), format_number(item.get("avg_time_us"), 6),
                format_number(item.get("value"), 6), item.get("unit", "—"),
                "通过" if item.get("verification_passed") else "失败",
            ] for item in sorted(hccl.get("metrics", []), key=lambda item: item.get("message_size_bytes", 0))]
            lines.extend(["", table(["消息大小 (B)", "平均时延 (us)", "算法带宽", "单位", "正确性"], hccl_rows)])

        capacity = cases.get("main_memory-capacity")
        if capacity:
            capacity_rows = [[
                item.get("card", "—"), item.get("chip", "—"), item.get("device", "—"),
                item.get("scope", "chip"),
                raw_metric_value(item), item.get("unit", "—")
            ] for item in capacity.get("metrics", [])]
            lines.extend([
                "",
                "## HBM 容量覆盖所选 Device 对应的 NPU/chip",
                "",
                "表格逐 chip 保留 `npu-smi` 的 `HBM Capacity(MB)` 原值；不乘二、不改标 MiB，也不隐式聚合为 card 或整机容量。",
                "",
                table(["Card", "Chip", "逻辑 Device", "Scope", "源字段值", "源单位"], capacity_rows),
            ])

    preflight = summary.get("host_preflight", {}).get("result", {}) if isinstance(summary.get("host_preflight"), dict) else {}
    lines.extend([
        "",
        "## 健康、拓扑与证据完整性",
        "",
        table(["证据节点", "结果", "说明/入口"], [
            ["Host inventory", status_text(preflight.get("status", "missing")), f"期望 {len(preflight.get('expected_device_ids', []))}，实际节点 {len(preflight.get('actual_device_node_ids', []))}"],
            ["Host occupancy", status_text(preflight.get("occupancy", {}).get("status", "missing")), f"仅检查所选 Device {preflight.get('occupancy', {}).get('checked_device_ids', [])}"],
            ["外层 summary", "存在", evidence_link(root, "summary.json", "summary.json")],
            ["Toolkit manifest", "存在" if manifest else "缺失", evidence_link(root, "manifest.json", "toolkit-evidence/manifest.json")],
            ["Runner log", "存在" if (root / "runner.log").is_file() else "缺失", evidence_link(root, "runner.log", "runner.log")],
        ]),
    ])
    if manifest:
        health_rows = []
        for phase in ("pre", "post"):
            health = manifest.get("health", {}).get(phase, {})
            coverage = health.get("coverage", {}) if isinstance(health.get("coverage"), dict) else {}
            health_rows.append([
                phase, status_text(health.get("result", "missing")),
                status_text(coverage.get("status", "missing")),
                ", ".join(coverage.get("uncovered", [])) or "无",
                evidence_link(root, "stdout", f"toolkit-evidence/health/{phase}/health.stdout"),
            ])
        lines.extend(["", table(["阶段", "已执行健康项", "能力覆盖", "未覆盖项", "原始输出"], health_rows)])
        lines.append("\n健康 `result=passed` 表示实际执行的 driver/cann/device/hbm 项通过；`coverage=partial` 单独表示某些能力（例如容器内 lost-card diagnosis）未覆盖。")

        diagnosis_rows = []
        for name, diagnosis in sorted(manifest.get("diagnostics", {}).items()):
            diagnosis_rows.append([
                name, status_text(diagnosis.get("status", "missing")),
                diagnosis.get("returncode", "—"),
                evidence_link(root, "stdout", f"toolkit-evidence/diagnostics/{name}/diagnosis.stdout"),
                evidence_link(root, "stderr", f"toolkit-evidence/diagnostics/{name}/diagnosis.stderr"),
            ])
        if diagnosis_rows:
            lines.extend(["", table(["诊断项", "状态", "rc", "stdout", "stderr"], diagnosis_rows)])

    lines.extend([
        "",
        "## 方法与指标口径",
        "",
        "- 算力由 DMI 厂商矩阵乘 microbenchmark 产生；BF16/FP16/FP32 使用 TFLOPS，INT8 使用 TOPS。",
        "- H2D 是 Host→Device，D2H 是 Device→Host，D2D 是同一 Device 内存搬运；带宽与时延是不同指标。",
        "- P2P 的参与层级、pair 方向和扫点策略以 Case 的 `p2p_scope`/`sweep_scope` 为准，反向值不推断。",
        "- HCCL Case 的 GB/s 是 hccl_test `alg_bandwidth`；MPI 负责进程启动，v1 不派生 bus bandwidth。",
        "- 报告中的均值、总体标准差 `σ` 和变异系数 `CV=σ/mean×100%` 仅描述本轮保存的序列，不构成跨轮稳定性结论。",
        "- `partial` 不自动否定连续指标；必须结合测量状态、诊断状态和 error 字段解释。",
        "",
        "## 限制与后续动作",
        "",
    ])
    limitations = []
    if manifest:
        limited_cases = [
            CASE_LABELS.get(name, name)
            for name, case in sorted(cases.items(), key=lambda item: case_sort_key(item[0]))
            if case.get("error")
        ]
        if limited_cases:
            limitations.append(
                "存在限制或错误的 Case：" + "、".join(limited_cases)
                + "；聚合原因见上表，逐命令事实见原始证据索引。"
            )
        health_uncovered = {
            item for health in manifest.get("health", {}).values()
            if isinstance(health, dict)
            for item in (health.get("coverage", {}).get("uncovered", [])
                         if isinstance(health.get("coverage"), dict) else [])
        }
        if health_uncovered:
            limitations.append("健康能力未覆盖：" + ", ".join(sorted(health_uncovered)))
    if summary.get("error"):
        limitations.append(str(summary["error"]))
    if not limitations:
        limitations.append("本轮未记录额外限制；结论仍只适用于 manifest 声明的单机环境和测试参数。")
    lines.extend(f"- {md_escape(item)}" for item in limitations)
    if diagnoses["unsupported"]:
        lines.append("- 若业务需要厂商诊断 PASS，需要取得适用于 A3/910C 的正式阈值配置后重跑；不得自行把连续数值改写为诊断通过。")
    if statuses["failed"]:
        lines.append("- 对失败 Case，先检查本报告链接的 command/stdout/stderr，再决定是否重跑；不得只依据兼容 `[FlagPerf Result]` 行排障。")
    if not statuses["failed"] and measurements["passed"]:
        lines.append("- 若要建立稳定性能基线，应进行多轮独立重复并比较中位数、范围和 CV；单轮结果只证明本轮口径。")

    lines.extend(["", "## 原始证据索引", "", "- " + evidence_link(root, "外层 summary", "summary.json")])
    if manifest:
        lines.append("- " + evidence_link(root, "Toolkit manifest", "toolkit-evidence/manifest.json"))
        lines.append("- " + evidence_link(root, "DMI 版本输出", "toolkit-evidence/environment/dmi-version.stdout"))
        lines.append("- " + evidence_link(root, "DMI 兼容性输出", "toolkit-evidence/environment/dmi-compatibility.stdout"))
        lines.append("- " + evidence_link(root, "设备拓扑", "toolkit-evidence/topology/npu-smi-topology.stdout"))
        for name in sorted(cases, key=case_sort_key):
            case = cases[name]
            metrics_path = f"toolkit-evidence/cases/{name}/metrics.json"
            raw_links = [evidence_link(root, "metrics", metrics_path)]
            records = command_records(case)
            stdout_links: list[str] = []
            stderr_links: list[str] = []
            for record in records:
                stdout = record.get("stdout", {}).get("path") if isinstance(record.get("stdout"), dict) else None
                stderr = record.get("stderr", {}).get("path") if isinstance(record.get("stderr"), dict) else None
                if stdout:
                    stdout_links.append(evidence_link(root, "stdout", "toolkit-evidence/" + str(stdout)))
                if stderr:
                    stderr_links.append(evidence_link(root, "stderr", "toolkit-evidence/" + str(stderr)))
            if stdout_links:
                raw_links.append(stdout_links[0])
            if stderr_links:
                raw_links.append(stderr_links[0])
            if len(records) > 1:
                raw_links.append(f"共 {len(records)} 条命令；完整索引见 metrics")
            lines.append(f"- **{md_escape(CASE_LABELS.get(name, name))}：** " + " · ".join(raw_links))
    lines.extend([
        "",
        "---",
        "",
        f"报告生成器 schema：`{REPORT_SCHEMA_VERSION}`。报告是 JSON 证据的确定性视图；发生冲突时以原始 manifest、command、stdout、stderr 和 SHA-256 为准。",
        "",
    ])
    return "\n".join(lines)


def generate_toolkit_report(result_dir: Path) -> dict[str, Any]:
    root = result_dir.resolve()
    summary = read_json(root / "summary.json")
    assert summary is not None
    summary.setdefault("run_id", root.name)
    manifest = read_json(root / "toolkit-evidence" / "manifest.json", required=False)
    if summary.get("suite") not in (None, "ascend-toolkit") and manifest is None:
        raise ReportError("result is not an ascend-toolkit run")
    if (
        manifest is not None
        and manifest.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS
    ):
        raise ReportError(f"unsupported toolkit manifest schema: {manifest.get('schema_version')}")

    cases = manifest.get("cases", {}) if manifest else {}
    assets: list[dict[str, Any]] = []
    candidates = (
        ("compute-overview.svg", compute_svg(cases)),
        ("d2d-device-distribution.svg", d2d_svg(cases.get("main_memory-bandwidth", {}))),
        ("p2p-unidirectional-heatmap.svg", p2p_svg(cases.get("interconnect-P2P_intraserver", {}), "unidirectional")),
        ("p2p-bidirectional-heatmap.svg", p2p_svg(cases.get("interconnect-P2P_intraserver", {}), "bidirectional")),
        ("transfer-latency.svg", sweep_svg({
            name: cases[name] for name in (
                "interconnect-h2d-latency", "interconnect-d2h-latency",
                "interconnect-P2P_intraserver-latency",
            ) if name in cases
        }, x_key="size_bytes", title="H2D / D2H / P2P 时延扫点", y_label="时延（ns）")),
        ("hccl-allreduce.svg", sweep_svg(
            {"interconnect-MPI_intraserver": cases["interconnect-MPI_intraserver"]}
            if "interconnect-MPI_intraserver" in cases else {},
            x_key="message_size_bytes", title="HCCL AllReduce 算法带宽", y_label="alg_bandwidth（GB/s）",
        )),
    )
    for name, content in candidates:
        reference = write_asset(root, name, content)
        if reference:
            assets.append(reference)

    monitor_assets: dict[tuple[str, int], str] = {}
    for case_name, case in sorted(cases.items(), key=lambda item: case_sort_key(item[0])):
        monitor = case.get("monitor") if isinstance(case.get("monitor"), dict) else {}
        units = monitor.get("units", []) if isinstance(monitor.get("units"), list) else []
        for unit_index, unit in enumerate(units):
            samples = monitor_unit_samples(root, unit)
            if unit.get("target_resource") == "hccs-links":
                svg = hccs_timeline_svg(
                    f"{CASE_LABELS.get(case_name, case_name)} / 监控单元 {unit_index + 1}",
                    samples, unit.get("workload_windows", []),
                )
            else:
                svg = monitor_heatmap_svg(
                    f"{CASE_LABELS.get(case_name, case_name)} / 监控单元 {unit_index + 1}",
                    samples, unit.get("workload_windows", []),
                )
            asset_name = (
                "monitor-" + case_name.removeprefix("computation-").lower()
                + f"-unit-{unit_index + 1}.svg"
            )
            reference = write_asset(root, asset_name, svg)
            if reference:
                assets.append(reference)
                monitor_assets[(case_name, unit_index)] = reference["path"]

    report_path = root / "report.md"
    monitor_report_path = root / "report_monitor.md"
    atomic_write(report_path, render_report(root, summary, manifest, assets))
    atomic_write(
        monitor_report_path,
        render_monitor_report(root, summary, manifest, monitor_assets),
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "path": str(report_path.relative_to(root)),
        "bytes": report_path.stat().st_size,
        "sha256": sha256(report_path),
        "monitor_path": str(monitor_report_path.relative_to(root)),
        "monitor_bytes": monitor_report_path.stat().st_size,
        "monitor_sha256": sha256(monitor_report_path),
        "outputs": {
            "main": artifact_ref(report_path, root),
            "monitor": artifact_ref(monitor_report_path, root),
        },
        "assets": assets,
    }


def generate_and_record(result_dir: Path) -> dict[str, Any]:
    root = result_dir.resolve()
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    assert summary is not None
    original_status = summary.get("status")
    try:
        metadata = generate_toolkit_report(root)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = generate_and_record(args.result_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
