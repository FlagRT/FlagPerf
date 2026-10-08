#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Deterministic root-level acceptance reports for Base evidence roots.

generate_benchmark_report.py renders one stored run; this module aggregates
every stored run under one or more evidence roots into a single acceptance
report: per-group medians and sample CV across recorded repetitions,
dual-layer (measurement/qualification) verdicts, and evidence-linked
tables. Like every Base report it is a deterministic view of stored JSON
evidence — it never recomputes measurements and renders gaps instead of
guessing when evidence is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable

from benchmark_report_schema import cell, table

ACCEPTANCE_SCHEMA_VERSION = 1
SUPPORTED_SUMMARY_SCHEMA_VERSIONS = (1, 2, 3)
SUPPORTED_RESULT_SCHEMA_VERSIONS = (1,)
GROUP_SUFFIX = re.compile(r"^(?P<base>.+?)-q(?P<rep>\d+)$")
STATUS_SEVERITY = {"passed": 0, "partial": 1, "failed": 2}
ARTIFACT_CHECKLIST = (
    "summary.json", "benchmark-result.json", "report.md", "report_monitor.md",
    "report-assets", "cleanup.json", "lease.json",
)
PROTOCOL_TEXT = """- 报告输入是证据根内已落盘的 summary.json 与 benchmark-result.json；生成器不重算测量，只聚合已记录的值。
- 每个组由同一证据根内共享组名（去掉重复序号 `-q<N>` 或单次后缀 `-final`）的 run 组成；smoke run 自成一组，仅作信息展示。
- 组内每个 run 取最慢 rank 值（metric 记录按 rank 取最小值）；组指标为中位数；离散度为样本标准差除以中位数（ddof=1）的百分比。
- 资格判定引用每个 run 自带 qualification 块的 max_cv_percent / repetitions_required：重复数不足或 CV 达到上限判为未通过；窗口类门槛由 run.py 在测量时强制执行并记录在 summary 的 qualification_error，本生成器原样呈现该错误而不另立窗口标准；capacity 模式不适用吞吐窗口；smoke 模式预期返回部分状态。
- 分层结论：测量层为 measurement_status（含 correctness），监控层为 monitoring_status，最终状态为 run 的 status 字段；多层取组内最差值。健康标注（如历史 ECC 降级）单独呈现，不与测量结论合并。
- 报告是证据的确定性阅读视图；发生冲突时以原始 summary.json、benchmark-result.json 及其 SHA-256 为准。"""


class ReportError(RuntimeError):
    """Raised when stored acceptance evidence cannot be rendered safely."""


def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ReportError(f"required acceptance input does not exist: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"JSON artifact must contain an object: {path}")
    return value


def parse_group(label: str) -> tuple[str, int | None]:
    match = GROUP_SUFFIX.match(label)
    if match:
        return match.group("base"), int(match.group("rep"))
    if label.endswith("-final"):
        return label[: -len("-final")], None
    return label, None


def worst_status(values: Iterable[str | None]) -> str:
    worst, severity = "not recorded", -1
    for value in values:
        if not value:
            continue
        level = STATUS_SEVERITY.get(value, 3)
        if level > severity:
            worst, severity = value, level
    return worst


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def domain_of(case: str) -> str:
    lowered = case.lower()
    if lowered.startswith("computation-"):
        return "算力"
    if lowered.startswith("interconnect-h2d") or lowered.startswith("interconnect-d2h"):
        return "传输"
    if lowered.startswith("interconnect-mpi") or lowered.startswith("interconnect-p2p"):
        return "通信"
    if lowered.startswith("main_memory-"):
        return "内存"
    return "其他"


def load_root(root: Path) -> dict[str, Any]:
    runs_dir = root / "base-runs"
    if not runs_dir.is_dir():
        raise ReportError(f"evidence root has no base-runs directory: {root}")
    runs: list[dict[str, Any]] = []
    gaps: list[str] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = read_json(run_dir / "summary.json", required=False)
        if summary is None:
            gaps.append(run_dir.name)
            continue
        result = read_json(run_dir / "benchmark-result.json", required=False) or {}
        group, rep = parse_group(run_dir.name)
        runs.append({
            "label": run_dir.name, "path": run_dir, "group": group, "rep": rep,
            "summary": summary, "result": result,
        })
    if not runs and not gaps:
        raise ReportError(f"evidence root contains no runs: {root}")
    return {"name": root.name, "path": root, "runs": runs, "gaps": gaps}


def run_value(record: dict[str, Any]) -> tuple[float, str] | None:
    metrics = record["result"].get("metrics") or []
    entries = [m for m in metrics if isinstance(m.get("value"), (int, float))]
    if not entries:
        return None
    slowest = min(entries, key=lambda m: m["value"])
    return float(slowest["value"]), str(slowest.get("unit", ""))


def run_window(record: dict[str, Any]) -> float | None:
    metrics = record["result"].get("metrics") or []
    windows = [m.get("elapsed_seconds") for m in metrics
               if isinstance(m.get("elapsed_seconds"), (int, float))]
    return min(windows) if windows else None


def qualification_of(record: dict[str, Any]) -> dict[str, Any]:
    block = record["summary"].get("qualification")
    return block if isinstance(block, dict) else {}


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    units: set[str] = set()
    windows: list[float] = []
    for record in records:
        entry = run_value(record)
        if entry is None:
            continue
        values.append(entry[0])
        units.add(entry[1])
        window = run_window(record)
        if window is not None:
            windows.append(window)
    block = qualification_of(records[0])
    mode = str(block.get("mode", "unknown"))
    reps_required = block.get("repetitions_required")
    max_cv = block.get("max_cv_percent")
    unit = sorted(units)[0] if len(units) == 1 else "|".join(sorted(units))
    errors = sorted({str(record["summary"].get("qualification_error")) for record in records
                     if record["summary"].get("qualification_error")})
    stats: dict[str, Any] = {
        "n": len(values), "unit": unit, "mode": mode,
        "min_window_s": min(windows) if windows else None,
        "repetitions_required": reps_required, "max_cv_percent": max_cv,
    }
    if len(values) >= 2:
        median = statistics.median(values)
        stats["median"] = median
        stats["cv_percent"] = (statistics.stdev(values) / median * 100) if median else None
    elif len(values) == 1:
        stats["median"] = values[0]
        stats["cv_percent"] = None
    else:
        stats["median"] = None
        stats["cv_percent"] = None
    if errors:
        stats["verdict"] = "harness 资格错误(" + "; ".join(errors) + ")"
    elif mode == "smoke":
        stats["verdict"] = "smoke（预期部分状态）"
    elif mode == "capacity":
        stats["verdict"] = "single-shot（窗口不适用）"
    elif not values:
        stats["verdict"] = "no-measurement"
    elif reps_required is not None and len(values) < int(reps_required):
        stats["verdict"] = f"重复不足({len(values)}/{reps_required})"
    elif stats["cv_percent"] is None or (
        max_cv is not None and stats["cv_percent"] >= float(max_cv)
    ):
        stats["verdict"] = "unstable"
    else:
        stats["verdict"] = "passed"
    return stats


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    loaded = [load_root(root) for root in roots]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for data in loaded:
        for record in data["runs"]:
            groups.setdefault((data["name"], record["group"]), []).append(record)
    rendered = []
    for (root_name, group), records in sorted(groups.items()):
        records.sort(key=lambda r: (r["rep"] is None, r["rep"] or 0, r["label"]))
        summary = summarize_group(records)
        case = str(records[0]["summary"].get("case", "unknown"))
        rendered.append({
            "root": root_name, "group": group, "records": records,
            "case": case, "domain": domain_of(case), "stats": summary,
        })
    for data in loaded:
        data["groups"] = [g for g in rendered if g["root"] == data["name"]]
    return loaded


def rel(target: Path, output: Path) -> str:
    return os.path.relpath(target, output.parent)


def link(text: str, target: Path, output: Path) -> str:
    return f"[{text}]({rel(target, output)})"


def env_section(data: dict[str, Any], output: Path) -> list[str]:
    lines = [f"### 证据根 {data['name']}", ""]
    starts, finishes = [], []
    images: set[tuple[str, str]] = set()
    heads: set[str] = set()
    devices: set[Any] = set()
    profiles: set[tuple[str, str, str]] = set()
    for record in data["runs"]:
        summary = record["summary"]
        if summary.get("started_at"):
            starts.append(str(summary["started_at"]))
        if summary.get("finished_at"):
            finishes.append(str(summary["finished_at"]))
        runtime = summary.get("runtime") or {}
        if runtime.get("image"):
            images.add((str(runtime.get("image")), str(runtime.get("image_id", ""))))
        lock = runtime.get("lock") or {}
        profile = lock.get("runtime_profile")
        manifest = lock.get("image_manifest") or {}
        if profile:
            profiles.add((str(profile), str(manifest.get("release_stage", "")),
                          str(manifest.get("validated"))))
        for binding in summary.get("device_bindings") or []:
            if binding.get("host_physical_id") is not None:
                devices.add(binding["host_physical_id"])
        identity = read_json(record["path"] / "code-identity.json", required=False)
        if identity and identity.get("git_head"):
            heads.add(str(identity["git_head"]))
    window = "not recorded"
    if starts and finishes:
        window = f"{min(starts)} → {max(finishes)}"
    lines += [
        "| 项 | 值 |", "|---|---|",
        f"| 运行窗口（UTC，源自 summary started/finished） | {window} |",
        f"| run 数 / 缺 summary 目录 | {len(data['runs'])} / {len(data['gaps'])} |",
        f"| 镜像 | {'; '.join(sorted(f'{i} ({d})' for i, d in images)) or 'not recorded'} |",
        f"| git head（code-identity） | {'; '.join(sorted(heads)) or 'not recorded'} |",
        f"| runtime profile（release_stage, validated） | {'; '.join(sorted(f'{p} ({s}, {v})' for p, s, v in profiles)) or 'not recorded'} |",
        f"| 参与物理设备 | {', '.join(str(d) for d in sorted(devices)) or 'not recorded'} |",
        "",
    ]
    return lines


def conclusion_row(group: dict[str, Any], output: Path) -> list[str]:
    stats = group["stats"]
    records = group["records"]
    case_link = link(group["records"][0]["label"], group["records"][0]["path"], output)
    return [
        cell(group["root"]), cell(group["group"]), cell(case_link), cell(group["case"]),
        cell(str(stats["n"])), cell(fmt(stats["median"])), cell(stats["unit"]),
        cell(fmt(stats["cv_percent"])),
        cell(worst_status(r["summary"].get("measurement_status") for r in records)),
        cell(worst_status(r["summary"].get("monitoring_status") for r in records)),
        cell(stats["verdict"]),
        cell(worst_status(r["summary"].get("status") for r in records)),
        cell(fmt(stats["min_window_s"], 3)),
    ]


def detail_rows(group: dict[str, Any], output: Path) -> list[str]:
    stats = group["stats"]
    lines = [f"### {group['group']}（{group['case']}，证据根 {group['root']}）", "",
             "| run | 值（最慢 rank） | 单位 | 窗口 s | 测量 | 健康 |", "|---|---:|---|---:|---|---|"]
    for record in group["records"]:
        entry = run_value(record)
        value = fmt(entry[0]) if entry else "n/a"
        unit = entry[1] if entry else "n/a"
        lines.append("| " + " | ".join([
            cell(link(record["label"], record["path"], output)), cell(value),
            cell(unit), cell(fmt(run_window(record), 3)),
            cell(str(record["summary"].get("measurement_status", "not recorded"))),
            cell(str(record["summary"].get("health_status", "not recorded"))),
        ]) + " |")
    lines += [
        f"组中位数 {fmt(stats['median'])} {stats['unit']}，CV {fmt(stats['cv_percent'])}%，"
        f"资格判定 {stats['verdict']}（n={stats['n']}）。",
        "",
    ]
    return lines


def capacity_rows(group: dict[str, Any], output: Path) -> list[str]:
    lines = ["| run | 保持 MiB | 报告 GB | GiB | 分配次数 | 测量 | 健康 |", "|---|---:|---:|---:|---:|---|---|"]
    for record in group["records"]:
        metrics = record["result"].get("metrics") or []
        metric = metrics[0] if metrics else {}
        summary = record["summary"]
        lines.append("| " + " | ".join([
            cell(link(record["label"], record["path"], output)),
            cell(str(metric.get("held_mib", "not recorded"))),
            cell(fmt(metric.get("value"))), cell(fmt(metric.get("value_gib_s"))),
            cell(str(metric.get("allocation_count", "not recorded"))),
            cell(str(summary.get("measurement_status", "not recorded"))),
            cell(str(summary.get("health_status", "not recorded"))),
        ]) + " |")
    lines.append("")
    return lines


def integrity_section(data: dict[str, Any], output: Path) -> list[str]:
    lines = [f"### {data['name']}", ""]
    missing: list[str] = []
    for record in data["runs"]:
        absent = [name for name in ARTIFACT_CHECKLIST
                  if not (record["path"] / name).exists()]
        if absent:
            missing.append(f"{record['label']} 缺少 {', '.join(absent)}")
    lines.append(f"- run 总数 {len(data['runs'])}；缺 summary.json 的目录 "
                 f"{len(data['gaps'])}{'：' + ', '.join(data['gaps']) if data['gaps'] else ''}。")
    if missing:
        lines.append("- 工件缺口：")
        lines += [f"  - {item}（{link(item.split(' ')[0], data['path'] / 'base-runs' / item.split(' ')[0], output)}）"
                  for item in missing]
    else:
        lines.append(f"- 工件清单核对：全部 run 具备 {'、'.join(ARTIFACT_CHECKLIST)}。")
    for extra in ("control", "failed-attempts"):
        target = data["path"] / extra
        if target.is_dir():
            entries = sorted(p.name for p in target.iterdir())
            lines.append(f"- {extra}/：{len(entries)} 项（{', '.join(entries[:8])}"
                         f"{'，…' if len(entries) > 8 else ''}）。")
    lines.append("")
    return lines


def limitation_flags(groups: list[dict[str, Any]], output: Path) -> list[str]:
    lines: list[str] = []
    for group in groups:
        stats = group["stats"]
        records = group["records"]
        where = link(f"{group['root']}/{group['group']}",
                     group["records"][0]["path"], output)
        if stats["mode"] not in ("smoke", "capacity") and stats["verdict"] != "passed":
            lines.append(f"- 未达资格：{where} 判定 {stats['verdict']}"
                         f"（CV {fmt(stats['cv_percent'])}%）。")
        elif stats["mode"] in ("qualification", "measured") \
                and stats["cv_percent"] is not None \
                and stats["cv_percent"] >= 4.0:
            lines.append(f"- 贴近阈值：{where} CV {fmt(stats['cv_percent'])}%，"
                         f"上限 {stats['max_cv_percent']}%。")
        health = worst_status(r["summary"].get("health_status") for r in records)
        if health == "degraded":
            lines.append(f"- 健康标注：{where} 含 degraded 设备（历史不可纠正 ECC 政策保留），"
                         f"正式资格需注明。")
        for record in records:
            if record["summary"].get("cleanup_status") not in (None, "passed") \
                    or record["summary"].get("lease_released") is not True:
                lines.append(f"- 清理异常：{link(record['label'], record['path'], output)}"
                             f" cleanup={record['summary'].get('cleanup_status')}"
                             f" lease_released={record['summary'].get('lease_released')}。")
    return lines


def render(loaded: list[dict[str, Any]], output: Path) -> str:
    groups = [g for data in loaded for g in data["groups"]]
    total_runs = sum(len(data["runs"]) for data in loaded)
    measured = sum(
        1 for g in groups for r in g["records"]
        if r["summary"].get("measurement_status") == "passed"
    )
    qualified = sum(1 for g in groups if g["stats"]["verdict"] == "passed")
    lines: list[str] = ["# FlagPerf Base Acceptance Report", ""]
    lines += ["## 技术摘要", "",
              f"- 证据根 {len(loaded)} 个（{'、'.join(data['name'] for data in loaded)}），"
              f"run 总数 {total_runs}，测量通过 {measured}。", ""]
    lines += table(["证据根", "组数", "资格通过组", "smoke/单次组", "未达资格组"], [
        [cell(data["name"]), cell(str(len(data["groups"]))),
         cell(str(sum(1 for g in data["groups"] if g["stats"]["verdict"] == "passed"))),
         cell(str(sum(1 for g in data["groups"] if g["stats"]["mode"] in ("smoke", "capacity")))),
         cell(str(sum(1 for g in data["groups"]
                      if g["stats"]["mode"] not in ("smoke", "capacity")
                      and g["stats"]["verdict"] != "passed")))]
        for data in loaded])
    lines += ["## 环境与运行边界", ""]
    for data in loaded:
        lines += env_section(data, output)
    lines += ["## 测量与资格是两个独立结论", "",
              "分层：测量（measurement_status，含 correctness）、监控（monitoring_status）、"
              "资格（组内中位数 CV、重复数与窗口，引用 run 自带 qualification 块）、"
              "最终（run status）。多层取组内最差值；健康标注单独呈现。", ""]
    lines += table(
        ["证据根", "组", "run", "case", "n", "中位数", "单位", "CV%", "测量", "监控", "资格", "最终", "窗口 s"],
        [conclusion_row(group, output) for group in groups])
    for domain, title in (("算力", "算力明细"), ("传输", "传输明细（H2D / D2H）"),
                          ("通信", "通信明细（MPI / P2P）"), ("内存", "设备内存带宽与容量")):
        subset = [g for g in groups if g["domain"] == domain]
        if not subset:
            continue
        lines += [f"## {title}", ""]
        for group in subset:
            if group["stats"]["mode"] == "capacity":
                lines += capacity_rows(group, output)
            else:
                lines += detail_rows(group, output)
    lines += ["## 健康、拓扑与证据完整性", ""]
    for data in loaded:
        lines += integrity_section(data, output)
    health_counts: dict[str, int] = {}
    for group in groups:
        for record in group["records"]:
            health = str(record["summary"].get("health_status", "not recorded"))
            health_counts[health] = health_counts.get(health, 0) + 1
    lines += ["健康分布（按 run）：",
              ", ".join(f"{key} {value}" for key, value in sorted(health_counts.items())) or "not recorded", ""]
    lines += ["## 方法与指标口径", "", PROTOCOL_TEXT, ""]
    lines += ["## 限制与后续动作", ""]
    flags = limitation_flags(groups, output)
    lines += (flags or ["- 无数据驱动的限制标记。"])
    lines.append("")
    lines += ["## 原始证据索引", ""]
    for data in loaded:
        lines.append(f"- 证据根 {data['name']}：{link(data['name'], data['path'], output)}"
                     f"（base-runs {len(data['runs'])} 个 run"
                     f"{'，含 control/ 审计与 failed-attempts/ 留证' if (data['path'] / 'control').is_dir() or (data['path'] / 'failed-attempts').is_dir() else ''}）。")
    lines += ["- 单 run 报告：各 run 目录内 report.md 与 report_monitor.md 由 "
              "`run.py report` 按原始证据再生。", ""]
    lines.append(f"Acceptance 报告生成器 schema：`{ACCEPTANCE_SCHEMA_VERSION}`。"
                 "报告是证据根内 JSON 的确定性视图；发生冲突时以原始 "
                 "summary.json、benchmark-result.json 及其 SHA-256 为准。")
    lines.append("")
    return "\n".join(lines)


def generate_acceptance_report(roots: list[Path], output: Path) -> dict[str, Any]:
    for root in roots:
        if not root.is_dir():
            raise ReportError(f"evidence root does not exist: {root}")
    loaded = collect(roots)
    text = render(loaded, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    groups = [g for data in loaded for g in data["groups"]]
    gaps = sum(len(data["gaps"]) for data in loaded)
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "status": "passed" if gaps == 0 else "passed-with-gaps",
        "roots": [str(data["path"]) for data in loaded],
        "runs": sum(len(data["runs"]) for data in loaded),
        "groups": len(groups),
        "qualified_groups": sum(1 for g in groups if g["stats"]["verdict"] == "passed"),
        "missing_summaries": gaps,
        "output": str(output),
        "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def generate_and_record(roots: list[Path], output: Path | None = None) -> dict[str, Any]:
    destination = output or (roots[0] / "acceptance-report.md")
    return generate_acceptance_report(roots, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_base_acceptance_report",
        description="Aggregate stored Base evidence roots into one acceptance report.",
    )
    parser.add_argument("--evidence-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        metadata = generate_and_record(
            [root.expanduser().resolve() for root in args.evidence_root],
            args.output.expanduser().resolve() if args.output else None,
        )
    except ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if metadata["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
