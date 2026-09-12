# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Read-only compatibility for existing schema-1 evidence; no execution path."""
import hashlib
import json

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def report(root):
    manifest = read(root / "artifacts.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact schema")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"missing or corrupt evidence: {relative}")
    summary = read(root / "summary.json")
    if summary.get("schema_version") != 1:
        raise ValueError("unsupported operation summary schema")
    lines = ["# Ascend abs 实验报告", "", f"状态：{summary['status']}", "",
             "协议：operation-abs-v1。单 Device；CPU FP64 独立参考；两种注册路径不是两个独立 Oracle。", "",
             "## 输入与运行身份", "", "```json",
             json.dumps({k: summary.get(k) for k in ("input", "case_config", "runtime", "selection")},
                        ensure_ascii=False, indent=2), "```", "",
             "## 结果", "", "| 路径 | 正确性 | 路由 | 每次调用中位数 ns | 范围 ns |", "|---|---|---|---:|---|"]
    for library, item in summary.get("paths", {}).items():
        timing = item.get("measurement", {})
        lines.append(f"| {library} | {timing.get('correctness_status', 'not-run')} | "
                     f"{item.get('routing', {}).get('status', 'not-run')} | {timing.get('median_ns', '—')} | "
                     f"{timing.get('min_ns', '—')}–{timing.get('max_ns', '—')} |")
    implementations = [p.get("routing", {}).get("implementation") for p in summary.get("paths", {}).values()]
    if len(implementations) == 2 and all(implementations):
        ids = [{(c['path'], c['sha256'], c['function']) for c in cs} for cs in implementations]
        lines += ["", "两条路径观察到相同 FlagGems abs 实现。" if ids[0] == ids[1]
                  else "两条路径观察到不同实现；具体证据见下方结构化结果。"]
    elif len(summary.get("paths", {})) == 2:
        backends = [p.get("routing", {}).get("backend", "unknown") for p in summary["paths"].values()]
        lines += ["", "观察到的 Backend：" + " / ".join(backends) + "。原生 CANN 设备 kernel 名称未采集。"]
    lines += ["", "## 计时与证据边界", "",
              "同步后的 perf_counter_ns 批次计时包含 Python 提交、输出分配和设备执行；不包含输入搬运、CPU 参考、路由跟踪与报告生成。",
              "首次调用单列，可能受到已有编译缓存影响，不代表冷编译时间。没有测量纯 kernel 时间或逐算子硬件遥测。",
              "每轮均校验实际输出；有限值精确比较，特殊值检查 NaN 类别、无穷符号及零符号。NaN payload 不作保证。", "",
              "## 逐路径证据", "", "```json", json.dumps(summary.get("paths", {}), ensure_ascii=False, indent=2), "```",
              "", "## 复现", "", "在原记录指定的匹配镜像和空闲设备上执行；输入文件以 SHA-256 校验。", "",
              "```bash", summary.get("reproduce", "未生成"), "```", "",
              "## 生命周期", "", "```json", json.dumps({k: summary.get(k) for k in
                  ("execution_status", "postflight_status", "error")}, ensure_ascii=False, indent=2), "```", "",
              "## 原始证据", ""]
    lines += [f"- [{p}]({p}) — `{h}`" for p, h in sorted(manifest["files"].items())]
    text = "\n".join(lines) + "\n"
    temporary = root / "report.md.tmp"
    temporary.write_text(text)
    temporary.replace(root / "report.md")
    return root / "report.md"
