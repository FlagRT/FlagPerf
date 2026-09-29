"""Deterministic Markdown/SVG views of P800 Toolkit evidence, never new measurements."""
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import statistics

from .contract import CASES
from .evidence import index, reference, sha256, write_json


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(map(md, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(md, row)) + " |" for row in rows]


def label(point):
    s = "card " + str(point.get("source", "?"))
    if "destination" in point:
        s += " to " + str(point["destination"])
        s += " / " + point.get("direction", "single-direction")
    if "bytes" in point:
        s += " / " + str(point["bytes"]) + " B"
    if point.get("mode") in ("h2d", "d2h"):
        s += " / " + ("pinned" if point.get("pinned") else "pageable") + " / " + ("async" if point.get("async") else "blocking")
    return s


def bars(title, rows, unit):
    if not rows:
        return None
    width, left, line = 1100, 420, 30
    height = 100 + len(rows)*line
    top = max(v for _, v in rows) or 1
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="13" fill="#263247">',
           f'<text x="20" y="28" font-size="18">{escape(title)} ({escape(unit)})</text>',
           f'<line x1="{left}" y1="50" x2="{left}" y2="{height-30}" stroke="#b7c0ca"/>',
           f'<text x="{left}" y="{height-10}">0</text>']
    for i, (name, value) in enumerate(rows):
        y = 60+i*line
        length = 530*value/top
        svg += [f'<text x="16" y="{y+15}">{escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{length:.3f}" height="20" fill="#3873a5"/>',
                f'<text x="{left+length+8:.3f}" y="{y+15}">{value:.6g}</text>']
    return "\n".join(svg+["</g></svg>\n"])


def timeline(title, samples, key, unit):
    valid = [(i, s["values"][key]) for i, s in enumerate(samples) if s.get("valid") and key in s.get("values", {})]
    high = max((v for _, v in valid), default=1) or 1
    width, height = 1100, 260
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>', '<g font-family="sans-serif" font-size="11" fill="#263247">',
           f'<text x="15" y="25" font-size="16">{escape(title)} ({unit})</text>',
           '<line x1="45" x2="1060" y1="215" y2="215" stroke="#b7c0ca"/>']
    # Break paths at invalid/missing samples. Never connect across telemetry gaps.
    prior = None
    for i, sample in enumerate(samples):
        x = 60 + i*1000/max(1, len(samples)-1)
        val = sample.get("values", {}).get(key) if sample.get("valid") else None
        svg.append(f'<text x="{x:.2f}" y="237" text-anchor="middle">S{i+1:02d}</text>')
        if val is None:
            svg.append(f'<text x="{x:.2f}" y="130" text-anchor="middle" fill="#a12c2c">N/A</text>')
            prior = None
            continue
        y = 210 - float(val)/high*145
        if prior is not None:
            svg.append(f'<line x1="{prior[0]:.2f}" y1="{prior[1]:.2f}" x2="{x:.2f}" y2="{y:.2f}" stroke="#3873a5"/>')
        svg += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#3873a5"/>',
                f'<text x="{x:.2f}" y="{y-9:.2f}" text-anchor="middle">{val:g}</text>']
        prior = (x, y)
    return "\n".join(svg+["</g></svg>\n"])


def generate(root):
    root = Path(root).resolve()
    summary = read(root / "summary.json", {})
    manifest = read(root / "toolkit-evidence/manifest.json", {})
    if summary.get("suite") != "kunlunxin-p800-toolkit" or manifest.get("schema_version") != 1:
        raise ValueError("not a supported P800 Toolkit result")
    cases = manifest.get("cases", {})
    assets = []
    (root / "report-assets").mkdir(exist_ok=True)
    def asset(name, svg):
        if svg is None:
            return None
        p = root / "report-assets" / name
        p.write_text(svg, encoding="utf-8")
        assets.append(reference(p, root))
        return p.relative_to(root).as_posix()
    lines = ["# Kunlunxin P800 Toolkit 测试报告", "", f"运行：{md(summary.get('run_id'))}", "",
             f"总体状态：**{md(summary.get('status'))}**；测量：**{md(summary.get('measurement_status'))}**；监控：**{md(summary.get('monitoring_status'))}**；厂商阈值诊断：**not-supported**。", "",
             "计算使用独立 Toolkit C++ 程序调用 XBLAS；传输调用 XRE。没有导入 PyTorch/Base workload，也没有将官方 fc_effciency 的存在或 help 输出当作测量通过。", "",
             "计时为 CLOCK_MONOTONIC 的原生 API 调用到同步完成，不包含进程启动、分配、预热及正确性回读；它不是纯 kernel 时间。容量为每卡 HBM 查询值。D2D/单向 P2P 不乘二；双向 P2P 单独运行两个并发方向，以实际两份 payload/两方向完成时间计算，含线程协调开销。", "",
             "INT8 使用 XBLAS fc_fusion（INT8 输入和 TGEMM、FP32 输出、maxima=127、alpha=1、beta=0、无 bias、LINEAR 激活）。它不是 INT8→INT32 GemmEx；后者在当前镜像的实测返回参数错误。", "",
             "Pinned 异步模式可能按配置将逻辑 payload 分为多次原生提交，最后统一等待。实际分块大小和 API 次数记录在 metrics 的 async_chunk_bytes / api_calls_per_sample；这是分块端到端带宽，不能与单次整块 API 调用混称。", "",
             "卡 1 已排除。占用卡数据属于 exploratory；本报告不晋级 candidate，也不替代 Base 双卡/八卡 qualification。", "",
             "厂商诊断仅有 xpu-smi 观测，不具备 Ascend DMI 的算力/带宽/信号质量阈值判定。xprofiler 的文件及 help 留档不代表采集了 kernel trace。", ""]
    if summary.get("error"):
        lines += ["执行错误："+md(summary["error"]), ""]
    lines += ["## 测试条件", "", "```json", json.dumps(manifest.get("settings", summary.get("resolved_plan", {}).get("settings", {})), ensure_ascii=False, sort_keys=True, indent=2), "```", "",
              "## 状态矩阵", ""]
    lines += table(["Case", "Measurement", "Monitoring", "Diagnosis", "有效指标", "证据"], [
        [c, cases[c].get("measurement_status"), cases[c].get("monitoring_status"), cases[c].get("diagnosis_status"),
         len(cases[c].get("metrics", [])), f"[metrics](toolkit-evidence/cases/{c}/metrics.json)"] for c in CASES if c in cases])
    for case in CASES:
        if case not in cases:
            continue
        value = cases[case]
        lines += ["", "## "+case, ""]
        metrics = value.get("metrics", [])
        rows = [[label(m.get("point", {})), m.get("repetition", "query"), m["field"], f"{m['value']:.8g}", m["unit"],
                 m.get("sample_count", "query"), f"{m.get('sample_cv', 0):.4%}" if m.get("sample_cv") is not None else "N/A",
                 f"[stdout](toolkit-evidence/{m['source']})"] for m in metrics]
        if rows:
            lines += table(["设备/模式/载荷", "重复", "字段", "值", "单位", "样本数", "样本 CV", "原始来源"], rows)
            for unit in sorted({m["unit"] for m in metrics}):
                chart = asset(case+"-"+unit.replace("/", "-")+".svg", bars(case,
                    [(label(m.get("point", {}))+(" / "+m["field"].split("/")[-1] if case == "main_memory-capacity" else "")
                      +" / R"+str(m.get("repetition", "Q")), m["value"]) for m in metrics if m["unit"] == unit], unit))
                if chart:
                    lines += ["", f"![{case}]({chart})", ""]
        else:
            lines += ["没有有效性能指标；不得从退出码、工具存在性或邻近设备结果推断数值。", ""]
        groups = defaultdict(list)
        if case == "main_memory-bandwidth":
            for target in value.get("targets", []):
                raw_path = root / "toolkit-evidence/cases" / case / target["target"] / "samples.json"
                raw = read(raw_path, {})
                times = raw.get("samples_ns", [])
                if times:
                    point = target["point"]
                    rows = [(f"sample {i+1}", point["bytes"]/ns) for i, ns in enumerate(times[:25])]
                    name = case+"-"+target["target"].replace("/", "-")+"-samples.svg"
                    chart = asset(name, bars("D2D first 25 measured samples / "+label(point), rows, "GB/s"))
                    lines += ["", "前 25 个实测 D2D 样本（仅图表截取，完整样本仍保留；载荷和参考厂商工具的固定工作集不同）：", "",
                              f"![D2D samples]({chart})", "", f"[完整 samples.json]({raw_path.relative_to(root).as_posix()})", ""]
        for m in metrics:
            groups[(label(m.get("point", {})), m["field"], m["unit"])].append(m["value"])
        if groups:
            lines += ["", "重复组（统计各独立进程的结果；单次内部样本不代替五次重复）：", ""]
            group_rows = []
            for (name, field, unit), values in sorted(groups.items()):
                avg = statistics.mean(values)
                cv = statistics.pstdev(values)/abs(avg) if len(values)>1 and avg else None
                group_rows.append([name, field, len(values), f"{statistics.median(values):.8g}", unit,
                                   f"{cv:.4%}" if cv is not None else "N/A",
                                   "insufficient-repeats" if len(values)<5 else "unstable" if cv is not None and cv>.05 else "stable-observed"])
            lines += table(["点", "字段", "独立重复", "中位数", "单位", "组 CV", "稳定性"], group_rows)
        for target in value.get("targets", []):
            if target.get("error"):
                lines += ["", "- "+md(target["target"])+"："+md(target["error"])]
    lines += ["", "## 生命周期和溯源", "", *[f"- {key}：{md(summary.get(key, 'not-run'))}" for key in ("cleanup_status", "postflight_status", "lease_released", "qualification_status")], "",
              "- [Manifest](toolkit-evidence/manifest.json)", "- [Native tool provenance](toolkit-evidence/provenance.json)",
              "- [Code identity](code-identity.json)", "- [Image identity](image-identity.json)", "- [Monitor report](report_monitor.md)",
              "- [Diagnosis coverage](toolkit-evidence/diagnostics/coverage.json)", "- [SHA256 index](sha256-index.json)", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    raw_samples = root / "toolkit-evidence/monitor/samples.jsonl"
    samples = [json.loads(x) for x in raw_samples.read_text().splitlines()] if raw_samples.is_file() else []
    monitor_lines = ["# P800 Toolkit 同期监控报告", "", "每个样本保留物理卡、有效性、时间与原值；缺失为 N/A，不补零。曲线只连相邻有效样本。", "",
                     "- [原始命令 JSONL](toolkit-evidence/monitor/samples.raw.jsonl)", "- [结构化 JSONL](toolkit-evidence/monitor/samples.jsonl)", "",
                     "## 逐测试点监控覆盖", ""]
    monitor_lines += table(["Case/target", "状态", "窗口内完整样本", "说明"], [
        [c+"/"+t["target"], t.get("monitoring_status"), json.dumps(t.get("monitor_sample_counts", {}), sort_keys=True),
         t.get("monitor_scope", "每目标至少 10 个完整落入真实 measurement 窗口的有效样本")]
        for c in CASES if c in cases for t in cases[c].get("targets", [])])
    for physical in sorted({s.get("physical_id") for s in samples if s.get("physical_id") is not None}):
        selected = [s for s in samples if s.get("physical_id") == physical]
        monitor_lines += ["", f"## 物理卡 {physical}", ""]
        for offset in range(0, len(selected), 24):
            group = selected[offset:offset+24]
            for key, unit in (("utilization_percent", "%"), ("used_memory_mib", "MiB"),
                              ("temperature_c", "C"), ("power_w", "W")):
                chart = asset(f"monitor-card-{physical}-{key}-{offset:05d}.svg",
                              timeline(f"Physical {physical} / samples {offset+1}-{offset+len(group)} / {key}", group, key, unit))
                monitor_lines += [f"![{key}]({chart})", ""]
        monitor_lines += table(["样本", "开始偏移 s", "结束偏移 s", "有效", "利用率 %", "显存 MiB", "温度 C", "功率 W", "错误"], [
            [i+1, f"{s['started_offset_s']:.6f}", f"{s['finished_offset_s']:.6f}", s["valid"],
             *[s.get("values", {}).get(k, "N/A") for k in ("utilization_percent", "used_memory_mib", "temperature_c", "power_w")], s.get("error", "")]
            for i, s in enumerate(selected)])
    (root / "report_monitor.md").write_text("\n".join(monitor_lines)+"\n", encoding="utf-8")
    return {"schema_version": 1, "status": "passed", "generator_sha256": sha256(Path(__file__)), "outputs": {"main": reference(root / "report.md", root),
            "monitor": reference(root / "report_monitor.md", root)}, "assets": assets}


def generate_and_record(root):
    root = Path(root)
    metadata = generate(root)
    summary = read(root / "summary.json", {})
    summary["report_generation"] = metadata
    write_json(root / "summary.json", summary)
    index(root)
    return metadata
