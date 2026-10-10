"""Deterministic Markdown/SVG views of P800 Toolkit evidence, never new measurements.

报告是 JSON 证据的确定性视图；章节结构对齐 Ascend Toolkit 实验报告的十二节信息架构
（技术摘要/环境与运行边界/测量与诊断双层结论/分域明细/健康与证据完整性/方法口径/
限制与后续动作/原始证据索引）。章节中的事实陈述由 manifest 状态推导，不引入新的判定。
"""
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import statistics

from .contract import CASES
from .evidence import index, reference, sha256, write_json

REPORT_SCHEMA_VERSION = 3

STATUS_TEXT = {"passed": "通过", "partial": "部分完成", "failed": "失败",
               "not-run": "未执行", "unsupported": "不支持", "missing": "缺失"}

CASE_DISPLAY = {
    "computation-BF16": "BF16 算力", "computation-FP16": "FP16 算力",
    "computation-FP32": "FP32 算力", "computation-INT8": "INT8 算力",
    "main_memory-bandwidth": "设备内存带宽 / D2D 拷贝吞吐", "main_memory-capacity": "HBM 容量",
    "interconnect-h2d": "H2D 带宽", "interconnect-d2h": "D2H 带宽",
    "interconnect-h2d-latency": "H2D 时延", "interconnect-d2h-latency": "D2H 时延",
    "interconnect-P2P_intraserver": "单机 P2P 带宽",
    "interconnect-P2P_intraserver-latency": "单机 P2P 时延",
}

# 正文分域章节；顺序对齐 Ascend 报告（算力 → 数据搬运 → 时延扫点 → P2P 覆盖 → 带宽与容量）。
FAMILIES = (
    ("computation", "算力：XBLAS 原生 GEMM",
     "算力报告指定规模下的单卡原生 GEMM 实测吞吐；独立 Toolkit C++ 程序直接调用 XBLAS，不经过 PyTorch，正确性回读不计时。"),
    ("transfer", "H2D 与 D2H：主机与设备间两个方向",
     "H2D 与 D2H 回答主机与设备之间两个方向的数据搬运问题；pinned 与 pageable、blocking 与 async 逐组合覆盖。"),
    ("latency", "H2D / D2H / P2P 时延扫点",
     "时延扫点测量小载荷单方向 API 调用至完成同步的主机耗时；逐载荷保留原值，不聚合。"),
    ("p2p", "单机 P2P 覆盖所选物理卡组合",
     "P2P 覆盖所选物理卡的全部无序 pair；双向单独运行两个并发方向，以实际两份 payload/两方向完成时间计算，含线程协调开销。"),
    ("memory", "设备内存带宽与 HBM 容量",
     "设备内存读写带宽与 runtime API 拷贝吞吐按不同口径分别报告；容量为每卡 HBM 查询值，逐卡原值保留、不聚合。"),
)

D2D_TITLES = {
    "d2d-kernel": "Scopy 设备内存读写带宽（2× payload）",
    "d2d": "XRE runtime memcpy 拷贝吞吐（1× payload）",
}
D2D_EXPLANATION = (
    "D2D 两种口径：Scopy（d2d-kernel）以 2×payload/耗时统计读写合计，作为设备内存带宽的实测参考；"
    "XRE runtime memcpy（d2d）以 1×payload/耗时统计 API 拷贝吞吐，供使用该 API 的应用参考。"
    "两者均包含调用至完成同步的耗时；runtime memcpy 吞吐不代表 HBM 带宽上限，Scopy 也不是理论峰值。"
    "分别按设备、载荷、模式和字段统计重复结果，禁止跨口径合并范围或中位数。"
    "两者的差距不能仅用 1×/2×计数解释；具体瓶颈需另行诊断。"
)


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return ["| " + " | ".join(map(md, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(md, row)) + " |" for row in rows]


def status_text(value):
    if value is None or value == "":
        return "未记录"
    return STATUS_TEXT.get(str(value), str(value))


def human_ts(value):
    text = str(value or "").replace("T", " ").split(".")[0].rstrip("Z")
    return text + " UTC" if text else "未记录"


def fmt_duration(seconds):
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if total < 60:
        return f"{total:.2f} s"
    return f"{int(total // 60)} min {total - 60 * int(total // 60):.1f} s"


def case_display(case):
    return CASE_DISPLAY.get(case, case)


def family_of(case):
    if case.endswith("-latency"):
        return "latency"
    if case.startswith("computation-"):
        return "computation"
    if case == "interconnect-P2P_intraserver":
        return "p2p"
    if case.startswith("interconnect-"):
        return "transfer"
    if case.startswith("main_memory-"):
        return "memory"
    return "other"


def metric_summary(metrics):
    if not metrics:
        return "无有效指标"
    # Pool only repetitions of the same device/mode/payload/field contract.
    groups = defaultdict(list)
    for entry in metrics:
        point = json.dumps(entry.get("point", {}), sort_keys=True)
        groups[(point, entry["field"], entry["unit"])].append(float(entry["value"]))
    parts = []
    for (encoded_point, field, unit), values in sorted(groups.items()):
        point = json.loads(encoded_point)
        scope = label(point)
        if point.get("mode") not in (None, "d2d", "d2d-kernel"):
            scope += " / " + point["mode"]
        if point.get("dtype"):
            scope += " / " + point["dtype"]
        parts.append(f"{scope} / {field}：n={len(values)}，"
                     f"{min(values):.3f}–{max(values):.3f} {unit}，"
                     f"中位数 {statistics.median(values):.3f}")
    return "<br>".join(parts)


def case_scope(case, value):
    targets = value.get("targets", [])
    if case.endswith("-latency"):
        return f"{len(targets)} 个扫点目标"
    if case == "interconnect-P2P_intraserver":
        return f"{len(targets)} 个有向 pair 目标"
    return f"{len(targets)} 个厂商执行目标"


def case_limits(value):
    errors = [md(t.get("target")) + "：" + md(t["error"])
              for t in value.get("targets", []) if t.get("error")]
    return "；".join(errors) if errors else "—"


def label(point):
    s = "card " + str(point.get("source", "?"))
    if "destination" in point:
        s += " to " + str(point["destination"])
        s += " / " + point.get("direction", "single-direction")
    if point.get("mode") in ("d2d", "d2d-kernel"):
        s += " / " + point["mode"] + (" / Scopy (2x)" if point["mode"] == "d2d-kernel" else " / XRE memcpy (1x)")
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
    recorded_plan = read(root / "resolved-plan.json", summary.get("resolved_plan", {})) or {}
    recorded_contract = read(root / "toolkit-evidence/toolkit-contract.json", {}) or {}
    excluded = recorded_plan.get("excluded_physical_ids", recorded_contract.get("excluded_physical_ids"))
    excluded_text = json.dumps(excluded) if isinstance(excluded, list) else "未记录"
    assets = []
    (root / "report-assets").mkdir(exist_ok=True)
    def asset(name, svg):
        if svg is None:
            return None
        p = root / "report-assets" / name
        p.write_text(svg, encoding="utf-8")
        assets.append(reference(p, root))
        return p.relative_to(root).as_posix()

    ordered = [c for c in CASES if c in cases] + sorted(c for c in cases if c not in CASES)
    measurements = Counter(str(cases[c].get("measurement_status", "missing")) for c in ordered)
    monitoring = Counter(str(cases[c].get("monitoring_status", "missing")) for c in ordered)
    diagnoses = Counter(str(cases[c].get("diagnosis_status", "missing")) for c in ordered)
    statuses = Counter(str(cases[c].get("status", "missing")) for c in ordered)
    overall = manifest.get("status", summary.get("status", "failed"))

    lines = [
        "# Kunlunxin P800 Toolkit 实验报告",
        "",
        "## 技术摘要",
        "",
        f"- **总体状态：{status_text(overall)}（`{overall}`）。** 本报告只整理已保存证据，不重新判定硬件结果。",
        f"- **测量层：** {len(ordered)} 个 Case 中，通过 {measurements['passed']}、部分完成 {measurements['partial']}、失败 {measurements['failed']}。",
        f"- **作证监控层：** 完整 {monitoring['passed']}、部分完成 {monitoring['partial']}、未执行 {monitoring['not-run']}。",
        f"- **厂商合规诊断层：** 通过 {diagnoses['passed']}、未执行 {diagnoses['not-run']}、失败 {diagnoses['failed']}。"
        "诊断层语义为厂商工具链合规校验（XBLAS/XRE 正确性回读 + 契约覆盖率校验 + xpu-smi 健康观测，2026-10-08 口径，见方法与指标口径）；"
        "本工具链不提供 DMI 式阈值判定工具。",
        f"- **最终 Case 状态：** 通过 {statuses['passed']}、部分完成 {statuses['partial']}、失败 {statuses['failed']}。",
    ]
    scale = manifest.get("settings", {}) or {}
    scale_items = [k for k in ("matrix_size", "payload_bytes", "p2p_payload_bytes", "repeat", "samples") if k in scale]
    if scale_items:
        lines.append("- **本轮规模口径：** " + "，".join(f"{k}={scale[k]}" for k in scale_items)
                     + "（settings 原值；跨 run 比较性能前先对齐口径，不同规模的数值不可直接互比）。")
    if overall == "partial" and measurements["passed"] == len(ordered) and ordered:
        gaps = []
        if monitoring["partial"] or monitoring["failed"]:
            gaps.append("监控证据不完整或采集失败")
        if any(t.get("error") for c in ordered for t in cases[c].get("targets", [])):
            gaps.append("存在带错误记录的执行目标")
        lines.append(f"- **关键解释：** 本轮所有性能测量均形成有效指标；已记录的证据缺口包括：{'、'.join(gaps)}。")
    if summary.get("error"):
        lines += [f"- **执行错误：** {md(summary['error'])}"]
    lines += ["", "## 环境与运行边界", ""]
    lines += table(["项目", "本轮事实"], [
        ["运行 ID", summary.get("run_id", "未记录")],
        ["套件", summary.get("suite", "未记录")],
        ["厂商 / 芯片", f"{summary.get('vendor', 'kunlunxin')} / {summary.get('chip', 'P800')}"],
        ["选中物理卡", json.dumps(manifest.get("selected_physical_device_ids", []))],
        ["排除物理卡（本轮保存的策略）", excluded_text],
        ["上下文 SHA-256", manifest.get("context_sha256", "未记录")],
        ["运行窗口", human_ts(manifest.get("started_at")) + " ～ " + human_ts(manifest.get("finished_at"))],
        ["清理与租约", f"cleanup={summary.get('cleanup_status', 'not-run')}，"
                       f"postflight={summary.get('postflight_status', 'not-run')}，"
                       f"lease_released={summary.get('lease_released')}，"
                       f"qualification={summary.get('qualification_status', 'not-run')}"],
    ])
    lines += ["",
              "运行设置（manifest.settings 原值）：", "",
              "```json",
              json.dumps(manifest.get("settings", summary.get("resolved_plan", {}).get("settings", {})),
                         ensure_ascii=False, sort_keys=True, indent=2),
              "```", "",
              "本报告范围为单机、选中物理卡；不外推到跨机 P2P 或多节点能力，也不替代 Base 双卡/八卡 qualification。"
              "计算类与数据搬运类的原始监控 timeline、逐样本原值和路由覆盖见 [report_monitor.md](report_monitor.md)。", "",
              "## 测量与诊断是两个独立结论", "",
              "`measurement_status` 说明性能命令、指标和设备覆盖；`monitoring_status` 说明同期监控覆盖；"
              "`diagnosis_status` 说明厂商工具链合规校验（XBLAS/XRE 正确性回读 + 契约覆盖率校验 + xpu-smi 健康观测）；"
              "最终状态综合三者，三者均通过才为通过。"
              "P800 工具链不提供 DMI 式阈值判定，诊断层采用合规校验语义（2026-10-08 口径，见方法与指标口径）。", ""]
    lines += table(["Case", "测量", "监控", "诊断", "最终", "指标摘要", "范围", "耗时", "限制/错误"], [
        [case_display(c), status_text(cases[c].get("measurement_status")),
         status_text(cases[c].get("monitoring_status")), status_text(cases[c].get("diagnosis_status")),
         status_text(cases[c].get("status")), metric_summary(cases[c].get("metrics", [])),
         case_scope(c, cases[c]), fmt_duration(cases[c].get("duration_s")), case_limits(cases[c])]
        for c in ordered])
    if "main_memory-bandwidth" in cases:
        lines += ["", D2D_EXPLANATION, ""]
    missing = [c for c in CASES if c not in cases]
    if missing:
        lines += ["", f"本轮未执行（合同定义但未运行）：{'、'.join(case_display(c) for c in missing)}。"]

    def render_case(case, value):
        lines_ = ["", "### " + case_display(case), ""]
        if case == "main_memory-bandwidth":
            lines_ += [D2D_EXPLANATION, ""]
        metrics = value.get("metrics", [])
        rows = [[label(m.get("point", {})), m.get("repetition", "query"), m["field"], f"{m['value']:.8g}", m["unit"],
                 m.get("sample_count", "query"), f"{m.get('sample_cv', 0):.4%}" if m.get("sample_cv") is not None else "N/A",
                 f"[stdout](toolkit-evidence/{m['source']})"] for m in metrics]
        if rows:
            lines_ += table(["设备/模式/载荷", "重复", "字段", "值", "单位", "样本数", "样本 CV", "原始来源"], rows)
            chart_groups = defaultdict(list)
            for metric in metrics:
                # D2D API throughput and kernel read+write need independent axes.
                mode = metric.get("point", {}).get("mode", "unknown") if case == "main_memory-bandwidth" else ""
                chart_groups[(mode, metric["unit"])].append(metric)
            for (mode, unit), chart_metrics in sorted(chart_groups.items()):
                suffix = "-"+mode if mode else ""
                title = D2D_TITLES.get(mode, case_display(case))
                chart = asset(case+suffix+"-"+unit.replace("/", "-")+".svg", bars(title,
                    [(label(m.get("point", {}))+(" / "+m["field"].split("/")[-1] if case == "main_memory-capacity" else "")
                      +" / R"+str(m.get("repetition", "Q")), m["value"]) for m in chart_metrics], unit))
                if chart:
                    lines_ += ["", f"![{title}]({chart})", ""]
        else:
            lines_ += ["没有有效性能指标；不得从退出码、工具存在性或邻近设备结果推断数值。", ""]
        if case == "main_memory-bandwidth":
            for target in value.get("targets", []):
                raw_path = root / "toolkit-evidence/cases" / case / target["target"] / "samples.json"
                raw = read(raw_path, {})
                times = raw.get("samples_ns", [])
                if times:
                    point = target["point"]
                    byte_factor = 2 if point.get("mode") == "d2d-kernel" else 1
                    rows = [(f"sample {i+1}", byte_factor*point["bytes"]/ns) for i, ns in enumerate(times[:25])]
                    name = case+"-"+target["target"].replace("/", "-")+"-samples.svg"
                    chart = asset(name, bars("D2D first 25 measured samples / "+label(point), rows, "GB/s"))
                    lines_ += ["", "前 25 个实测 D2D 样本（仅图表截取，完整样本仍保留；载荷和参考厂商工具的固定工作集不同）：", "",
                               f"![D2D samples]({chart})", "", f"[完整 samples.json]({raw_path.relative_to(root).as_posix()})", ""]
        groups = defaultdict(list)
        for m in metrics:
            groups[(label(m.get("point", {})), m["field"], m["unit"])].append(m["value"])
        if groups:
            lines_ += ["", "重复组（统计各独立进程的结果；CV 使用样本标准差 ddof=1；单次内部样本不代替五次重复）：", ""]
            group_rows = []
            for (name, field, unit), values in sorted(groups.items()):
                avg = statistics.mean(values)
                cv = statistics.stdev(values)/abs(avg) if len(values) > 1 and avg else None
                group_rows.append([name, field, len(values), f"{statistics.median(values):.8g}", unit,
                                   f"{cv:.4%}" if cv is not None else "N/A",
                                   "insufficient-repeats" if len(values) < 5 else "unstable" if cv is not None and cv > .05 else "stable-observed"])
            lines_ += table(["点", "字段", "独立重复", "中位数", "单位", "组 CV", "稳定性"], group_rows)
        for target in value.get("targets", []):
            if target.get("error"):
                lines_ += ["", "- " + md(target["target"]) + "：" + md(target["error"])]
        return lines_

    for key, title, blurb in FAMILIES:
        members = [c for c in ordered if family_of(c) == key]
        if not members:
            continue
        lines += ["", "## " + title, "", blurb, ""]
        for case in members:
            lines += render_case(case, cases[case])
    others = [c for c in ordered if family_of(c) == "other"]
    if others:
        lines += ["", "## 其他测量", ""]
        for case in others:
            lines += render_case(case, cases[case])

    lines += ["", "## 健康、拓扑与证据完整性", "",
              f"- 选中物理卡 {json.dumps(manifest.get('selected_physical_device_ids', []))}；"
              f"本轮保存的排除物理卡策略：{excluded_text}。",
              "- 每个执行目标保留命令、stdout/stderr 与 SHA-256；指标与原始输出均可追溯到 toolkit-evidence/cases/。",
              "- 同期监控逐卡 timeline 与逐样本原值见 [report_monitor.md](report_monitor.md)；"
              "缺失样本记 N/A，不补零；曲线只连相邻有效样本。",
              "- [Manifest](toolkit-evidence/manifest.json) · [Native tool provenance](toolkit-evidence/provenance.json)"
              " · [诊断覆盖](toolkit-evidence/diagnostics/coverage.json) · [SHA256 索引](sha256-index.json)", "",
              "## 方法与指标口径", "",
              "- 计算使用独立 Toolkit C++ 程序调用 XBLAS；传输调用 XRE。没有导入 PyTorch/Base workload，"
              "也没有将官方 fc_effciency 的存在或 help 输出当作测量通过。",
              "- 计时为 CLOCK_MONOTONIC 的原生 API 调用到同步完成，不包含进程启动、分配、预热及正确性回读；"
              "它不是纯 kernel 时间。容量为每卡 HBM 查询值。runtime memcpy D2D/单向 P2P 使用一份 payload；"
              "D2D kernel 使用读+写两份 payload；双向 P2P 单独运行两个并发方向，"
              "以实际两份 payload/两方向完成时间计算，含线程协调开销。",
              "- INT8 使用 XBLAS fc_fusion（INT8 输入和 TGEMM、FP32 输出、maxima=127、alpha=1、beta=0、无 bias、"
              "LINEAR 激活）。它不是 INT8→INT32 GemmEx；后者在当前镜像的实测返回参数错误。",
              "- Pinned 异步模式可能按配置将逻辑 payload 分为多次原生提交，最后统一等待。"
              "实际分块大小和 API 次数记录在 metrics 的 async_chunk_bytes / api_calls_per_sample；"
              "这是分块端到端带宽，不能与单次整块 API 调用混称。",
              "- 重复组统计各独立进程的结果；CV 使用样本标准差 ddof=1；单次内部样本不代替五次重复。",
              "- 厂商诊断层口径（2026-10-08）：P800 的诊断层定义为厂商工具链合规校验——每个测量点的原生命令都经 "
              "XBLAS/XRE 正确性回读（checked_values 逐点核对）与契约覆盖率校验，配合 xpu-smi pre/post 健康观测；"
              "三者齐全且测量通过即诊断层通过。当前镜像不提供 ascend-dmi 等价的算力/带宽/信号质量阈值判定工具；"
              "与 Ascend 报告并排对比时，诊断列语义为“合规校验”而非“阈值判定”，属厂商能力差异的如实呈现。", "",
              "## 限制与后续动作", "",
              "- 占用卡数据属于 exploratory；本报告不晋级 candidate，也不替代 Base 双卡/八卡 qualification。",
              "- 当前镜像不提供 Ascend DMI 等价的算力/带宽/信号质量阈值判定工具；诊断层为合规校验语义（见方法与指标口径）。"
              "xprofiler 的文件及 help 留档不代表采集了 kernel trace。",
              "- INT8→INT32 GemmEx 在当前镜像实测返回参数错误，INT8 走 XBLAS fc_fusion（见方法与指标口径）。"]
    if summary.get("error"):
        lines += [f"- 执行错误：{md(summary['error'])}"]
    for case in ordered:
        for target in cases[case].get("targets", []):
            if target.get("error"):
                lines += [f"- {case_display(case)} / {md(target['target'])}：{md(target['error'])}"]
    lines += ["", "## 原始证据索引", ""]
    for case in ordered:
        lines.append(f"- {case_display(case)}：[metrics](toolkit-evidence/cases/{case}/metrics.json)"
                     f" · [目标目录](toolkit-evidence/cases/{case}/)")
    lines += ["- [Manifest](toolkit-evidence/manifest.json)",
              "- [Native tool provenance](toolkit-evidence/provenance.json)",
              "- [Code identity](code-identity.json)",
              "- [Image identity](image-identity.json)",
              "- [Monitor report](report_monitor.md)",
              "- [Diagnosis coverage](toolkit-evidence/diagnostics/coverage.json)",
              "- [SHA256 index](sha256-index.json)",
              "", "---", "",
              f"报告生成器 schema：`{REPORT_SCHEMA_VERSION}`。报告是 JSON 证据的确定性视图；"
              "发生冲突时以原始 manifest、command、stdout、stderr 和 SHA-256 为准。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    raw_samples = root / "toolkit-evidence/monitor/samples.jsonl"
    samples = [json.loads(x) for x in raw_samples.read_text().splitlines()] if raw_samples.is_file() else []
    monitor_lines = ["# P800 Toolkit 同期监控报告", "", "每个样本保留物理卡、有效性、时间与原值；缺失为 N/A，不补零。曲线只连相邻有效样本。", "",
                     "- [原始命令 JSONL](toolkit-evidence/monitor/samples.raw.jsonl)", "- [结构化 JSONL](toolkit-evidence/monitor/samples.jsonl)", "",
                     "## 逐测试点监控覆盖", ""]
    monitor_lines += table(["Case/target", "状态", "窗口内完整样本", "说明"], [
        [case_display(c)+"/"+t["target"], t.get("monitoring_status"), json.dumps(t.get("monitor_sample_counts", {}), sort_keys=True),
         t.get("monitor_scope", "每目标至少 10 个完整落入真实 measurement 窗口的有效样本")]
        for c in ordered if c in cases for t in cases[c].get("targets", [])])
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
    monitor_lines += ["---", "",
                      f"监控报告生成器 schema：`{REPORT_SCHEMA_VERSION}`。发生冲突时以 xpu-smi 原始输出、manifest 和 SHA-256 为准。", ""]
    (root / "report_monitor.md").write_text("\n".join(monitor_lines), encoding="utf-8")
    return {"schema_version": REPORT_SCHEMA_VERSION, "status": "passed", "generator_sha256": sha256(Path(__file__)), "outputs": {"main": reference(root / "report.md", root),
            "monitor": reference(root / "report_monitor.md", root)}, "assets": assets}


def generate_and_record(root):
    root = Path(root)
    metadata = generate(root)
    summary = read(root / "summary.json", {})
    summary["report_generation"] = metadata
    write_json(root / "summary.json", summary)
    index(root)
    return metadata
