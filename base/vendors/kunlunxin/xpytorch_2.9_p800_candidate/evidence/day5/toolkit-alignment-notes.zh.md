# P800 对齐 Toolkit 参考包：现状与待办清单

参考基准：`参考资料/20260914T133423Z`（**Toolkit 域**结果包，Ascend 910C，`kind: toolkit`，`suite: ascend-toolkit`，由 `base/run.py toolkit run` 产出）。

状态：**2026-09-22 核对**。本文逐项列出参考包的真实参数、P800 现状与缺口。P800 运行主仓仍是 `klx:/home/kzhang519/Zhiyu/runtime-team/FlagPerf`，物理卡 5。

## 一、参考包是什么

| 项目 | 事实 |
|---|---|
| 域 | `base/toolkits/**`（Toolkit 套件），与我们前五天做的 `base/benchmarks/**`（Base 套件）**并列且独立** |
| 厂商工具 | 全部测量由 `ascend-dmi`（MindCluster ToolBox 26.1.0）产出 |
| 规模 | 12 个 case、70 条命令、456 个文件 |
| 产物 | `summary.json`（三层状态）+ `report.md` + `report_monitor.md` + `report-assets/*.svg` + `toolkit-evidence/{manifest,environment,cases,health,topology,diagnostics}` + `host-preflight/<host>/` |
| 三层状态 | `measurement_status`（测量）/`monitoring_status`（同期监控）/`diagnosis_status`（厂商阈值诊断）→ 最终 `status` |
| 证据契约 | `toolkit-evidence/cases/<case>/<target>/{microbenchmark.stdout,microbenchmark.stderr}` + `metrics.json{commands[],metrics[],status,measurement_status,diagnosis_status,sweep_scope,duration_s}` |

每个 case 的 `metrics[]` 条目都带 `field`（厂商输出中的字段路径）、`source`、`unit`，可回溯到原始 stdout —— 这一点与本项目"证据优先"的要求一致。

## 二、已对齐项

| 项目 | 参考参数 | 本轮处理 |
|---|---|---|
| **H2D/D2H 带宽载荷** | `-s 536870912 --et 50` = **512 MiB × 50** | 八种请求组合统一 512 MiB（迭代数按本项目 15 秒采样下限上调至 400–1000，逐变体记录）。**已重跑并合格** |

附带结论修正：pinned + non_blocking 在 512 MiB 下正常，此前记录的"2 GiB 上限"实为 4 GiB 载荷特有的厂商限制。

## 三、待对齐清单

### 3.1 测量类

| # | case | 参考命令（实测） | 参考指标 | P800 现状 | 缺口与建议 |
|---|---|---|---|---|---|
| 1 | computation-{BF16,FP16,FP32,INT8} | `ascend-dmi -f -t <dtype> -d ID --et 80 -q --fmt json` | TFLOPS / TOPS，逐设备 | Base 域已有**已验证**的 8192³ 测量（含 INT8 设备内核修正、BF16 定位） | 规模**无法逐字对齐**（工具不暴露 shape）；建议复用现有测量，在 toolkit 层按 `metrics[]` 契约重新落盘，并把"实现方式不同（自研 vs 厂商工具）"写入 scope |
| 2 | **interconnect-h2d-latency** | `ascend-dmi -l -t h2d -s {512,4096,65536,1048576} -d ID` | ns，**16 点/case** | **无** | 需新建：本机无厂商工具，需自研时延测量（小消息、多次重复取中位数），并按 16 点扫点落盘。**全仓库仅 Ascend 有此 case，无其他厂商先例** |
| 3 | **interconnect-d2h-latency** | 同上，`-t d2h` | ns，16 点 | **无** | 同上 |
| 4 | main_memory-bandwidth（D2D） | `--bw -t d2d -d ID -q --fmt json`（工具固定 size/次数） | GB/s，**每设备 25 样本序列** | **无**（Base 域 Day 6 计划内） | 需新建；P800 对应实现＝设备内 `clone()` 类搬运，注意参考包口径"标准结果逐逻辑 Device 原样保留，不再执行 ×2" |
| 5 | main_memory-capacity | `npu-smi info -t memory -i CARD -c CHIP`（并读 ecc） | **逐 chip** `HBM Capacity(MB)`，不乘二、不聚合 | **无** | 需新建；P800 对应＝`xpu-smi` 的内存容量字段（本机每卡 98304 MiB）。注意这是**容量查询口径**，不是"实际可分配量"测试——与 Base 域 `main_memory-capacity`（持有分配）语义不同，两者都要写明 |
| 6 | interconnect-P2P_intraserver | `--bw -t p2p --ds A --dd B`，6 个无序对 | GB/s，单向+双向各 6 值 | **无** | 需两卡 + XCCL（Base 域 Day 6 计划内）。参考包口径：每个无序对只按升序执行一次 A→B，反向不推断 |
| 7 | interconnect-P2P_intraserver-latency | `-l -t p2p -s 65536 --ds A --dd B`，6 对 | ns，单一 64 KiB 点 | **无** | 同上；依赖 P2P 通道 |

### 3.2 基础设施类

| # | 项目 | 参考包做法 | P800 现状 | 缺口 |
|---|---|---|---|---|
| 8 | Toolkit 执行器 | 由 `base/executors/toolkit.py` 驱动 | **硬编码 Ascend**（`base.vendors.ascend.provider` 导入、privileged 容器、挂载 `_common/ascend/A3/`） | 需泛化为 provider 模式，新增 `kunlunxin/P800` provider（host preflight、容器 spec、lease、监控、证据根） |
| 9 | 证据运行器 | `base/toolkits/_common/ascend/A3/evidence_runner.py`（驱动厂商工具、捕获原始输出、解析 metrics、写三层状态） | 无 Kunlunxin 版 | 需新增 `_common/kunlunxin/P800/evidence_runner.py`，实现同一 metrics/证据契约 |
| 10 | case 入口 | `base/toolkits/<case>/ascend/A3/main.sh` → 调 evidence_runner | Kunlunxin 仅有 R300p/R310p 的 **legacy 风格**（编译运行 `/opt/util/examples/...`、依赖 `/opt/xre`、`/opt/xhpc`——本机 M1 镜像中不存在） | 需为每个 case 新增 `kunlunxin/P800/main.sh`（现代风格） |
| 11 | 报告生成器 | `base/generate_toolkit_report.py`（中文报告、SVG 图表、三层状态表、逐设备表） | 生成器存在但为 **Ascend 专用**（`CASE_ORDER`、DMI 解析、诊断层） | 需泛化：case 顺序/名称、诊断层来源、图表数据源 |
| 12 | 诊断层 | `ascend-dmi --diagnosis {aiflops,bandwidth,hbm,signalQuality}` 阈值诊断 | 无对应厂商工具 | 需定策略：用 `xpu-smi` 可得项（ECC/温度/功耗/健康/拓扑）做等价诊断并**显式标注覆盖范围**，或按 case 记 `not-supported` |

## 四、依赖与建议顺序

```text
① 时延扫点（h2d/d2h）        ← 唯一完全缺失、无任何厂商先例的测量；不依赖两卡
② HBM 容量                   ← 语义最简单（xpu-smi 查询），可直接对接
③ D2D 带宽                   ← 单卡可做
④ 执行器泛化 + P800 evidence_runner + 报告泛化   ← 上面三项的承载层，可与①~③并行
⑤ P2P 带宽/时延              ← 依赖两卡与 XCCL（Base 域 Day 6 计划内）
⑥ 诊断层策略落地             ← 依赖①②可得的 xpu-smi 证据
```

工作量粗估：①+②+③ 约 1 天（含实机与证据）；④ 约 1 天；⑤ 依赖两卡窗口；⑥ 约半天。

## 五、需要确认的两点

1. **测量实现方式**：参考包全部由厂商工具产出，本机没有昆仑芯的对应工具。P800 侧按"自研测量实现同一 case 集与产物格式"推进，还是你指定某个昆仑芯厂商工具（有名称/路径我就按它的真实输出接）？
2. **诊断层策略**：用 xpu-smi 可得项做等价诊断并标注覆盖范围，还是统一记 `not-supported`？

未确认前，我按建议顺序先做 ①（时延扫点）与 ②（容量）——这两项不依赖两卡，且能立刻缩小与参考包的差距。
