> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书链接转入 [历史引用索引](history/history-references.md)，其出处保留。

# 09 CLI 与批量运行

> **当前实现更新（2026-10-26，D33）**：已落地动词为 record/inspect/import/page/seal/analyse。record 默认 final，窗口模式须显式 --capture-mode windowed；B run 的 inspect 自动有界。report/campaign/watch 仍为后续目标。生产目录与命令见[默认 B 使用说明](history/history-references.md#ref-e854eaa4d4fc)，整合证据见[交付记录](history/history-references.md#ref-dbcab1a14471)。下文 2026-10-18 状态保留历史语境。

> 上游：[02 架构](02-architecture.md) §3/§4、[05 预设](05-probes-and-vendors.md) §3、[07 分析器](07-analyzers-and-reports.md)

## 1. 设计原则：按动词扩展，不按分析种类扩展

> **当前实现（T9，2026-10-18）**：CLI 目前只有两个动词——**`record`** 与 **`inspect`**（`python3 -m flagprof --help`）。下文的 `report`、`campaign`、`watch` 均为**设计目标（未实现）**：`campaign` 属 P5，`watch` 属 P3，`report` 随分析器落地（阶段见 [10 路线图](10-roadmap.md)）。目前的"campaign"只是 `tools/acceptance/` 下的验收脚本（如 `t93_campaign.py`），不是产品命令，也不读写下文的 `_index/`。

v1 的 10 个顶层命令是按"分析种类"长出来的（accuracy / performance / preview / oracle / performance-baseline / repack-reference…），能力增加则命令增加（[01 §2.4](history/01-diagnosis.md)）。

v2 固定为 **5 个动词**，新增能力只增加 `report` 的分析器名或 `record` 的预设，不增加顶层命令。

| 动词 | 作用 |
|---|---|
| `record` | 采集。三种接入方式都走它 |
| `report` | 从一个或多个 run 产出分析与报告 |
| `campaign` | 批量：矩阵展开、调度、断点续跑、汇总 |
| `watch` | 在线尾读 + 规则引擎 + Responder |
| `inspect` | 查看 run/store/registry 的元信息（原 `list`/`export` 的合并） |

## 2. record 的三种形态

> **当前实现（T9，2026-10-18）**：
>
> - **record**：`flagprof record --store <dir> --vendor <id> [--preset {perf,modules}] [--active N] [--windows N] [--invalid-policy {stop,continue}] [--plugins <descriptors.json>] -- <python 目标>`。`--store`、`--vendor` 必传；`--preset` 默认 `perf`；`--active`（默认 5）、`--windows`（默认 2）是每窗步数与窗口数。只注入 wrapper 直接启动的 Python 目标，后代与 worker 须显式 attach（[D32](DECISIONS.md#d32--record-自动注入范围只注入-wrapper-直接启动的-python-目标)，见 [02 §3](02-architecture.md#3-目标接入attach--wrap--launch)）；被测代码当前须显式调用 `step()` 推进窗口（该方式按 [D33](DECISIONS.md#d33--恢复attach-不改被测代码)（用户 2026-10-20）降为**可选项**：默认模式恢复"attach 不改被测代码"、由工具侧自动推进窗口，机制待设计，见 [02 §3](02-architecture.md#3-目标接入attach--wrap--launch)）。目标退出后另起独立进程导入，再 finalize。
> - **inspect**：`flagprof inspect <run> [--verify] [--json]`，只接**单个 run 路径**，读已存的身份与证据状态。
> - **未实现**（设计目标，下文保留）：`--probes`、`--seal`、`--inputs-from`、`--experiment`、`--layers`、`--blob`、`--out`、`--target`、`--task`、`--profile`、`--component`；`perf-lite`、`accuracy`、`coverage`、`monitor` 预设（见 [05 §3](05-probes-and-vendors.md#3-预设preset)）；`--vendor` 自动探测（当前必须显式传）；`inspect runs --model --where ...` 的 run 查询（§5.1，P5）。
> - **已知缺口（T9.4 待修）**：`inspect` 不显示 run 级 `truncated`/`lost`（被 SIGKILL 的 run 在 inspect 里看不出不完整）；`inspect` 对不存在的路径输出为空且退出 0；`record` 的窗口日志来源尚未接线，生产 record 下"日志报错判 invalid"不会触发，也不留"日志未检查"的痕迹（均见 [05 §4.3](05-probes-and-vendors.md#43-生命周期义务当前实现t92026-10-18)）。

```bash
# attach：用户自己改两行代码，这里不需要 CLI
python -c "import flagprof; flagprof.attach(preset='perf'); ..."

# wrap：注入已有脚本或服务
flagprof record --preset perf-lite --out runs/ -- python -m vllm.entrypoints.api_server --model ...

# launch：内置运行器（campaign 的主力）
flagprof record --preset perf --target hf:$MODELS/Qwen3-Embedding-0.6B --task embedding --out runs/
flagprof record --profile profiles/qwen3-embedding-0.6b.yaml --preset accuracy --seal --out runs/
```

> 上面三种形态的 CLI 示例均为**设计目标（未实现）**；当前可用的 wrap 形态见本节开头的"当前实现"。第二条 launch 示例中的 `--seal` 另见下表注。

常用参数：

| 参数 | 说明 |
|---|---|
| `--preset` | 见 [05 §3](05-probes-and-vendors.md) |
| `--probes +tensor,-compile` | 在预设基础上增删单个探针 |
| `--seal` / `--inputs-from <run>` | seal 模式；复用另一个 run 的输入（**设计目标（未实现，阶段 P4）**，见下表后的注） |
| `--experiment k=v` | 声明本 run 的实验变量（供 identity diff 校验） |
| `--layers all\|auto\|depth=N` | 自动选层策略覆盖 |
| `--blob <boundary>` | 对指定边界落张量 blob |
| `--vendor ascend` | 显式指定厂商适配（默认自动探测；当前必须显式传，无自动探测） |

> **`--seal` 注（避免同名误读）**：此处与 [02 §4](02-architecture.md#4-两种运行模式observe-与-seal) 的 seal 是 **P4 的实验 seal**——封存输入文件 + 哈希，并校验与对照 run 的环境身份。当前 SDK 层的 **sealed-window 快照**（官方 profiler 窗口产物的逐文件 SHA 快照，见 [02 §5](02-architecture.md#5-一次-run-的数据流)）是另一回事：只封存采集产物，不封存输入、不校验身份。**不能因为 profiler 窗口已封存，就声称 `record --seal` 已完成。**

## 3. 组件开关与对照矩阵

继承 v1 的 off/on/both 语义（[00 §4](00-requirements.md)）：

```bash
flagprof record --component flaggems=both --component flagtree=on --component flagcx=off ...
```

| 取值 | 含义 |
|---|---|
| `off` / `on` | 该组件在本 run 中关闭/启用 |
| `both` | 展开为**两个 run**（off 与 on 各一），自动构成对照；两个 run 共享输入（seal），`experiment` 字段自动填好 |

**一次只允许一个组件取 `both`**（v1 规则，保留）。原因：两个组件同时 `both` 会产生 4 个 run 的交叉，归因变得不唯一 —— 要做多因素对照，用 `campaign` 显式写矩阵，让用户意识到自己在做什么。

`both` 展开的两个 run 自动记录为一个 **pair**，`report` 可直接消费：

```bash
flagprof report --pair <pair_id> route.impact oracle.layer
```

## 4. report

```bash
flagprof report <run>                                   # 跑全部可用分析器，出 HTML + result.json
flagprof report <run> perf.op perf.layer                # 只跑指定分析器
flagprof report --ref <run_A> --cmp <run_B> oracle.layer oracle.first_divergence
flagprof report <run> --export missing_ops,op_perf      # 只要 §4 的机器可读导出
```

规则：

- 分析器不可用时 panel 说明原因（[07 §6](07-analyzers-and-reports.md)），命令本身**不失败**；
- `--ref/--cmp` 先做 identity diff，未被 `experiment` 解释的差异导致对比类分析器拒绝出结论。

## 5. campaign：500 × 5 的组织方式

```yaml
# campaigns/top500-batch1.yaml
name: top500-batch1-embedding
matrix:
  model:    { from: profiles/batch1/*.yaml }        # 展开为 N 个模型
  vendor:   [ascend, nvidia]
  component.flaggems: [off, on]
preset: perf
seal: true
inputs_shared_by: [model]        # 同一模型在不同 vendor/开关下复用同一份输入
retry: { max: 2, on: [device.error, comm.timeout] }
isolate: { on: [device.error], scope: device }
report: { per_run: true, summary: true }
```

| 能力 | 说明 |
|---|---|
| 矩阵展开 | `model × vendor × component` 的笛卡尔积，每个格子是一个 run |
| 输入共享 | `inputs_shared_by` 保证跨厂商对比用同一份输入（Oracle 差分的前提） |
| 断点续跑 | 每个格子的状态记在 campaign 索引中；重跑只执行 `pending`/`failed` 的格子（N5） |
| 重试与隔离 | P5：依据 run 的退出状态与超时重试、跳过；P3 之后：由 `campaign` Responder 依据 Finding 隔离设备/节点（[08 §4](08-online-and-fault.md)、[10](10-roadmap.md)） |
| 汇总 | 聚合各 run 的 `result.json`，产出批量报告与跨厂商矩阵（R89、R86/R88） |

### 5.1 campaign 索引

```text
<store>/_index/
  runs.jsonl          每个 run 一行：run_id、model、vendor、preset、experiment、status、关键指标摘要
  campaigns/<name>.json   矩阵定义 + 每格状态 + run_id 映射
```

索引由 manifest 与 `result.json` 生成，可随时重建（store 是唯一真相来源）。查询：

```bash
flagprof inspect runs --model 'qwen3*' --vendor ascend --status failed
flagprof inspect runs --where 'route.fallback_ratio > 0.2' --limit 20
```

> **run_id 与 manifest 从 P0 就按可索引设计**（[02 §9](02-architecture.md)），即使 campaign 功能本身在 P5 才做。

## 6. 配置分层

优先级从低到高：

```text
内置默认 → registry 中的预设 → 用户全局配置(~/.flagprof/config.yaml)
  → campaign 配置 → profile YAML → CLI 参数 → 环境变量(FLAGPROF_*)
```

最终生效的完整配置写入 `manifest.probes` 与 `manifest.identity`，保证 run 可复现（N4）。环境变量优先级最高是为了 wrap 模式：注入的 `sitecustomize` 只能通过环境变量接收配置。
