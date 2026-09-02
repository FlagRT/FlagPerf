<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagPerf Base 使用指南

本目录提供 FlagPerf 的基础规格评测能力。当前推荐入口是统一宿主命令
`python3 base/run.py`，但 Benchmark 与 Toolkit 仍使用独立执行器、权限和结果语义：

- **Benchmark**：通过 PyTorch 2.10、Torch-FL 和原始 Base Case 测量 workload；
- **Toolkit**：通过 MindCluster ToolBox、`ascend-dmi`、`npu-smi` 或 HCCL Test
  执行厂商测量和诊断；
- **report**：从已有证据确定性重建 Markdown/SVG 报告，不重跑硬件。

当前统一入口只承诺 **Ascend 单宿主**。原多机入口保存在
[`legacy/cluster_run.py`](legacy/cluster_run.py)，仅用于迁移兼容。

## 1. 环境基线

标准 operator runtime 锁定在：

- CANN 9.0.0；
- MindCluster ToolBox 26.1.0（从宿主挂载）；
- Python 3.11.15；
- PyTorch 2.10、Torch-FL、Triton Ascend 和 FlagGems；
- 镜像 `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`。

Torch-FL 拥有 PyTorch PrivateUse1/`torch.flagos`，不要在该镜像中安装或混用
`torch_npu`。锁文件、镜像身份和验证边界位于
[`vendors/ascend/torch_fl_2.10/`](vendors/ascend/torch_fl_2.10/)。

运行前确认：

1. Docker daemon 可用；
2. `npu-smi info` 能看到目标 NPU，且目标 Device 空闲；
3. `configs/ascend910_cann9_local.yaml` 中的 ToolBox、驱动和设备路径与宿主一致；
4. 使用 `--npu-ids` 或 `--device-ids` 显式选择本轮资源；
5. 仅在确认高权限容器可接受后传入 `--allow-privileged-root`。

所有命令均建议从仓库根目录执行：

```bash
cd FlagPerf
python3 base/run.py --help
```

## 2. 先生成静态计划

`--dry-run` 只校验仓库配置并打印静态计划，不检查 Docker、NPU 占用或物理到逻辑
Device 映射，也不会生成虚假的 image ID：

```bash
python3 base/run.py benchmark run \
  --case computation-FP16 \
  --npu-ids 7 \
  --monitor on \
  --dry-run

python3 base/run.py toolkit run \
  --case computation-FP16 \
  --npu-ids 7 \
  --dry-run
```

示例中的 NPU 7 不是固定要求。正式运行前必须按当前宿主拓扑替换为已授权的空闲物理
NPU ID；也可改用 `--device-ids 14,15` 一类逻辑 Device 集合。两种选择器互斥。

## 3. 运行 Benchmark

下面的命令使用 NPU 7 映射出的逻辑 Device 启动原始 FP16 Case，并对实际选中设备采集
同窗 `npu-smi info -t usages` 证据：

```bash
python3 base/run.py benchmark run \
  --case computation-FP16 \
  --npu-ids 7 \
  --nproc-per-node 2 \
  --monitor on \
  --allow-privileged-root
```

Benchmark 保留原 Case 的 YAML 合并、warmup、计时、公式和 `[FlagPerf Result]` 输出语义。
Ascend 适配器只负责 Torch-FL 初始化、`flagos:<local_rank>` 设备选择和设备同步；Gloo
只用于 GEMM 等非通信 workload 的控制 barrier。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--case` | 必填的 Base Benchmark Case |
| `--case-config` | 只读挂载的 Ascend Case YAML 覆盖文件 |
| `--nproc-per-node` | 本机 torchrun rank 数；默认等于解析出的逻辑 Device 数 |
| `--monitor on\|off` | 是否采集独立状态的同窗监控，默认 `on` |
| `--timeout` | 容器硬超时，默认 3600 秒 |
| `--allow-high-risk-case` | 显式允许容量/OOM/长时间 Case |

`main_memory-capacity` 必须额外传入 `--allow-high-risk-case`。不要在共享机器上用完整参数
直接试跑该 Case。

## 4. 运行 Ascend Toolkit

Toolkit 会主动执行 DMI/HCCL 等性能命令，因此需要独立授权：

```bash
python3 base/run.py toolkit run \
  --case computation-FP16 \
  --npu-ids 7 \
  --allow-privileged-root \
  --allow-disruptive-dmi
```

`--case` 可重复使用；不指定时执行默认安全集合，但统一入口仍要求显式选择设备。
`--compute-monitor` 和 `--data-movement-monitor` 分别控制计算与数据搬运作证，默认均为
`on`。Toolkit 数值来自厂商 microbenchmark，不应在 workload、设备范围、计时边界和公式
未对齐时直接与 Benchmark 相除或归因。

完整 Case、参数、证据和限制见
[`toolkits/ASCEND_A3_910C_TEST_MECHANISM.md`](toolkits/ASCEND_A3_910C_TEST_MECHANISM.md)。

## 5. FlagCX P2P candidate

Ascend 单机 P2P Benchmark 不能使用标准 operator runtime。它要求独立 communication
candidate、单节点、恰好两个 rank，以及仓库 allowlist 中的有界 Case 配置：

```bash
python3 base/run.py benchmark run \
  --config base/configs/ascend910_cann9_p2p_candidate.yaml \
  --case interconnect-P2P_intraserver \
  --device-ids 14,15 \
  --nproc-per-node 2 \
  --case-config base/benchmarks/interconnect-P2P_intraserver/ascend/case_config.smoke.yaml \
  --monitor off \
  --allow-privileged-root \
  --allow-candidate-runtime \
  --dry-run
```

确认静态计划、设备空闲和授权后才可移除 `--dry-run`。公共 distributed backend 仍为
Torch-FL `flagos`，FlagCX 是其内部通信数据面。`--allow-candidate-runtime` 只允许执行候选
镜像，不会将 manifest 晋级为 passed。

当前 candidate 已完成 C2 双 rank sentinel 和 C4 有界 Base smoke，但 peer failure、hung
collective、正式性能、跨机和长稳尚未验收。因此
[`vendors/ascend/torch_fl_2.10_flagcx/validation-summary.json`](vendors/ascend/torch_fl_2.10_flagcx/validation-summary.json)
仍记录 `overall_status=partial`、`production_eligible=false`。

## 6. 结果、报告和退出码

默认结果位于 `base/result/<RUN_ID>/`，核心证据包括：

- `summary.json`：外层状态、运行时、设备、权限和生命周期；
- `resolved-plan.json`：正式 preflight 后解析出的完整执行计划；
- `benchmark-result.json` 或 `toolkit-evidence/manifest.json`：领域测量事实；
- `report.md`、`report_monitor.md` 和 `report-assets/`：确定性阅读视图；
- Case 配置快照、原始日志、pre/postflight 和 SHA-256 索引。

离线重建报告：

```bash
python3 base/run.py report --run-id benchmark-YYYYMMDDTHHMMSSZ
```

统一状态边界：

- `0`：`passed`；
- `1`：执行或测量失败，或报告重建失败；
- `2`：配置/授权错误，或执行测量通过但所请求证据不完整而得到 `partial`。

Benchmark 的 `execution_status`、`measurement_status`、`monitoring_status`、
`postflight_status` 和 `report_status` 相互独立。监控不完整不能伪装成测量通过或失败；
`--monitor off` 记为 `not-run`，不使成功测量失败。

## 7. 入口与职责边界

```text
Host
  base/run.py
    ├─ BenchmarkExecutor -> Docker -> benchmark_worker.py -> torchrun -> 原 Case
    ├─ ToolkitExecutor   -> Docker -> evidence_runner.py -> DMI/npu-smi/HCCL
    └─ report            -> 已保存证据 -> Markdown/SVG
```

两个 Executor 只共享运行时锁、设备选择、preflight、lease 和外层证据约定，不共享 Case
执行、权限、计时公式、诊断或结果 schema。`run_toolkit.py`、`run_local.py` 和
`container_main.py` 是迁移兼容入口，不是新的并列主入口。

架构、适配规则和旧集群流程见
[`../docs/base/base-case-doc.md`](../docs/base/base-case-doc.md)。
