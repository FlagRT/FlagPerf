# Operation：算子精度、性能与诊断

Operation 选择一个算子及输入规模，在指定设备上检查结果是否正确、实际走了什么实现，并测量耗时和显存。
当前提供 52 个 Case（算子案例）；可以运行原生 PyTorch 路径、FlagGems 路径，或在同一次运行中比较两条路径。
两条路径分别对 CPU 参考检查正确性，不把其中一条直接当作另一条的真值。

| 目的 | 命令 |
|---|---|
| 找到算子与可用输入类型 | `list` |
| 查看执行计划或运行算子 | `run` |
| 从已有记录重建/汇总报告 | `report` |
| 检查失败证据，必要时显式复放 | `diagnose` |

## 1. 准备环境

需要 Linux、Python 3.10+、Docker 与设备访问权限。`list` 和 `run --dry-run` 只需宿主 Python 标准库，不启动设备。
真实执行还需要包含对应 PyTorch、厂商设备库及 FlagGems 的兼容镜像。

Ascend 默认使用 CANN 9 / PyTorch 2.10 / Torch-FL 的锁定组合，按 [Ascend 环境说明](../docs/ascend/README.md) 准备。
该镜像不会由程序自动下载。NVIDIA、寒武纪、昆仑芯、天数智芯和沐曦需要显式提供匹配厂商的 `--image`；
注册入口可用不代表每个厂商和 dtype 都已经实机通过。

所有命令从 **FlagPerf 仓库根目录** 执行。先选当前空闲设备，下面的 `0` 是物理编号示例。
Ascend 镜像涉及的特权权限由 `--allow-privileged-root` 显式开启；只在获准使用的主机上运行。

## 2. 跑通一个小案例

```bash
# 查看算子及输入类型，不需要模型或设备环境。
python3 operation/run.py list --names-only
python3 operation/run.py list --case abs

# 先看计划，不启动 Docker。
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run

# 确认环境与空闲设备后执行原生路径。
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --allow-privileged-root
```

终端最后打印 `Report:`、`Summary:` 和 `Results:`。打开 `Report:` 指向的 `report.md`，先看执行和正确性状态，
再看路由与性能。输出默认保存在 `operation/result/operation-<id>/`，每个 Case/dtype/路径/设备有自己的子目录。
`smoke` 是小规模执行；默认 `daily` 有更多重复轮次。它们的输入规模与采样次数不同，不能直接当作同一工作量比较。

## 3. 比较原生与 FlagGems

FlagGems 是算子实现库；`--oplib both` 在相同输入和规模下分别运行原生与 FlagGems 路径：

```bash
python3 operation/run.py run --vendor ascend --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke --allow-privileged-root
```

每一侧都独立检查 CPU 参考和路由。结果不足以比较时保留原因，不产生看似有效的性能比值。
只想运行一侧，可选择 `--oplib nativetorch`（默认）或 `--oplib flaggems`。

## 常用变化

```bash
# 指定两个算子，或为同一算子指定多个规模。
python3 operation/run.py run --vendor ascend --device-ids 0 --case abs --case mm \
  --dtype FP16 --workload smoke --allow-privileged-root
python3 operation/run.py run --vendor ascend --device-ids 0 --case mm \
  --size M=128,N=256,K=128 --size M=512,N=512,K=512 --allow-privileged-root

# 正式计时之外，额外采集设备时间线；会增加执行成本。
python3 operation/run.py run --vendor ascend --device-ids 0 --case mm \
  --dtype FP16 --workload smoke --profiling timeline --allow-privileged-root

# 已有报告可离线重建。
python3 operation/run.py report --run-dir operation/result/operation-YOUR_RUN_ID
```

`--warmup`、`--iters`、`--rounds` 调整采样；`--profiling off|timeline|full` 控制独立采集，默认 off。
旧参数 `--profile` 保留为 `--workload` 的兼容别名，两者不能同时指定。
没有指定 `--case` 时会规划全部案例；第一次使用建议明确选择一个小案例。
`--spectflops` 是所选设备范围、dtype 对应的峰值分母；不填写时利用率为 N/A，不猜测硬件峰值。

## 失败后从哪里开始

先使用不占用设备的诊断，`--source-task` 指向结果中的具体 Case 子目录，而不是整个汇总目录：

```bash
python3 operation/run.py diagnose --source-task /absolute/path/to/case-result --dry-run
python3 operation/run.py diagnose --source-task /absolute/path/to/case-result
```

默认只读取已封存证据。显式 `--replay` 才会使用保存的输入和原镜像，在所选设备上重新执行一次 probe。
它不是自动修复，也不重复原性能计时循环。具体流程见 [诊断与复放](../docs/operations/diagnosis.md)。

- [如何读指标与报告](../docs/operations/results.md)
- [Ascend 适配边界](vendors/ascend/README.md)
- [厂商适配与旧集群入口](../docs/operations/operations-case-doc.md)
