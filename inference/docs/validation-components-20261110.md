# 成熟功能同步验证（2026-11-10）

本次将组件对照及精度、性能链路迁入公开目录，并保留公开仓库已有的厂商适配与修复。验证全部在 CPU 上执行：公共测试覆盖当前代码，归档复验检查同一份设备证据经过新代码后是否保持原数值和质量状态。功能用法见 [组件对比](components.md)，范围见 [迁移记录](migration.md#2026-11-10-成熟功能增量)。

## 来源与兼容处理

| 项目 | 版本或处理 |
|---|---|
| 公开基础 | `e47be5ff56b7f1f4d72da585b46562225e0e26ed` |
| 首次导出对应开发版本 | `ac23227c14a76a578449c216118a360496ad513f` |
| 本次成熟功能来源 | `748fb9a405ce6dc5f947f18feb2d284806c6cfb2`，共 20 个生产文件 |
| 后续独立提取 | `launch --preset coverage` 的 CLI 选项 |
| 公开适配保留 | P800、平头哥、NVIDIA，以及生命周期、输入身份、窗口时钟、OneRec TP 修复 |
| 显存采集兼容 | launcher 合并官方 profiler 参数与标准化组件配置，保留 `--profile-memory` |
| 通信快捷参数 | `--flagcx on/off` 用于 Ascend；其他适配继续使用显式 vendor/backend |
| Store 协议 | 仍为 1.2；monitor/watch、telemetry 与 schema 1.3 扩展未迁入 |

`record --tensor-boundaries model|layer|PATH` 和 `--tensor-ops-in` 在公开基础版本已存在。本次补齐 launch 参数、TP2 embedding runner 和组 Oracle，使两侧先以相同配置采集，再按同 rank 配对。模型/block 复制边界另做跨 rank 一致性检查；同一输入不因两个 rank 被计为两份样本。

## 当前代码回归

| 检查 | 结果 | 覆盖 |
|---|---|---|
| Python 3.9 标准库测试 | 81 项，70 通过、11 跳过 | 公共适配、生命周期、身份、组件、Oracle、CLI；可选 torch 环境测试跳过 |
| Python 3.11 标准库测试 | 81 项，70 通过、11 跳过 | 同上 |
| 固定镜像 CPU torch/Gloo | 14 项全部通过 | 精度采集、两进程 Qwen embedding、复制边界、失败停止执行、replay、通信事件 |
| 离线 wheel 构建与安装 | 通过 | 新编译器与组 Oracle 模块、四厂商契约 JSON、已安装 CLI |

标准库回归从 `inference/` 执行：

```bash
python3 -B -S -m unittest discover -s tests -v
```

可选运行环境为 Python 3.11.15、torch 2.10.0+cpu、Transformers 5.5.3；固定镜像 ID 为 `sha256:1147704dc7ddab371b5e8400525585d54a8be3d290d843e93cbe83552bb7bf03`。容器未映射设备，网络关闭，Gloo 使用容器回环网络。镜像中的 torch_npu 依赖宿主驱动库只读挂载；显式设置用户名称与 `/tmp` 缓存目录后运行：

```bash
python3 -B -m unittest tests.probes.test_accuracy tests.probes.test_components \
  tests.probes.test_replay tests.test_communication -v
```

早期镜像环境失败和修复前 `--profile-memory` 回归失败均保留在验证原件中。显存参数回归先复现 perf/modules 两种模式丢参，再验证修复；对应 [CLI 测试](../tests/test_component_cli.py)。

## 已归档设备证据复验

归档来自 2026-11-03 的 Qwen3-Embedding-0.6B × 910C 组件验收。用当前公共 `run.py report` 写入新的报告目录，逐项比较原报告的 result JSON 与导出 JSONL；仅归一化报告目录和已知归档路径别名。11 组报告、331 份结构化文件全部匹配，包含状态、质量原因和指标字段。

| 精度条件 | 独立重算张量对数 |
|---|---:|
| FlagTree model / layer / drill | 24 / 372 / 129 |
| FlagCX model / layer / drill（两 rank 合计） | 48 / 744 / 272 |
| 合计 | 1,589 |

独立校验先核对完整 blob 的 SHA256，再用 NumPy 重算 `max_abs` 与 `relative_l2`，逐项对上新报告。OneRec 的 HCCL/FlagCX 两侧、两个 rank，各 32 token 的轨迹与原验收一致。

性能四条件为 FlagTree off/on 单卡、FlagCX off/on TP2，共 6 个 Store。直接遍历公共 Store 事件重算设备事件数和耗时总和，与新报告一致；每个 Store 均为 1 次预热和 2 次测量前向，TP2 每 rank 共 168 次模型 AllReduce。该耗时总和用于检查报告算术，不代表去重后的设备占用或端到端延迟。

复验沿用历史覆盖：FlagCX 算子下钻的来源不完整项、性能格式与关联的 partial/outside 状态均保留。本次没有新增设备实验，没有据此扩大设备支持范围或给出稳态加速比。

工作区验证原件位于 `runtime-team/personal/evidence/inference-public-sync-20261110/`：`migration-inventory.json` 记录迁入文件及摘要；`archive-comparison.json` 记录逐文件比对；`independent-recomputation.json` 记录张量、事件和 token 复算；另保留完整命令脚本、测试日志及各次失败。历史 Store、模型、镜像及大型报告继续由开发工作区管理，不随公开代码发布。
