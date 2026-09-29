# Inference：模型精度与性能分析

Inference 在同一组设备和输入上运行模型，帮助你回答两个问题：**更换组件后输出变化多大？模型运行时间和显存占用怎样变化？**
当前模型为 **Qwen3-Embedding-0.6B**，它将文本转换成向量；执行方式是 PyTorch + Transformers eager。
Ascend 支持单卡与单机张量并行（TP：把同一模型分到多张卡）；NVIDIA 提供单卡接口，尚未完成实机验证。

| 要做的事 | 命令 | 得到什么 |
|---|---|---|
| 查看能力与限制 | `list` | 当前模型、厂商和功能清单 |
| 记录输出、比较数值 | `accuracy` | 模型输出或指定模块输出的差异 |
| 测量模型整体或逐层性能 | `performance` | 时延、吞吐、显存；TP 另有通信采样 |
| 找到可用于当前环境的 FlagGems 函数集合 | `preview` | 探测记录、可恢复检查点、通过复验的策略 |
| 筛选或打包已有性能报告 | `report` | total/layer 视图、模块/rank/shape 筛选及实体证据包 |
| 导出模型图 | `export` | FX 或 ONNX 图及检查结果 |

组件开关 `off` 表示使用对应的默认实现，`on` 表示启用该组件，`both` 表示分别运行两侧进行比较。
**FlagGems** 提供算子实现，**FlagTree** 提供 Triton 编译器，**FlagCX** 提供集合通信后端。
一次最多比较一个组件；其他组件保持固定。具体限制见 [支持清单](support.json)。

## 1. 准备环境

需要 Linux 主机、Python 3.10 或更新版本、Docker 使用权限、可用设备驱动和本地模型权重。
设备侧依赖放在你已准备的兼容镜像中。先按 [环境准备](../docs/inference/environment.md) 确认镜像和权重，特别注意：
**Inference 的 Ascend 路径使用 torch_npu，Operation 的默认 Ascend 路径使用 Torch-FL，它们的镜像要求不同。**

从 FlagPerf 仓库根目录开始：

```bash
python3 -m venv .venv-inference
. .venv-inference/bin/activate
python3 -m pip install -r inference/requirements.txt
cd inference
cp config/default.yaml config/local.yaml
```

编辑 `config/local.yaml`，填写实际值，其余参数先保持默认：

```yaml
model:
  name: qwen3_embedding_0.6b
  path: /absolute/path/to/Qwen3-Embedding-0.6B
  dtype: bfloat16
  attention: eager
container:
  image: your-local-compatible-image:tag
  shm_size: 4g
```

上面是需要修改的两个区块；在复制出的完整配置中修改它们即可。镜像名是占位示例，不是可直接拉取的发布物。
`config/local.yaml` 已被 Git 忽略，方便保存本机路径。

Ascend 还需从包含原生 `triton-ascend` 的本地镜像准备编译器资产。
将下面的镜像名替换为实际来源；它可以与运行镜像不同，但必须与运行栈兼容：

```bash
python3 tools/prepare_vendor_compiler.py --image your-vendor-compiler-image:tag
python3 run.py list
```

资产默认放在 `runtime_assets/ascend-triton/`，不进入版本库；同一路径不重复覆盖。
NVIDIA 不需要这一步，需在配置中选择 `runtime.vendor: nvidia` 和兼容 CUDA 镜像。

## 2. 完成第一次运行

先使用原生路径，不需要策略。下面的物理设备 `0` 应替换为当前空闲设备；Ascend 可先用 `npu-smi info` 查看占用。

```bash
python3 run.py accuracy --config config/local.yaml --device 0 \
  --flaggems off --output result/native
```

终端最后打印 `Results:`。打开该目录的 `report.md`，先看执行状态、模型与输入条件，再看输出记录。
这里只有单路径结果，没有 off/on 差异。`result.json` 保存结构化结果，失败原因和对应日志入口也在其中。

程序离线加载权重，不自动下载模型。示例输入为 12 条文本；每条有稳定的 ID，详见 [输入格式](inputs/README.md)。
输出目录必须尚不存在，重跑请换一个目录名。

## 3. 比较 FlagGems 开关前后

先运行 preview，找到当前镜像、设备、模型和输入下可用的函数集合：

```bash
python3 run.py preview --config config/local.yaml --device 0 \
  --output result/preview
```

preview 会多次启动 worker，耗时比一次模型运行更长。可加 `--budget-seconds 300` 先做有限探测；
该预算不包含准备、缓存导入和清理，复验未完成时不会发布 verified 策略，可按 [Preview 指南](../docs/inference/preview.md)续探。

在报告中确认策略为 `verified` 后，精度和性能可使用同一份策略：

```bash
python3 run.py accuracy --config config/local.yaml --device 0 \
  --flaggems both --policy result/preview/preview/policy.yaml \
  --levels model layer --output result/accuracy

python3 run.py performance --config config/local.yaml --device 0 \
  --flaggems both --policy result/preview/preview/policy.yaml \
  --output result/performance
```

默认精度比较同一设备配置上的原生输出与 FlagGems 输出；它不是 NVIDIA 对照，也不是 CPU FP64 真值比较。
`verified` 表示当前输入下的执行、函数命中及有限值边界检查通过，**不表示应用精度已经达标**。
性能主时延从输入已在设备上开始，覆盖 forward、pooling、归一化及最终同步；加载和搬运另列。
更多解释见 [如何读结果](../docs/inference/results.md)。

## 4. 找到具体耗时层

```bash
python3 run.py performance --config config/local.yaml --device 0 \
  --level layer --layers all --output result/layer-native
```

层级模式会独立运行整模型基准、层计时、profiler 和显存采集。先读 `report.md` 的整体变化，再进入 `layer-index.md` 查看具体层。
如何选择模块、切换视图、筛选和打包报告，见[逐层性能指南](../docs/inference/layer.md)。

## 接下来做什么

- [配置参考](../docs/inference/configuration.md)：输入、层选择、采样参数、设备与覆盖规则。
- [Preview 与续探](../docs/inference/preview.md)：分组搜索、预算、缓存和 unknown 原因。
- [指标来源](../docs/inference/metrics.md)：采集 API、公式、原始字段与适用边界。
- [进阶使用](../docs/inference/advanced.md)：TP、FlagTree/FlagCX 比较、preview 续探、图导出、报告重建。
- [环境准备与排错](../docs/inference/environment.md)：依赖、模型目录、设备权限及策略失配。
- [目录与扩展边界](../docs/inference/inference-case-doc.md)：各目录职责和执行流程。
- [旧版迁移说明](../docs/inference/migration.md)：旧模型、编译引擎及 SSH 流程的历史恢复方式。
