# Inference：模型精度、性能与运行监控

Inference 使用固定模型和输入，比较原生实现与 FlagGems、FlagTree、FlagCX 的输出及性能，也可与预先生成的 A100 参考包比较。
当前提供 Qwen3-Embedding-0.6B、DeepFM、DLRM、OneRec-8B；被测端为 Ascend 单卡或单机 TP，A100 用于精度与性能参考采集。

| 目标 | 命令 | 结果 |
|---|---|---|
| 查看功能及限制 | `list` | 模型、厂商、并行规模与能力清单 |
| 比较模型或模块输出 | `accuracy` | 同设备差分；可选 A100 精度比较 |
| 测量总体或逐层性能 | `performance` | 时延、吞吐、显存、通信及独立监控 |
| 生成当前环境的组件策略 | `preview` | 探测记录、检查点和 verified 策略 |
| 筛选、打包或重绘性能报告 | `report` | total/layer 视图、筛选和 portable 报告 |
| 导出模型图 | `export` | 模型支持的 CPU 原生 FX/ONNX 图 |
| 生成 A100 参考 | `oracle` / `performance-baseline` | 可复用的精度/性能参考包 |
| 恢复已有采集的分析 | `analyze` | 新目录中的分析、比较及符合条件的封包 |
| 转换历史参考包 | `repack-reference` | 独立核心包与诊断附件 |

`off` 使用原生实现，`on` 启用组件，`both` 运行两侧比较。三个组件默认均为 off，一次最多比较一个组件。
实际支持范围见 [support.json](support.json)；模型差异见[模型指南](../docs/inference/models.md)。

## 1. 准备环境

需要 Linux、Python 3.10 或更新版本、Docker 使用权限、设备驱动、本地权重和兼容镜像。
Ascend 推理使用 **torch_npu + CANN 9 + PyTorch 2.10**；vLLM 镜像可作为环境底座，被测引擎仍为 PyTorch eager。
镜像由用户自行准备，公开配置不绑定本机镜像或模型目录。完整要求见[环境与排错](../docs/inference/environment.md)。

从仓库根目录执行：

```bash
python3 -m venv .venv-inference
. .venv-inference/bin/activate
python3 -m pip install -r inference/requirements.txt
cd inference
cp config/default.yaml config/local.yaml
```

在复制出的完整配置中修改这两个区块，填入真实路径和镜像：

```yaml
model:
  name: qwen3_embedding_0.6b
  path: /absolute/path/to/Qwen3-Embedding-0.6B
  dtype: bfloat16
  attention: eager
container:
  image: your-compatible-local-image:tag
  shm_size: 4g
```

镜像名为占位示例。模型离线加载，不自动下载；`config/local*.yaml` 和运行资产已被 Git 忽略。
DeepFM/DLRM 先生成封存资产，再使用对应配置；宿主 launcher 环境不包含生成资产的 PyTorch 依赖，准备命令见[模型指南](../docs/inference/models.md)。

## 2. 完成第一次原生运行

将设备 `0` 替换为当前可用的物理设备号，输出目录必须尚不存在：

```bash
python3 run.py list
python3 run.py accuracy --config config/local.yaml --device 0 --output result/native
python3 run.py performance --config config/local.yaml --device 0 --output result/performance-native
```

原生运行无需准备 Triton 编译器包。终端最后打印 `Results:`；先检查该目录的 `result.json` 状态，再打开 `report.md`。
性能默认约每秒独立采样监控，曲线入口为 `monitor.html`；关闭用 `--monitor off`。
随仓文本输入共 12 条，格式见[输入说明](inputs/README.md)。

## 3. 比较组件

使用组件比较前，从含原生 triton-ascend 的兼容镜像提取并封存编译器：

```bash
python3 tools/prepare_vendor_compiler.py --image your-native-compiler-image:tag
python3 run.py preview --config config/local.yaml --device 0 --output result/preview
python3 run.py accuracy --config config/local.yaml --device 0 --flaggems both \
  --policy result/preview/preview/policy.yaml --levels model layer --output result/accuracy
python3 run.py performance --config config/local.yaml --device 0 --flaggems both \
  --policy result/preview/preview/policy.yaml --level layer --output result/layer
```

正式比较只接受当前身份下的 verified 策略。更改执行源码、镜像、模型、输入、设备、TP 或组件组合后需要新 preview。
verified 表示选定函数执行、命中和有限值检查通过，不代表业务精度、硬件 kernel 全覆盖或稳定加速。

普通任务成功、分析完整且无异常时，默认精简原始 tensor/trace，保留指标、固定输入、日志、配置和源码。
诊断时使用 `--keep-tensors` 或 `--keep-evidence`；失败、partial、异常和参考生产任务保留完整现场。
精简后仍可阅读和重绘报告，缺失原始数据时无法重新计算相应指标。见[结果指南](../docs/inference/results.md)。

## 4. 继续使用

- [模型指南](../docs/inference/models.md)：四模型准备、输入、输出与 TP 范围。
- [A100 参考与跨设备比较](../docs/inference/references.md)：生产、搬移及消费核心包。
- [Preview](../docs/inference/preview.md)：预算、续探、联合比较及 unknown 原因。
- [逐层性能](../docs/inference/layer.md)：模块选择、筛选、通信与 portable 报告。
- [监控](../docs/inference/monitor.md)：曲线、指标口径和后端扩展。
- [离线恢复](../docs/inference/offline.md)：成本预检、analyze 和历史包转换。
- [配置参考](../docs/inference/configuration.md)、[指标来源](../docs/inference/metrics.md)、[进阶使用](../docs/inference/advanced.md)。
- [目录与扩展](../docs/inference/inference-case-doc.md)、[版本迁移](../docs/inference/migration.md)。
