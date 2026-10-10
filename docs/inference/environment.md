# 环境准备与排错

宿主启动器只依赖 Python 3.10+ 和 PyYAML。设备侧需要用户准备好的镜像、驱动和模型；程序不会安装设备依赖或下载权重。

| 环境 | 依赖与用途 |
|---|---|
| Ascend 被测端 | Linux、Docker、CANN 9、PyTorch 2.10、torch_npu；文本模型另需 Transformers、tokenizers、safetensors、NumPy |
| FlagGems 对照 | 兼容 FlagGems 与 Triton 实现；按当前身份先 preview |
| FlagTree 对照 | 兼容 FlagTree，固定其他组件并重新联合 preview |
| FlagCX 对照 | TP 中的兼容 FlagCX；组生命周期限定为单次 worker |
| A100 参考端 | 已准备 CUDA 环境，PyTorch 2.10.0+cu126、Transformers 5.5.3、NumPy 1.26.4；支持单卡 oracle 和 single/TP2 性能基线 |
| 完整离线分析 | CPU PyTorch、NumPy、safetensors、PyYAML；无加速设备需求 |
| CPU 图导出 | PyTorch、NumPy、PyYAML 和模型加载依赖；ONNX 另需 onnx、onnxscript；不初始化设备 |

Ascend 路径使用 torch_npu。Operation 的 Torch-FL 环境要求不能直接代替推理环境。vLLM 仅可作为镜像底座，不参与这里的模型执行。
CANN profile 的约束为 `TASK_QUEUE_ENABLE=0`；容器运行前解析并记录镜像完整 SHA256 ID，组件和包身份在 worker 中核对。

## 准备模型

Qwen3-Embedding 和 OneRec 使用完整本地权重及 tokenizer。OneRec 要求标准 Qwen3 CausalLM 配置、BF16 权重分片和 generation_config，不执行模型目录内的自定义 Python。
DeepFM/DLRM 通过[模型指南](models.md)中的 CPU 工具生成权重和合成输入，目录生成一次后封存。生成资产不随源码发布。

宿主的 launcher 虚拟环境只有 PyYAML，不能直接执行模型资产生成或完整分析。若宿主已有 CPU PyTorch 和 safetensors，可直接运行模型指南中的准备命令；也可以复用含这些依赖的兼容镜像。
例如，从 `inference/` 准备 DeepFM，仅挂载源码和资产输出，不挂载设备或驱动：

```bash
mkdir -p runtime_assets
docker run --rm --network none --read-only --tmpfs /tmp:rw,exec,size=1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=flagperf -e LOGNAME=flagperf \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:$PWD:ro" -v "$PWD/runtime_assets:$PWD/runtime_assets:rw" -w "$PWD" \
  --entrypoint python3 your-compatible-image:tag \
  -m benchmarks.deepfm.prepare_assets --output runtime_assets/deepfm-v1
```

DLRM 将最后一行替换为 `-m benchmarks.dlrm.prepare_assets --output runtime_assets/dlrm-v1`。
用于完整离线分析的环境还需 NumPy。图导出只需 CPU 依赖；文本模型另需 Transformers、tokenizers、safetensors 和本地 tokenizer/权重。

## 准备原生编译器

三个组件均 off 的原生 accuracy/performance 不需要编译器包。preview 或需要原生 Triton 一侧的组件比较，执行：

```bash
python3 tools/prepare_vendor_compiler.py --image your-native-triton-ascend-image:tag
```

工具根据镜像中实际安装位置提取包，不假定 Python 前缀；拒绝覆盖已有目录和将 FlagTree 当作原生编译器。
`runtime_assets/ascend-triton/manifest.json` 记录来源镜像 ID 和逐文件摘要，运行前重新校验。可用 `--output` 选择新目录，并修改 `vendors.ascend.vendor_compiler`。

## A100 环境工具

在用户已有的 CUDA 容器内，使用 Python 3.12 创建隔离环境；两个脚本都会检查 Python 版本并联网安装锁定依赖：

```bash
bash tools/prepare_oracle_env.sh
bash tools/prepare_performance_env.sh
```

环境默认位于 `runtime_assets/a100-venv` 和 `runtime_assets/a100-performance-venv`，已有目录会拒绝覆盖。可传第一个位置参数选择新环境目录。
脚本不会激活当前 shell；后续分别使用对应的 `bin/python`，见[A100 指南](references.md)。离线机器需预先准备兼容独立环境，再使用该环境的 Python；这些脚本不提供离线安装流程。

## 常见问题

| 现象 | 检查方式 |
|---|---|
| 未填写模型或镜像 | 设置 `model.path` / `container.image`，或传 `--model-path` / `--image` |
| 模型加载失败或输出非有限 | 检查完整权重、分片、tokenizer、dtype；核对来源摘要，保留失败现场 |
| 设备访问失败或资源忙 | 检查 Docker 权限、设备文件和当前占用，选择可用设备并重新生成身份 |
| 编译器 manifest 不符 | 重新从兼容的原生编译器镜像提取到新目录，不编辑摘要绕过校验 |
| 策略身份不符 | 按当前模型、输入、镜像、设备/rank 和组件组合重新 preview |
| monitor 有缺失指标 | 查看 monitor 状态、有效点数、provider 能力及访问权限；缺失值不会填零 |
| CUDA trace 分析超时 | 保留原始证据，在新目录用 `analyze` 或 `report --reanalyze` 恢复 |
| 重分析缺少 tensor/trace | 原任务需保留完整证据；已经精简的数据无法凭报告恢复 |

`--dry-run` 只估计阶段、输入与资源成本；它不验证实际设备和镜像可运行性。手工容器使用 `run_inference.py` 时应在自身环境重新 preview，环境身份必须与策略一致。
