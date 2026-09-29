# Inference 环境准备

[返回入门指南](../../inference/README.md)

宿主 Python 负责读取配置和启动 Docker；模型、PyTorch、编译器及通信库在所选镜像内执行。
本仓接入已有兼容环境，不附带可直接拉取的完整 Inference 镜像，也不在运行时自动安装或升级设备依赖。
镜像 tag 可自行命名：启动器解析它的真实 image ID，并在整次运行中固定使用该 ID。

## 宿主与设备

- Linux；宿主 Python 3.10+ 和 `inference/requirements.txt` 中的 PyYAML。
- Docker 可访问本机守护进程；Ascend 主机有匹配 CANN 的驱动及 `npu-smi`；NVIDIA 主机有可用的容器 GPU 支持。
- 选用空闲物理设备，保证模型目录可读、输出目录可写。单卡只挂载选中设备，容器内使用逻辑设备 0。
- TP 需要单机 2、4 或 8 张不同设备及 HCCL 可用网络；支持参数数量不等于这些规模均已实机验证。入门从 TP=2 开始。

## 设备侧镜像要求

| 用途 | 镜像需要包含 |
|---|---|
| Ascend 模型执行 | Python 3.11、CANN 9、兼容的 PyTorch 与 torch_npu、Transformers、Accelerate、NumPy、PyYAML、Triton、FlagGems |
| FlagTree on | 提供 `triton` 模块的 FlagTree 分发包；与运行栈匹配 |
| TP FlagCX on | 可导入并注册 `flagcx` 进程组的 Ascend FlagCX 构建 |
| 图导出 | PyTorch export、ONNX、ONNXScript 及其依赖 |
| NVIDIA 单卡 | CUDA 版 PyTorch、Transformers、Accelerate、Triton、FlagGems 等对应 CUDA 依赖 |

当前实现所用的参考组合为 PyTorch 2.10.0、torch_npu 2.10.0、Transformers 5.5.3、
Accelerate 1.13.0、NumPy 1.26.4、PyYAML 6.0.3；三组件组合使用 FlagTree
`0.6.0+ascend.git15ec1a6c`、FlagCX 0.13.0 和 cann-shmem 1.6.0。
版本名本身不足以证明兼容：实际加载位置、包和源码摘要进入运行身份，新环境仍需自己的 preview 和精度检查。
即使所有开关为 off，准备阶段也会记录 Triton 身份；Ascend 默认还要有下面的厂商编译器资产。

Inference 直接使用 torch_npu；不要向其镜像混装抢占同一设备注册槽的 Torch-FL。
Base/Operation 的 Torch-FL 组合按 [Ascend 指南](../ascend/README.md) 管理。

## 本地模型目录

从模型发布方获得完整 **Qwen3-Embedding-0.6B** 权重及 tokenizer 文件，保留原目录结构。
当前适配读取 `config.json`、`tokenizer_config.json`、`model.safetensors`，还需模型随附的 tokenizer 词表等文件。
在配置 `model.path` 或参数 `--model-path` 中填写目录，不填写单个权重文件路径。
程序设置离线加载，不会在文件不全时自动联网补齐。该目录以只读方式挂载到容器。

## 准备厂商编译器

FlagTree 与厂商 Triton 都使用 `triton` 模块名。工具通过隔离进程和只读资产选择实际编译器，避免运行中替换已导入模块。
需要一个本地镜像，其中原生 `triton-ascend` 及 `triton` 分发元数据完整，且实际导入的不是 FlagTree。

```bash
# 在 inference 目录执行；替换为实际镜像。
python3 tools/prepare_vendor_compiler.py --image your-vendor-image:tag
```

工具无需设备或联网，会自动查询 Python 包位置，提取编译器及元数据，保存来源 image ID、包版本和文件 SHA-256。
默认目录为 `runtime_assets/ascend-triton`。每次运行检查清单与实际文件，策略还绑定资产摘要。
需要另一份资产时用 `--output /absolute/new/bundle`，并同步设置 `vendors.ascend.vendor_compiler`。
如果已在当前环境运行，模型、资产和策略也必须可读；使用 `run_inference.py` 替代 `run.py`，并在该环境重新 preview。
手工设置镜像标识不能代替环境一致性核验。

Ascend 默认统一设置 `TASK_QUEUE_ENABLE=0`、`TRITON_ENABLE_TASKQUEUE=true`。
这些值对比较双方相同；不要只在某一侧改变队列设置。FlagCX 的当前适配将原生组保留至一次性 worker 退出，
不适用于长驻服务反复创建/销毁进程组的稳定性结论。

## 常见问题

| 现象 | 处理 |
|---|---|
| `No module named yaml` | 激活宿主虚拟环境，安装 `inference/requirements.txt` |
| 要求设置 model.path/image | 填写用户配置，或传入 `--model-path` / `--image` |
| 模型文件缺失 | 核对完整模型目录、读权限和文件名；不要只下载配置文件 |
| 找不到编译器清单 | 运行资产准备工具，并核对配置的目录 |
| 编译器摘要不匹配 | 保留旧资产，重新导出到新目录；重新 preview |
| Docker 无权限或设备不可见 | 核对当前账号权限、驱动和所选物理编号；程序不自动换卡 |
| `npu-smi` 无进程，但容器设备不可用或返回 `EBUSY` | 检查普通容器能否访问选中设备；宿主可见不等于容器可用。保留失败记录，由操作者选择可用设备后重新 preview，勿用 CPU 回退掩盖问题 |
| 设备忙、HCCL 初始化失败 | 检查当前占用、设备映射及网络配置；不是精度判定 |
| 策略身份不匹配 | 查看 `policy-mismatch.json`，在当前源码、环境与输入下重新 preview |
| preview 预算不足 | 查看报告及 `preview/budget.json`，按进阶指南在新目录续探 |
| 运行失败 | 先读 `result.json` 的阶段与错误，再读对应 worker 的 `run.log` |
