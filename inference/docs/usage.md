# 使用 flagprof

在仓库根目录执行 `python -m pip install ./inference`，也可安装构建好的 wheel。随后可以在任意目录运行 `flagprof --help`。离线操作使用 Python 3.9+ 标准库，采集需要在目标程序的 Python 环境中安装 FlagPerf 和相应设备 SDK。

## 选择结果位置

```bash
FP_WORKSPACE="$PWD/flagprof-results"
mkdir -p "$FP_WORKSPACE"
```

采集或导入用 `--store` 选盘。检查点和大索引放在该路径下面的用户私有目录，独立于正式 run 和 capture。相同封存输入和配置重试时会校验并复用检查点。完成或失败的临时工作由程序清理。查看 run 只需它的路径和可读的关联证据，每次查询使用单独的系统临时目录。生成报告可用 `--out` 单独选盘，排序、索引和报告文件都使用该盘。

后续操作会回收同一位置中断后遗留的私有临时目录。导入会清理 30 天未使用的检查点，或在可用空间低于盘容量 5% 与 2 GiB 中较小的值时，优先回收最久未用的检查点。正在使用的检查点和正式结果会保留。

监控、图导出和结果解读分别见[监控指南](monitoring.md)、[图导出](graph-export.md)和[结果报告](user-result-reports.md)。

## 第一次使用，先读一个样例

下面的命令只需已安装的 `flagprof` 和 Python。它们生成一条合成的 Chrome trace，用于练习封存、导入、查询和报告。请在新目录执行。

```bash
python -c 'import json; print(json.dumps({"traceEvents":[{"ph":"X","cat":"kernel","name":"example","pid":1,"tid":1,"ts":10,"dur":2,"args":{"device":0,"stream":1}}]}))' > trace.json
flagprof seal trace.json --normal-close --vendor nvidia --out capture
flagprof import capture --store results --run-id sample
flagprof inspect results/sample --verify
flagprof page results/sample --limit 1
flagprof report results/sample --out report
```

打开 `report/report.html` 可离线查看。报告将样例中缺少的硬件身份、采集配置和时钟证据标为未知。需要在另一台机器查看时，执行 `flagprof delivery publish results/sample --out archive`，复制整个 `archive`，执行 `flagprof delivery restore archive --out restored`。本例恢复结果中的 `mapping.run` 可以直接传给 `inspect`。

在装有 PyTorch 的环境，从仓库根目录用 `python -m pip install './inference[batch]'` 安装批量工具后，也可使用自带的 CPU CTR 示例。以下命令在同一工作目录生成固定输入、运行和比较两次推理。

```bash
flagprof-batch --example cpu > models.yaml
flagprof-make-ctr --samples 8 --seed 531 --out packages/ctr-v1
flagprof-batch --config models.yaml --filter ctr-dlrm --device cpu --out batch-results
```

YAML 中模型和输入的相对路径以该 YAML 所在目录为准。`flagprof-batch --example models` 和 `--example hf2` 提供可编辑的多模型名册。

## 采集自己的推理程序

在已准备的设备环境内，直接启动 Python 目标。`record` 自动注入的范围是它直接启动的 Python 进程：

```bash
flagprof record \
  --vendor ascend --preset perf \
  --store "$FP_WORKSPACE/capture" \
  -- python3 your_inference.py
```

NVIDIA 环境使用 `--vendor nvidia`。`perf/modules` 默认 `final` 采集完整会话，目标正常退出后独立导入；`coverage/perf-lite/accuracy` 转入窗口生命周期，显式 `--capture-mode windowed` 也走窗口路径。窗口采集需要目标推进 `handle.step()`，内置 runner 已提供相应边界。包装 shell、服务进程或自行再启动的子进程，需要在真正执行推理的进程使用 `flagprof.attach()`；接口见 [总体架构](02-architecture.md)。

平头哥 PPU 环境使用 `--vendor thead`（设备名为 CUDA 兼容的 `cuda:N`），适配器通过 `ppu-smi` 读取设备身份并按 UUID 关联。容器无法查询宿主时，可用 `FLAGPROF_THEAD_IDENTITY` 提供宿主清单；锁定镜像、最小命令、集合通信识别边界与验收状态见 [PPU Inference 适配](../../docs/thead/inference-v2.md)。

`--preset perf` 采集设备与主机性能；`modules` 面向模块；`perf-lite` 使用轻量主机探针；`coverage` 面向路由覆盖；`accuracy` 用于张量数值诊断。`--profile-memory` 增加官方显存记录；`--dispatch each-window` 为逐调用关联提供主机记录。探针对运行的扰动应通过匹配的无采集运行测量。

## 使用内置运行器

模型路径、输入与设备均显式提供；模型权重不随代码发布。embedding 输入 JSONL 每行是字符串或含 `text`、`query`、`input` 的对象：

```bash
flagprof record \
  --vendor ascend --preset perf --store "$FP_WORKSPACE/embedding" \
  -- python3 -m flagprof.runner hf:/absolute/path/to/model \
     --inputs /absolute/path/to/inputs.jsonl --batch-size 1 \
     --device npu:0 --dtype bfloat16 --outputs "$FP_WORKSPACE/runner-output"
```

`--inputs` 当前只取前 `batch-size` 行组成一个批次，`--forwards` 重复这个批次，不遍历其余行。需要多批次或固定数值身份时使用 `--inputs-from`；文件结构及合成包生成示例见 [固定输入包](../../docs/kunlunxin/inference-v2-validation.md#固定输入包生成)。

generation 使用 `--task generate` 和 `--generation-mode accuracy|performance`。双进程入口为 `flagprof launch --nproc-per-node 2 --comm-backend hccl|flagcx|nccl`，具体目标参数见 `launch --help` 与 `python -m flagprof.runner --help`。只有支持该运行器并匹配设备环境的模型可直接运行；其他目标保留自身加载逻辑，通过进程采集或 Python API 接入。

## 读取与报告

将采集或导入返回的 run 路径写入 `FP_RUN`：

```bash
export FP_RUN=/absolute/path/to/store/run
flagprof inspect "$FP_RUN" --verify --json
flagprof page "$FP_RUN" --offset 0 --limit 10
flagprof report "$FP_RUN" \
  --out "$FP_WORKSPACE/report"
```

报告生成于 `--out`，HTML 和 JSONL 可离线阅读。双 rank 报告将 `FP_RUN` 换成 launch 产生的 `group.json`。观察数据完整性、身份和覆盖率后，再解读耗时；设备时间、主机时间和请求窗口是不同口径。缺失关联或时钟证据会保留原因。

从已封存 capture 重新生成 Store：

```bash
flagprof import /absolute/path/to/sealed-capture \
  --store "$FP_WORKSPACE/reimport-runs"
```

`seal` 处理正常关闭的外部官方 trace；`analyse` 在带对应 SDK 的环境中分析已封存 raw capture。升级后的派生证据通过新导入生成，历史 Store 和报告保留原样。

## Oracle 与算子重放

`accuracy` 采集需注册模型、样本和输入身份；内置 runner 完成这些接入。任意脚本只启用 preset 不等于已建立跨 run 可比性。`record --preset accuracy` 默认不存张量 blob，完整误差与 replay 需显式 `--tensor-blob all` 并检查预算/跳过计数；`launch --preset accuracy` 默认保存 all，仍须核对实际覆盖。

`record` 和 `launch` 均支持以下采集范围，Python API 使用对应的 `tensor_boundaries` / `tensor_ops_in` 配置：

| 参数 | 实际采集 |
|---|---|
| `--tensor-boundaries model` | runner 提交的 pooled/output 模型终点 |
| `--tensor-boundaries layer` 或省略 | 默认 embedding、各 Transformer block 和模型终点 |
| `--tensor-boundaries PATH` | JSON 数组指定模块路径，另保留 runner 模型终点 |
| 以上模式加 `--tensor-ops-in '<root>.layers.0'` | 增加指定模块内算子输入与输出，要求 batch 1 |

名为 model/layer 的自定义文件使用 `./model` / `./layer`。采集层级与 `--tensor-blob` 的持久化策略分别设置；报告只分析已采集的数据。

FlagTree / FlagCX 参数、同规模 TP2 对比和性能命令见 [组件对比](components.md)。`report oracle --ref/--cmp` 也接受两侧 `group.json`：各 rank 同号配对，复制边界跨 rank 核对，逻辑样本只计一次。

```bash
flagprof report oracle \
  --ref /path/to/reference-run --cmp /path/to/compared-run \
  --out "$FP_WORKSPACE/oracle"
flagprof replay prepare \
  --ref /path/to/reference-run --cmp /path/to/compared-run \
  --call-key 'selected-call-key' --out "$FP_WORKSPACE/replay-input"
```

算子重放需要两侧完整张量 blob，生成包后在相应设备环境执行 `flagprof replay run "$FP_WORKSPACE/replay-input" --side ref --device "$DEVICE" --store "$FP_WORKSPACE/replay-runs"`。设置 `DEVICE` 为与该侧匹配的设备，对比侧使用 `--side cmp`。输出误差、传播误差和 replay 新引入误差按各自证据读取。详细契约见 [分析器与报告](07-analyzers-and-reports.md)。

## 设备最小验收

改动涉及采集、厂商适配、导入或报告口径时，在对应设备环境运行一次最小样本全流程，并在 PR 中附上命令、被测 commit 与下列检查结果。以下用一条输入、batch 1 的 embedding 样本；默认重复同一批三次，不据此声称多样本覆盖：

```bash
printf '%s\n' '"FlagPerf inference sample one."' \
  > "$FP_WORKSPACE/accept-inputs.jsonl"
flagprof record \
  --vendor ascend --preset perf \
  --store "$FP_WORKSPACE/accept" \
  -- python3 -m flagprof.runner hf:/absolute/path/to/Qwen3-Embedding-0.6B \
     --inputs "$FP_WORKSPACE/accept-inputs.jsonl" --batch-size 1 \
     --device npu:0 --dtype bfloat16 --outputs "$FP_WORKSPACE/accept-runner"
```

NVIDIA 环境把 `--vendor ascend` 换成 `--vendor nvidia`，`--device npu:0` 换成 `--device cuda:0`。将 record 输出中的 `run` 写入 `FP_RUN` 后读取并生成报告：

```bash
flagprof inspect "$FP_RUN" --verify --json
flagprof report "$FP_RUN" \
  --out "$FP_WORKSPACE/accept-report"
```

| 检查点 | 预期 |
|---|---|
| record 输出（最后一行 JSON，同时写入 capture 下的 `record-status.json`） | `status` 与 `import_completion` 为 `complete`，`exit_code` 为 0 |
| `inspect --verify --json` | 每条 trace 的 `import_status` 为 `complete`，`integrity` 为 `verified` |
| `report` 的 `result.json` | `perf.device` 为 `ok` 或 `partial`；其余 panel 为 `unavailable` 时附带原因（`notes`） |
| `report.html` | 可离线打开，设备头条与覆盖率与 `result.json` 一致 |

`partial`、`unknown`、`unlinked` 是带原因的有效结论；验收记录写明原因，不改写为通过。

## CPU 冒烟测试

无设备、无 torch 的环境中，在 `inference/` 下执行：

```bash
python3 -B -S -m unittest discover -s tests -v
```

测试检查离线读取层（`store`、`analyzers`、`report`）不加载 torch 与厂商模块，并用假厂商插件走通 record → import → inspect → report；同时覆盖组件选择、精度 CLI、Oracle、TP2 组计数及 P800/平头哥适配。PR 修改 `inference/**` 时由 GitHub Actions 在 Python 3.9 和 3.11 自动运行。

完整 CPU 套件的固定依赖已在 Python 3.11 验证，使用该版本的解释器创建独立虚拟环境（不继承厂商镜像的 site-packages），直接依赖版本记录在 `tests/requirements-cpu.txt`。在 `inference/` 下执行：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r tests/requirements-cpu.txt -r tests/requirements-graph.txt
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider tests
```

GitHub `model-contracts` 作业执行同一套完整回归，包含模型／逐层／算子边界、两进程 Gloo TP2、Oracle、replay 和训练。TP2 使用本地生成的微型 Qwen，要求 Accelerate；Triton 用于编译器探针契约，不执行设备 kernel，不下载模型。

厂商镜像中即使设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，Transformers 仍可能独立探测并导入 `torch_npu`；未挂载昇腾驱动时会在模型导入阶段报 `libascend_hal.so` 缺失。CPU 测试应使用上述独立环境；实际 NPU 执行仍使用厂商镜像及其驱动。编译器选择测试在独立子进程中运行，保留来源冲突检查，也不会改变父进程已经加载的 Triton。

这些 CPU 检查和历史 Store 的报告复验不扩大既有设备验收范围。

## 归档与恢复（可选）

`flagprof delivery publish|verify|restore` 把 Store、capture 等目录打包成带逐文件摘要的归档，并在新目录恢复相对依赖。恢复命令使用用户选定的新目录，返回对象 ID 和本次恢复路径的映射。

## P800 候选运行栈

在目标程序已有的 XPyTorch 环境安装 FlagPerf。单卡采集使用已分配的逻辑设备和选定的结果目录。例如，从上面的工作目录运行：

```bash
flagprof record --vendor kunlunxin --store "$FP_WORKSPACE/p800" -- python my_inference.py
```

适配器在创建采集进程时设置 XPyTorch tracing 变量。多卡使用 `launch --vendor kunlunxin --nproc-per-node 2 --comm-backend nccl` 等受支持的组合，适配器为每个 rank 设置对应的逻辑设备。只有直接在应用内调用 `attach` 时，才需要在加载 torch 前设置 `XPU_ENABLE_PROFILER_TRACING=1` 和所选逻辑卡号 `XPU_CUPTI_ENABLE_DEVICE`。

适配器调用 `xpu-smi` 并按框架报告的 UUID 核对设备。容器无法访问宿主 SMI 时，可以将宿主查询得到的清单只读挂载，并把 `FLAGPROF_P800_IDENTITY` 指向该文件。清单字段、既有实测镜像和 TP2 依赖见 [P800 适配资料](../../docs/kunlunxin/inference-v2.md)。采集返回的 run 可以直接交给本页的 `inspect`、`page` 和 `report` 示例。

既有实测记录了 OneRec 单卡与 TP2 在第 14 个生成 token 出现差异，TP2 精度仍待查明。可核对[验证记录与开放缺陷](../../docs/kunlunxin/inference-v2-validation.md)。
