# P800 Inference v2 适配与验证

当前为候选适配，主要已测运行链路可完成；**OneRec 单卡与 TP2 第 14 个 token 不同（P800-INF-007），根因未确认，TP2 精度尚未通过，性能保持 `not_qualified`。**使用 `--vendor kunlunxin`；CUDA 兼容设备名 `GPU` 不代表 NVIDIA 硬件。Base、Operation 的资格不自动转移到 Inference。

先读 [从空工作区到报告](inference-v2-quickstart.md)（容器、输入、基线、采集、结果判读），再读 [验证结果与开放缺陷](inference-v2-validation.md)（历史实测、证据可得性、P800-INF-001–007）。本文说明固定栈与适配机制。

## 运行环境

已测镜像为 `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`，固定 image ID：

```text
sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a
```

容器解释器使用 `/root/miniconda/envs/python310_torch29_cuda/bin/python`，默认解释器不保证匹配。已测 Python 3.10.18、PyTorch 2.9.0+cu129、Transformers 4.57.1、FlagGems 4.2.1rc0、FlagTree 0.6.1+xpu3.6。SMI 查询字段分别记录 driver `5.0.21.47`、XPU-RT `10.2`，不与 SMI 页眉版本混用。

采集进程必须在导入 torch 前设置：

```bash
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_CUPTI_ENABLE_DEVICE=0
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

`XPU_CUPTI_ENABLE_DEVICE` 是容器逻辑卡号。TP2 launcher 为每个 worker 设置对应 rank；不得把宿主物理编号直接写作逻辑编号。只设置第一个变量可能退出 0 却没有设备轨；因此还需检查真实设备事件与导入账本。

## 设备身份与隔离

使用宿主 `xpu-smi -m`、`xpu-smi -i PHYSICAL_ID -q` 核对 P800 OAM、PCI BDF、UUID、minor、占用和健康状态。按查询得到的 `/dev/xpuN` 和 `/dev/xpuctrl` 映射选定卡，不能根据 SMI 编号猜 minor。运行前通过调度器或站点租约机制保留设备，运行后核对进程表与显存回收；Inference CLI 不会自动获取 Base 的 DeviceLease。其他用户的容器与进程不能作为清理对象。

将宿主查询所得身份写入 JSON 并只读挂载到容器，设置 `FLAGPROF_P800_IDENTITY` 为容器内路径。接口结构如下；字段值必须来自本机查询：

```json
{"source":"xpu-smi","devices":[{"model":"P800","uuid":"<queried UUID>","pci_bdf":"<queried BDF>","driver":"5.0.21.47","xpu_smi_runtime":"10.2","host_device_node":"/dev/xpuN"}]}
```

`devices` 按容器逻辑顺序排列，适配器逐卡核对 torch 返回的 UUID。JSON 是操作者提供的测量记录，不是硬件证明签名；应同时保留原始 SMI 输出、容器参数和哈希供审查。

## 最小运行

从 `inference/` 执行，按 [公共使用说明](../../inference/docs/usage.md) 准备有足够空间的 ext4 输出盘、TMPDIR 和 scratch。模型权重单独准备，不随仓库分发。以下是命令接口速查，路径、环境变量和输入文件需先按 [快速开始](inference-v2-quickstart.md) 准备：

```bash
python -B -m flagprof record --vendor kunlunxin --preset perf \
  --timeout 600 --store "$FP_WORKSPACE/qwen-perf" --dispatch each-window \
  -- python -B -m flagprof.runner hf:/models/Qwen3-Embedding-0.6B \
  --inputs /fp/texts.jsonl --batch-size 1 --max-length 32 \
  --dtype bfloat16 --attention eager --device cuda:0 --warmup 0 --forwards 2 \
  --outputs "$FP_WORKSPACE/qwen-output"
python -B -S -m flagprof inspect /absolute/path/to/run --verify --json
python -B -S -m flagprof report /absolute/path/to/run --out "$FP_WORKSPACE/qwen-report"
```

`python` 在以上命令中指固定镜像解释器。默认 `perf/modules` 采用 final 完整会话；其他预设和显式 windowed 按 step 采集。`modules` 开启完整 stack trace，真实模型 trace 可超过默认 1 GiB；必须事先核算原始文件、封存副本、索引和派生数据的总空间，再显式设置 `--source-max-bytes`。不能只按原始 trace 大小估算磁盘，也不能因存储中止修改原失败。

`capture-only → analyse → import` 支持将采集和离线处理分开；每次重试使用新 run，原 capture 和失败目录保留。离线 `inspect/page/report/delivery` 不要求导入 torch 或厂商 SDK。

## 模型与诊断范围

执行策略的模型/库、项目 runner、操作者及后端职责见 [OneRec 执行策略说明](inference-v2-execution-policy.md)。当前 runner 手工编排生成，不能自动声称等同模型官方 `generate()`；分析不以改变策略取得一致。

- Qwen3-Embedding-0.6B：真实前向、五预设、固定输入 accuracy、算子下钻、allocator memory、有限 FlagGems include 与报告均已有设备实测。
- OneRec-8B：使用既有 Qwen3 架构约束、batch 1、BF16/eager、32 步 greedy 生成；accuracy 遵循 EOS 或上限，performance 固定步数。保留 prefill/decode、token 和 KV 释放记录。
- 模型资产可从公开来源下载，但须记录文件 revision、字节数与 SHA。新建的公开验证输入包必须使用自己的 oracle ID；不能冒充历史 NVIDIA/Ascend reference 包。
- Replay 继续只接受公共实现的 mean.dim、softmax.int、rsqrt、pow.Tensor_Scalar，且需要同构身份、完整 blob 和已经不同的真实输入。SiLU 的拒绝是当前范围边界。
- TP2 使用 `launch --vendor kunlunxin --comm-backend nccl|flagcx --nproc-per-node 2`，模型组与 Gloo 控制组分别记录。容器需要可用的网络接口、足够共享内存和 memlock 配置；`--network=none` 已实测导致 BKCL 初始化失败。双卡 collective smoke 不代替 OneRec TP2 模型验收。


**TP2 为候选诊断路径，P800-INF-007 未闭合前不声明其精度通过。** 固定镜像缺少 Transformers TP 所需的 `accelerate`。已测独立依赖目录为 `accelerate==1.11.0`，wheel SHA-256 `a628fa6beb069b8e549460fc449135d5bd8d73e7a11fd09f0bc9fc4ace7f06f1`；可用固定解释器 `pip install --no-deps --target /path/to/overlay /path/to/verified.whl` 在独立目录准备，先核对其余依赖；只读挂载为 `/overlay`，设置 `PYTHONPATH=/code/inference:/overlay`，不覆盖镜像内 torch。TP2 容器另需第二张已保留卡及身份 JSON 第二项。Transformers 4.57.1 通过模型 `tp_plan` 属性接入显式计划，词表头使用 `colwise_rep`（列分片并汇聚完整输出），每 rank 核验本地权重形状 `[88192,4096]`。这证明分片实际发生，不证明单卡与 TP2 精度等价。其他版本保留原显式计划接口。

P800-INF-007 的 BF16 logits 并列舍入已从历史 blob 定位；离线重投影仅作诊断。曾增加的 FP32 词表头策略已撤回，当前保持原 BF16 执行，差异如实记录，见 [缺陷记录](inference-v2-validation.md)。

```bash
python -B -m flagprof launch --vendor kunlunxin --comm-backend flagcx \
  --nproc-per-node 2 --preset accuracy --timeout 1500 --store "$FP_WORKSPACE/onerec-tp2" \
  -- python -B -m flagprof.runner hf:/models/OneRec-8B --task generate \
  --inputs-from /fp/onerec-package --batch-size 1 --max-length 256 \
  --max-new-tokens 32 --dtype bfloat16 --attention eager --device cuda:0 \
  --warmup 0 --forwards 1 --generation-mode accuracy --outputs "$FP_WORKSPACE/onerec-output"
python -B -S -m flagprof report "$FP_WORKSPACE/onerec-tp2/group.json" \
  --out "$FP_WORKSPACE/onerec-tp2-report"
```

输入包须带文件哈希、模型架构与生成协议，可执行生成示例见 [固定输入包生成](inference-v2-validation.md#固定输入包生成)，语义设计见 [模型接入](../../inference/docs/06-model-onboarding.md)。检查 `group.json` 执行状态和报告 `result.json` 语义状态；报告命令退出 0 也可能保存 `partial`。32 步 OneRec TP2 预期每 rank 2304 次模型 AllReduce、32 次词表 AllGather；控制组独立计数。

## 时钟验证

固定 P800 exporter 已独立验证三个会话、每次 20/100/250 ms 三个 `record_function` 标记。记录调用外与内部的 `time.time_ns()` 及 RAW 前后采样；九个导出标记的起止均落在独立墙钟区间内。验证公式为 `wall_ns = exported_ts_us * 1000 + baseTimeNanoseconds`，原 trace 与 SHA 同时保留。证据只适用于上述锁定栈。

适配器仅在 collector 版本匹配且导出基时唯一时提供 RAW→wall 分段插值，不外推；缺采样、不单调、身份不匹配或采集不完整均保留 unavailable/partial。完整请求需连续生成步和最后同步完成证据。设备事件保留来源时钟；`clock_valid:false` 与未知同步误差不因主机映射验证而升级。

## 结果解释与已知限制

独立 P800 格式契约复用已观察到的 Kineto 字段结构；不复用 NVIDIA 的硬件能力、时钟转换或完整性结论。Exporter 未显式提供的配置、丢失计数和完整性保持 unknown。原生 kernel 归属与项目 signature/layer 关联分别统计，不能把前者当作后者。

请求 gap 使用上述受限时钟桥，逐请求报告完整性和裁剪原因。通信类别使用明确 collective 元数据，或独立 float AllReduce 与真实 BF16 TP2 验证过的有限名称族 `^void kl3_all_reduce<(?:float|bfloat16)>\(`；普通 reduce/gather、未知 dtype 和泛化 NCCL 名称均不据此分类。类别识别不生成调用关联。设备到 collective 的归属仍要求 launch flow 与同线程已登记标记的严格连接，不能用邻近时间补造。functional 与 imperative collective 的进程组、序列和逻辑字节数分别记录；主机入队耗时不等于设备通信耗时。Allocator 记录不等于整卡 HBM；占用率等 NVIDIA 兼容属性不能当作 P800 硬件计数器。

固定镜像采用旧 FlagGems `runtime.register.Register` 接口，组件兼容该接口并保留新接口路径。只注册明确 include，分别记录实际 route hit、fallback/excluded/error；启用标志或命令退出 0 不代表算子已经走 FlagGems。编译探针只声明结构 listener 能观察到的 cache hit/miss，无法据耗时推断内存缓存复用或编译失败。

测试结果与尚未满足的资格见验证记录。任何失败、中止、unknown 和 partial 均保留，不能据命令退出 0 或候选适配提交宣称完整性能资格。

## 验证、缺陷与维护

统一在 [验证结果与开放缺陷](inference-v2-validation.md) 维护实测范围、回归计数、P800-INF-001–007、单卡/TP2 分歧及证据可得性，避免多份状态表漂移。`status=ok` 表示分析可用；数值是否相同必须读取具体误差，group complete 只说明组执行完成。

更换镜像、驱动、模型、输入、采集器或源码后，记录新身份并复测受影响范围，保留旧失败与新目录。设备或输入不一致时拒绝比较，不修改容差/原始证据使其通过；新公开输入包使用独立 oracle ID。capture loss、设备时钟误差和 BKCL 关联缺口须有新证据才可改变状态，不自动晋级。
