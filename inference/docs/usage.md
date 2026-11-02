# 使用 flagprof

命令从 `FlagPerf/inference` 执行，先设置 `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`。`python -m flagprof --help` 是当前已实现命令的入口；00–12 文档同时包含设计方案和历史阶段记录。

## 准备输出目录

默认 final 路径的 capture、Store、work、scratch 要求实际文件系统为 ext4。选择本机有空间的 ext4 盘；目录名或软链接本身不能证明文件系统类型。以下路径由使用者替换，运行后保留打印出的 run 路径：

```bash
export FP_WORKSPACE=/path/on/ext4/flagprof-session
mkdir -p "$FP_WORKSPACE/tmp" "$FP_WORKSPACE/scratch"
export TMPDIR="$FP_WORKSPACE/tmp"
export SQLITE_TMPDIR="$TMPDIR"
export FLAGPROF_SCRATCH="$FP_WORKSPACE/scratch"
findmnt -T "$FP_WORKSPACE"
```

Store、capture、work、scratch 使用各自目录；输出 run 和报告使用新路径。NVIDIA 支持显式 `--capture-only --capture-filesystem overlay` 的采集路径，之后在 ext4 导入。其他文件系统支持由代码中的准入检查决定。

## 采集自己的推理程序

在已准备的设备环境内，直接启动 Python 目标。`record` 自动注入的范围是它直接启动的 Python 进程：

```bash
python3 -B -m flagprof record \
  --vendor ascend --preset perf \
  --store "$FP_WORKSPACE/capture" \
  --work "$FP_WORKSPACE/import-work" \
  --scratch "$FP_WORKSPACE/scratch" \
  -- python3 your_inference.py
```

NVIDIA 环境使用 `--vendor nvidia`。默认 `final` 采集完整会话，目标正常退出后独立导入；窗口模式需显式选择并推进 step。包装 shell、服务进程或自行再启动的子进程，需要在真正执行推理的进程使用 `flagprof.attach()`；接口见 [总体架构](02-architecture.md)。

`--preset perf` 采集设备与主机性能；`modules` 面向模块；`perf-lite` 使用轻量主机探针；`coverage` 面向路由覆盖；`accuracy` 用于张量数值诊断。`--profile-memory` 增加官方显存记录；`--dispatch each-window` 为逐调用关联提供主机记录。探针对运行的扰动应通过匹配的无采集运行测量。

## 使用内置运行器

模型路径、输入与设备均显式提供；模型权重不随代码发布。embedding 输入 JSONL 每行是字符串或含 `text`、`query`、`input` 的对象：

```bash
python3 -B -m flagprof record \
  --vendor ascend --preset perf --store "$FP_WORKSPACE/embedding" \
  --work "$FP_WORKSPACE/embedding-work" \
  -- python3 -m flagprof.runner hf:/absolute/path/to/model \
     --inputs /absolute/path/to/inputs.jsonl --batch-size 1 \
     --device npu:0 --dtype bfloat16 --outputs "$FP_WORKSPACE/runner-output"
```

generation 使用 `--task generate` 和 `--generation-mode accuracy|performance`。双进程入口为 `flagprof launch --nproc-per-node 2 --comm-backend hccl|flagcx|nccl`，具体目标参数见 `launch --help` 与 `python -m flagprof.runner --help`。只有支持该运行器并匹配设备环境的模型可直接运行；其他目标保留自身加载逻辑，通过进程采集或 Python API 接入。

## 读取与报告

将采集或导入返回的 run 路径写入 `FP_RUN`：

```bash
export FP_RUN=/absolute/path/to/store/run
python3 -B -S -m flagprof inspect "$FP_RUN" --verify --json \
  --work "$FP_WORKSPACE/query-work"
python3 -B -S -m flagprof page "$FP_RUN" --offset 0 --limit 10 \
  --work "$FP_WORKSPACE/query-work"
python3 -B -S -m flagprof report "$FP_RUN" \
  --out "$FP_WORKSPACE/report" --work "$FP_WORKSPACE/report-work"
```

报告生成于 `--out`，HTML 和 JSONL 可离线阅读。双 rank 报告将 `FP_RUN` 换成 launch 产生的 `group.json`。观察数据完整性、身份和覆盖率后，再解读耗时；设备时间、主机时间和请求窗口是不同口径。缺失关联或时钟证据会保留原因。

从已封存 capture 重新生成 Store：

```bash
python3 -B -S -m flagprof import /absolute/path/to/sealed-capture \
  --store "$FP_WORKSPACE/reimport-runs" \
  --work "$FP_WORKSPACE/reimport-work" --scratch "$FP_WORKSPACE/scratch"
```

`seal` 处理正常关闭的外部官方 trace；`analyse` 在带对应 SDK 的环境中分析已封存 raw capture。升级后的派生证据通过新导入生成，历史 Store 和报告保留原样。

## Oracle 与算子重放

`accuracy` 采集需注册模型、样本和输入身份；内置 runner 完成这些接入。任意脚本只启用 preset 不等于已建立跨 run 可比性。

```bash
python3 -B -S -m flagprof report oracle \
  --ref /path/to/reference-run --cmp /path/to/compared-run \
  --out "$FP_WORKSPACE/oracle"
python3 -B -S -m flagprof replay prepare \
  --ref /path/to/reference-run --cmp /path/to/compared-run \
  --call-key 'selected-call-key' --out "$FP_WORKSPACE/replay-input"
```

算子重放需要两侧完整张量 blob，生成包后在相应设备环境执行 `replay run --side ref|cmp --device ... --store ...`。输出误差、传播误差和 replay 新引入误差按各自证据读取。详细契约见 [分析器与报告](07-analyzers-and-reports.md)。
