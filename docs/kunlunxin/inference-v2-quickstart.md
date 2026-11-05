# P800 Inference v2：从空工作区到报告

适用于 [适配指南](inference-v2.md) 锁定的 Linux/FlagRT 环境。先阅读 [验证与开放缺陷](inference-v2-validation.md)：当前候选适配的 OneRec 单卡/TP2 数值分歧尚未解释，不能将运行成功当作精度或性能资格通过。

## 宿主准备

准备候选源码、本地模型、空的 ext4 工作目录和实际查询所得的身份 JSON。镜像的拉取权限、宿主驱动和设备节点由使用者准备；本仓库不分发厂商二进制。用 `docker image inspect --format '{{.Id}}' <标签>` 核对指南中的本地 image ID。image ID 不是 registry pull digest，不能直接用作 `docker pull` 地址。

用 `xpu-smi -m`、`xpu-smi -i <物理编号> -q` 核对空闲、健康、UUID、BDF 和实际 minor，保留原始输出。按查询结果选择 `/dev/xpuN`，不能从 SMI 编号猜 minor。身份 JSON 的 `devices` 按容器逻辑卡顺序排列，结构见适配指南。

先通过所在环境的调度器或租约机制保留选定卡。**本模块不会自动获取 Base 的 `DeviceLease`**；工具的进程/profiler 所有权保护不能代替设备调度。结束后核对本轮进程和显存回收，不停止他人的任务。

下面是单卡容器模板。先填写五个绝对路径及实际设备节点，所有宿主源路径须已存在。源码、模型与身份只读挂载：

```bash
FP_CODE=/absolute/path/to/FlagPerf
FP_MODEL=/absolute/path/to/Qwen3-Embedding-0.6B
FP_HOST_WORK=/absolute/path/on/ext4/new-session
FP_HOST_IDENTITY=/absolute/path/to/identity.json
FP_DEVICE_NODE=/dev/xpuN
FP_IMAGE=sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a
findmnt -T "$FP_HOST_WORK"
docker run --rm -it --network=bridge --cap-drop=ALL \
  --device "$FP_DEVICE_NODE" --device /dev/xpuctrl \
  --shm-size=8g --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$FP_CODE,dst=/code,readonly" \
  --mount "type=bind,src=$FP_MODEL,dst=/models/Qwen3-Embedding-0.6B,readonly" \
  --mount "type=bind,src=$FP_HOST_WORK,dst=/fp" \
  --mount "type=bind,src=$FP_HOST_IDENTITY,dst=/identity.json,readonly" \
  -e FLAGPROF_P800_IDENTITY=/identity.json \
  -e XPU_ENABLE_PROFILER_TRACING=1 -e XPU_CUPTI_ENABLE_DEVICE=0 \
  -e TOKENIZERS_PARALLELISM=false -e RAYON_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e PYTHONPATH=/code/inference -w /code/inference \
  --entrypoint /bin/bash "$FP_IMAGE"
```

两个 XPU tracing 变量须在 import torch **之前**设置；`XPU_CUPTI_ENABLE_DEVICE` 是容器逻辑卡号。确认容器可见卡与身份 JSON 顺序相符。容器模板不构成设备独占资格。

## 容器内目录与输入

```bash
export PY=/root/miniconda/envs/python310_torch29_cuda/bin/python
export FP_WORKSPACE=/fp
mkdir -p /fp/tmp /fp/scratch
export TMPDIR=/fp/tmp
export SQLITE_TMPDIR=/fp/tmp
export FLAGPROF_SCRATCH=/fp/scratch
findmnt -T /fp
"$PY" -B -S -m flagprof --help
printf '%s\n' '"FlagPerf inference sample one."' > /fp/texts.jsonl
```

ext4 是实际文件系统要求，软链接或目录名不构成证明。Store、capture、导入 work、查询 work、scratch、模型输出和报告各用独立目录。预留原始 trace、封存副本、Store、索引和报告的总空间；不只按权重或 trace 大小估算。所有重跑使用新目录。

`--inputs` 每行是字符串或含 text/query/input 的对象，当前只取前 `batch-size` 行组成一个批次，`--forwards` 重复该批次，不遍历全部行；多批次及精度验收改用 [固定输入包](inference-v2-validation.md#固定输入包生成)。

## 无采集基线与采集

先执行无采集基线，再用相同模型、输入和执行协议进行采集：

```bash
"$PY" -B -m flagprof.runner hf:/models/Qwen3-Embedding-0.6B \
  --inputs /fp/texts.jsonl --batch-size 1 --max-length 32 \
  --dtype bfloat16 --attention eager --device cuda:0 --warmup 0 --forwards 2 \
  --outputs /fp/qwen-baseline
"$PY" -B -m flagprof record --vendor kunlunxin --preset perf \
  --timeout 600 --dispatch each-window \
  --store /fp/qwen-capture --work /fp/qwen-import-work --scratch /fp/scratch \
  -- "$PY" -B -m flagprof.runner hf:/models/Qwen3-Embedding-0.6B \
  --inputs /fp/texts.jsonl --batch-size 1 --max-length 32 \
  --dtype bfloat16 --attention eager --device cuda:0 --warmup 0 --forwards 2 \
  --outputs /fp/qwen-perf-output
```

从 record 最后一行 JSON 的 `run` 取真实路径，写入 `FP_RUN` 后查询；不要假定各 preset 都位于 `runs/` 下。final 与 windowed 的产物布局不同。

```bash
export FP_RUN=/absolute/path/printed/by/record
"$PY" -B -S -m flagprof inspect "$FP_RUN" --verify --json --work /fp/query-work
"$PY" -B -S -m flagprof page "$FP_RUN" --offset 0 --limit 10 --work /fp/query-work
"$PY" -B -S -m flagprof report "$FP_RUN" --out /fp/qwen-report --work /fp/report-work
```

报告包含 `report.html`、`result.json` 和导出表，可离线打开。模型输出在 `runner.json` 与 `.bin`：核对 shape、dtype、finite、SHA 及实际文件字节。基线相同只证明采集未改变这些样本，不等于模型数学精度认证。

## 预设与结果判读

| preset | 默认采集与用途 | 需要知道的边界 |
|---|---|---|
| perf | final 官方完整会话 | 逐调用关联显式加 dispatch；显存加 `--profile-memory` |
| modules | final，模块与 stack 诊断 | 大 trace 先估算源预算、总磁盘空间和导入时间 |
| perf-lite | 窗口生命周期，轻量主机探针 | 未采设备/显存/逐算子时对应面板 unavailable 是预期 |
| coverage | 窗口生命周期，dispatch/route | 开销大，诊断耗时不作纯性能结论 |
| accuracy | 窗口生命周期，全层数值探针 | dispatch 必须 off；完整误差比较显式加 `--tensor-blob all`，默认不存 blob |

accuracy Store 单独运行 `report RUN` 可能不含性能面板；这不是精度差分结论。数值比较使用 `report oracle --ref REF_RUN --cmp CMP_RUN --out NEW_DIR`，先核对身份和 alignment，再读取实际误差与首次差异。`ok` 表示分析可用，不表示数值相同。

显式 `perf --capture-mode windowed --active 1 --windows 2` 需要目标调用 `handle.step()`。内置 Qwen 在前向间推进，OneRec 在完整生成请求后推进；不能将其 32 个 decode 步当作 32 次 profiler window step。scheduled window 不能自动代表完整请求。

modules 默认源预算 1 GiB，按实际大小设置 `--source-max-bytes`（例如 8 GiB 为 8589934592），该值不是总磁盘预算或任意规模保证。capture-only/analyse/import 分离采集与离线处理，适用选项见 CLI 帮助；P800 不沿用 NVIDIA 的 overlay capture 例外。

| 层次 | 必须读取 | 不代表什么 |
|---|---|---|
| 目标执行 | final record-status 的 exit_code/status；windowed 的 target_exit/importer_exit/finalize；TP2 group | 退出 0 不证明有设备轨或数值相同 |
| 导入与完整性 | inspect 的 trace、SHA、来源账本、分片/writer 状态 | registered shards verified 不证明 exporter 没丢事件 |
| 分析与数值 | 各 panel 的 status/notes、coverage；oracle 实际差异 | panel ok 不等于误差为 0 |
| 性能资格 | disposition、时钟、完整性、可比性、重复与干扰证据 | 单次耗时不等于性能达标 |

## 排障与归档

| 现象 | 检查与处理 |
|---|---|
| 前置变量拒绝、没有设备轨 | 两个 XPU 变量须先于 torch；逻辑编号/UUID 与实际调用一致 |
| ext4/目录准入失败 | findmnt 核对真实文件系统，work/scratch/输入 Store 分开 |
| inputs/model SHA mismatch | 核对输入包与权重版本；为新资产生成新包，不修改历史 manifest 伪装匹配 |
| modules 预算/导入超时 | 保留失败及 sealed capture，核算预算与空间，另目录恢复导入 |
| TP 初始化失败 | 核对网络接口、两卡可见性、accelerate overlay、shm/memlock |
| report 成功但 panel partial/unavailable | 读 notes/coverage，区分未采与缺证据，不改写成通过 |
| 弃用、profiler 未预热警告 | 保留原日志；未预热可能影响计时解释，不据此授予性能资格 |

离线 import/inspect/page/report 可用 `-S` 且无 torch；analyse 处理厂商 raw 时仍可能需要对应 SDK。delivery publish/verify/restore 是本地证据归档命令，不会自动上传；restore 目标须为 ext4 新目录。归档源码 commit、环境、容器参数、模型/输入 SHA、原始日志和全部失败/补测目录；分享前检查路径、设备标识和业务输入是否适合公开。
