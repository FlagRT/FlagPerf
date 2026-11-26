# P800 Inference v2：从空工作区到报告

以下命令采用 flagprof 0.3.0 和支持 P800 官方 profiler 的 Linux/XPyTorch 环境。历史实测栈见[适配指南](inference-v2.md)。先阅读 [验证与开放缺陷](inference-v2-validation.md)：当前候选适配的 OneRec 单卡/TP2 数值分歧尚未解释，不能将运行成功当作精度或性能资格通过。

## 宿主准备

准备 FlagPerf 0.3.0 源码或 wheel、本地模型和可写的输出目录。选用支持 XPyTorch profiler 的运行环境。复现历史测量时，可在宿主用 `docker image inspect` 对照[实测镜像](inference-v2.md)。

用 `xpu-smi -m`、`xpu-smi -i <物理编号> -q` 核对占用、健康、UUID、BDF 和实际 minor，按查询结果映射设备节点。适配器默认在容器内查询 SMI。容器不能查询时，可把宿主生成的身份 JSON 只读挂载并设置 `FLAGPROF_P800_IDENTITY`，格式见[适配指南](inference-v2.md#设备身份与隔离)。

先通过所在环境的调度器或租约机制保留选定卡。**本模块不会自动获取 Base 的 `DeviceLease`**；工具的进程/profiler 所有权保护不能代替设备调度。结束后核对本轮进程和显存回收，不停止他人的任务。

下面以源码只读挂载为例。把示例路径、设备节点和镜像替换为本机可用的值。

```bash
FP_CODE=/absolute/path/to/FlagPerf
FP_MODEL=/absolute/path/to/Qwen3-Embedding-0.6B
FP_HOST_WORK=/absolute/path/to/new-session
FP_DEVICE_NODE=/dev/xpuN
FP_IMAGE=your-xpytorch-image:tag
mkdir -p "$FP_HOST_WORK"
docker run --rm -it --network=bridge --cap-drop=ALL \
  --device "$FP_DEVICE_NODE" --device /dev/xpuctrl \
  --shm-size=8g --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$FP_CODE,dst=/code,readonly" \
  --mount "type=bind,src=$FP_MODEL,dst=/models/Qwen3-Embedding-0.6B,readonly" \
  --mount "type=bind,src=$FP_HOST_WORK,dst=/fp" \
  -e TOKENIZERS_PARALLELISM=false -e RAYON_NUM_THREADS=4 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e PYTHONPATH=/code/inference -w /code/inference \
  --entrypoint /bin/bash "$FP_IMAGE"
```

需要宿主身份清单时，在 `docker run` 中增加 `--mount "type=bind,src=$FP_HOST_IDENTITY,dst=/identity.json,readonly" -e FLAGPROF_P800_IDENTITY=/identity.json`，其中 `FP_HOST_IDENTITY` 为宿主文件的绝对路径。`record` 和 `launch` 在启动官方采集进程前准备 tracing 环境。

## 容器内目录与输入

激活含 XPyTorch 的环境后执行以下命令。`PY` 取当前环境的解释器。如已在该环境安装 wheel，可直接使用安装好的模块。

```bash
export PY="$(command -v python)"
export FP_WORKSPACE=/fp
"$PY" -B -m flagprof --help
printf '%s\n' '"FlagPerf inference sample one."' > /fp/texts.jsonl
```

`--store` 所选盘需要容纳原始 trace、封存副本、Store、检查点和索引。查看使用系统临时目录，报告工作文件跟随 `--out`。同一封存采集重试导入时，沿用原 `--store` 即可复用校验通过的检查点。

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
  --store /fp/qwen-capture \
  -- "$PY" -B -m flagprof.runner hf:/models/Qwen3-Embedding-0.6B \
  --inputs /fp/texts.jsonl --batch-size 1 --max-length 32 \
  --dtype bfloat16 --attention eager --device cuda:0 --warmup 0 --forwards 2 \
  --outputs /fp/qwen-perf-output
```

从 record 最后一行 JSON 的 `run` 取真实路径，写入 `FP_RUN` 后查询；不要假定各 preset 都位于 `runs/` 下。final 与 windowed 的产物布局不同。

```bash
export FP_RUN=/absolute/path/printed/by/record
"$PY" -B -m flagprof inspect "$FP_RUN" --verify --json
"$PY" -B -m flagprof page "$FP_RUN" --offset 0 --limit 10
"$PY" -B -m flagprof report "$FP_RUN" --out /fp/qwen-report
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

`record --preset modules` 默认源预算为 4 GiB，按实际大小设置 `--source-max-bytes`（例如 8 GiB 为 8589934592），该值不是总磁盘预算或任意规模保证。capture-only/analyse/import 分离采集与离线处理，选项见 CLI 帮助。

| 层次 | 必须读取 | 不代表什么 |
|---|---|---|
| 目标执行 | final record-status 的 exit_code/status；windowed 的 target_exit/importer_exit/finalize；TP2 group | 退出 0 不证明有设备轨或数值相同 |
| 导入与完整性 | inspect 的 trace、SHA、来源账本、分片/writer 状态 | registered shards verified 不证明 exporter 没丢事件 |
| 分析与数值 | 各 panel 的 status/notes、coverage；oracle 实际差异 | panel ok 不等于误差为 0 |
| 性能资格 | disposition、时钟、完整性、可比性、重复与干扰证据 | 单次耗时不等于性能达标 |

## 排障与归档

| 现象 | 检查与处理 |
|---|---|
| 前置变量拒绝、没有设备轨 | 核对逻辑设备与 UUID；直接调用 attach 时须在加载 torch 前设置两个 tracing 变量 |
| 输出写入失败 | 核对所选结果盘的权限、可用空间和底层 I/O 错误 |
| inputs/model SHA mismatch | 核对输入包与权重版本；为新资产生成新包，不修改历史 manifest 伪装匹配 |
| modules 预算/导入超时 | 保留失败及 sealed capture，核算预算与空间，对同一封存采集和目标 Store 重试导入 |
| TP 初始化失败 | 核对网络接口、两卡可见性、accelerate overlay、shm/memlock |
| report 成功但 panel partial/unavailable | 读 notes/coverage，区分未采与缺证据，不改写成通过 |
| 弃用、profiler 未预热警告 | 保留原日志；未预热可能影响计时解释，不据此授予性能资格 |

离线 import/inspect/page/report 只需已安装 FlagPerf 的标准库环境；analyse 处理厂商 raw 时仍可能需要对应 SDK。delivery publish/verify/restore 是本地证据归档命令，不会自动上传；restore 目标须为所选可写盘上尚不存在的目录。归档源码 commit、环境、容器参数、模型/输入 SHA、原始日志和全部失败/补测目录；分享前检查路径、设备标识和业务输入是否适合公开。
