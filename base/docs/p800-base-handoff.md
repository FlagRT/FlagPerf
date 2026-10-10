# 昆仑芯 P800 Base 使用与维护手册

更新：2026-10-08。适用源码基线：`zhiyu/kunlunxin-p800@be0f2ea575849a5fdadd8e3e5ab1db317b304bff`，后续本轮提交仅同步文档和审计附件。本文覆盖 Base Benchmark；Toolkit 有独立契约与报告。

## 1. 交付范围与入口

Base 功能收尾成立：15 个 case 已全部分类，10 个可执行、5 个启动前跳过；114 个最终 run 的测量及正确性通过。正式资格与项目交付分别记录：严格资格脚本通过 15 个重复组；用户接受 D2H pageable/nonblocking 三组的不足 15 秒窗口，其中两组稳定、一组仍为 unstable。八卡测量与重复性通过，但含卡 2 历史 ECC，不能写成健康资格通过。镜像保持 `candidate / validated:false`。

只使用 `python3 base/run.py benchmark run`、`benchmark preflight` 与 `report`。`acceptance-report` 已在 `be0f2ea5` 移除；历史聚合报告保留作归档，不能作为现行命令或替代严格资格脚本。厂商选择器为 **`kunlunxin`**，芯片为 **`P800`**；没有 `kunlunxinb` 选择器。

| Case | 实现及卡数 | 已验证范围 / 限制 |
|---|---|---|
| computation-FP32 | XPYTORCH `torch.mm`，单卡 | 8192³；TF32 关闭不等于内部 IEEE 精度认证 |
| computation-FP16 | XPYTORCH `torch.mm`，单卡 | 8192³；输入输出 FP16，CPU FP64 reference |
| computation-BF16 | XPYTORCH `torch.mm`，单卡 | 8192³；约 FP32 吞吐，保留后端性能限制 |
| computation-INT8 | `xtorch_ops.gemm_I8_I8_bf16_nt`，单卡 | INT8 输入、缩放、BF16 输出；历史 CPU fallback 数字撤回 |
| interconnect-h2d / interconnect-d2h | 预分配 `Tensor.copy_`，单卡 | 512 MiB，各四种 host-memory/请求模式；D2H pageable/nonblocking 有时长例外 |
| main_memory-bandwidth | 预分配 `Tensor.copy_`，单卡 | 4 GiB；读+写按 2×payload；不是物理 HBM 峰值 |
| main_memory-capacity | 实际持有分配至有界 OOM，单卡 | 98,132 MiB；高风险显式开启，单次验证及释放检查 |
| interconnect-MPI_intraserver | FlagCX/BKCL AllReduce，代码接受 2–8 rank | 2/6 卡严格五次资格通过；7 卡仅三次；8 卡五次稳定但健康 degraded |
| interconnect-P2P_intraserver | FlagCX/BKCL send/recv，恰好 2 rank | rank 0→1 单向，五次资格通过 |
| computation-FP64 | 启动前 skipped | 锁定栈 probe 输入 FP64 却输出 FP32 |
| computation-FP8 | 启动前 skipped | 锁定栈 mm/scaled-mm API 未实现 |
| computation-TF32 | 启动前 skipped | 可控 TF32 路线未证明；不推断芯片无能力 |
| interconnect-MPI_interserver / interconnect-P2P_interserver | 启动前 skipped | 当前统一执行器只解析单宿主 `nnodes=1` |

两个传输方向合计 **8 个变体**，不会把 Base case 数量变成 Toolkit 的 12 case。MPI 不支持 1 rank。

## 2. 登录、环境与身份

```bash
ssh klx
cd /home/kzhang519/Zhiyu/runtime-team/FlagPerf
git status --short --branch
git rev-parse HEAD
```

SSH 使用已配置别名，提权在同一终端交互执行 `sudo -v`。密码不得写入命令文件、证据或 Git。实际测试前重新确认选卡、时间窗口、占用和镜像；历史卡号不代表当前可用。

| 项 | 锁定值 / 解释 |
|---|---|
| 宿主配置 | `base/configs/kunlunxin_p800_xpytorch29.yaml`，扩展名虽为 YAML，内容是 JSON |
| runtime profile | `base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/` |
| 镜像 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` |
| image ID | `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975` |
| Python / PyTorch | 锁定镜像静态审计为 3.10.18 / 2.9.0+cu129；不能从镜像名推测实测版本 |
| 执行环境 | `/root/miniconda/envs/python310_torch29_cuda`，XPYTORCH / torch_xmlir，原生路径 |
| Driver / Runtime | Driver 5.0.21.47；工具 XPU-RT 10.2、头文件 5.0.21、镜像 XRE 5.13 分属不同来源 |
| 发布状态 | manifest 的 `release_stage=candidate`、`validated=false`、空 validation scope；本轮未晋级 |

设备身份按 **SMI physical ID → query minor/UUID → 容器节点 → framework UUID join → logical ID** 解析。物理卡号、节点后缀、rank 不可互换；例如历史卡 5 对应 `/dev/xpu7`，这是该次绑定，不能硬编码到新环境。

冻结标签 `p800-base-frozen-20261008` 解引用为 `7371195ea11426ee33511fac1363f04d74eabe09`，该历史提交包含后来移除的聚合层。远端分支在本轮开始时已是 `be0f2ea5`；不要把标签等同于现行交付源码，也不要重写标签。本轮文档提交仅保存在 klx 和本机各自仓库，未推送。

## 3. 执行架构和配置

`run.py` → 静态 vendor registry/provider → case resolver → bounded executor → runtime bootstrap/UUID 校验 → driver/workload → metric/correctness/monitor → postflight/cleanup/lease release → per-run report。

- `base/vendors/protocol.py`、`registry.py` 定义厂商接口和显式注册；provider 决定运行时、设备、容器和门禁。
- `base/benchmarks/case_assets.py` 按 `generic < vendor < chip < override` 合并配置；host/worker 共用资产哈希。
- `base/executors/bounded_benchmark.py` 编排 P800 执行；`base/benchmarks/drivers/kunlunxin.py`、`day6.py`、`transfer.py` 承担设备和指标语义。
- `base/monitoring/kunlunxin_usage.py` 留存 xpu-smi 原文和时窗；缺失值不填零。
- `base/generate_benchmark_report.py` 是与 Ascend 共用的 Base 每 run 报告生成器（schema 3）；`base/qualification.py` 验证五次组，不是已移除的聚合报告层。

单卡计算/传输/内存为 1 rank；MPI 2–8 rank，P2P 恰好 2 rank。非 privileged 容器只映射选中设备及 `/dev/xpuctrl`。通信需要已有的 internal bridge：

```bash
sudo docker network inspect flagperf-p800-internal
```

若确实不存在，在完成本机地址段冲突检查后创建：`sudo docker network create --internal --subnet 172.31.254.0/24 flagperf-p800-internal`。这是共享前置设施，不能在单 run cleanup 时删除。

| 参数 / 环境变量 | 正确用法 |
|---|---|
| `--physical-device-ids` | P800 显式物理选卡；不与 `--device-ids` / `--npu-ids` 混用 |
| `--allow-candidate-runtime` | Base 候选运行时真实执行需显式允许；不是 PR0 的 `--allow-candidate` |
| `--case-config` | 工作量覆盖文件；不能绕过 runtime requirements |
| `--reservation-end` / `--reservation-reference` | 带时区的未来截止时间 / 本轮授权依据 |
| `--privilege-command 'sudo -n'` | 宿主 Docker/句柄查询权限；不是 privileged 容器开关 |
| `--result-dir` | 每次用全新目录，禁止覆盖历史 run |
| `--allow-high-risk-case` | 仅容量等高风险 case 的显式允许 |
| `USE_FLAGGEMS=0` | profile 锁定原生 XPYTORCH 路径 |
| `CUDA_VISIBLE_DEVICES` | bootstrap 按已验证映射设置；不要直接填 SMI 卡号 |
| `XPU_VISIBLE_DEVICES` / `XPU_EVENT_KL3_ENABLE` | 锁定 bootstrap 真正 unset；禁止照抄其他路径的 KL3 设置 |
| `P800_KNOWN_READONLY_MONITORS=barad_agent` | 仅本轮确认并允许该常驻监控时设置；代码仍校验已知可执行文件与 argv |
| `P800_ALLOW_FOREIGN_HANDLES=1` | 共享设备 exploratory 的独立授权；不能据此取得空闲卡资格 |

## 4. 单卡最小运行

以下 `5` 是静态示例。真实执行前设置当前已授权的 `P800_CARD`、带时区 `P800_END`、`P800_AUTH`、全新 `P800_RUN`；这些 shell 变量仅保存非敏感运行参数。

```bash
python3 -B base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP32:P800 --physical-device-ids 5 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

独立身份检查可用 `benchmark preflight`，仅接收一张物理卡，timeout 为 30–180 秒。性能入口 timeout 为 30–600 秒；资源窗口需额外覆盖清理预算。

```bash
sudo -v
python3 -B base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP32:P800 \
  --case-config base/benchmarks/computation-FP32/kunlunxin/P800/case_config.smoke.yaml \
  --physical-device-ids "${P800_CARD:?set authorized physical ID}" --nproc-per-node 1 \
  --allow-candidate-runtime --privilege-command 'sudo -n' \
  --reservation-end "${P800_END:?set future ISO-8601 time with timezone}" \
  --reservation-reference "${P800_AUTH:?set authorization reference}" \
  --result-dir "base/result/${P800_RUN:?set unique run name}" --monitor off --timeout 300
```

资格运行去掉 smoke override、改 `--monitor on`，分别启动五次独立进程，固定配置、镜像、源码和 UUID 集合。默认计算配置：FP32 2000 iterations / warmup 100；FP16、BF16 5000 / 100；INT8 8000 / 10，均为 8192³。不要把早期 4096³ / 20,000 iterations 方案当作当前默认。

## 5. 传输、内存与通信

H2D/D2H 覆盖文件为：
`base/benchmarks/interconnect-{h2d,d2h}/kunlunxin/P800/case_config.measured-{pageable,pinned}-{blocking,nonblocking}.yaml`。
花括号是文件族说明，CLI 每次只能指定其中一个真实路径。八种变体独立统计；`non_blocking=true` 只表请求，不证明与计算重叠。

Base 传输计时循环直接调用整块 `Tensor.copy_`；**Toolkit 的 XRE 1 MiB 分块 workaround 不是 Base 实现**。Base 锁定栈保留历史 4 GiB pinned async error 999；512 MiB 的通过不能外推所有尺寸。

D2D 使用预分配 source/destination 的 `copy_`，不是在循环里 `clone()` 分配。容量使用 `main_memory-capacity:P800` 并显式加 `--allow-high-risk-case`，只统计实际持有字节数；容量短窗口的监控 partial 原样保留，释放、cleanup、postflight 必须通过。

通信默认 YAML 是 smoke。正式组需独立 override，例如复现既有 256 MiB 的两卡 MPI 规模：

```bash
cat > /tmp/p800-mpi-qualified.yaml <<'YAML'
Melements: 64
WARMUP: 10
ITERS: 3500
MODE: qualification
YAML
python3 -B base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case interconnect-MPI_intraserver:P800 \
  --case-config /tmp/p800-mpi-qualified.yaml \
  --physical-device-ids 0,5 --nproc-per-node 2 \
  --allow-candidate-runtime --timeout 300 --monitor on --dry-run
```

真实执行移除 `--dry-run` 并加入第 4 节的授权窗口、权限和新目录参数。卡对必须重新选择；P2P 换成对应 case、保持两个 rank。八卡历史配置为 4200 iterations，3500 次不足窗口的失败记录保留；仅说明已有实测，不自动启动八卡复测。

公式：GEMM=`2*M*N*K*ITERS/seconds`；H2D/D2H/P2P=`payload*ITERS/seconds`；D2D=`2*payload*ITERS/seconds`；AllReduce `algbw=message*ITERS/seconds`、`busbw=algbw*2*(n-1)/n`。GB/s 用 1e9，GiB/s 用 2³⁰；多 rank 每 run 取最小有效值，不能相加。

## 6. 读取结果和重建报告

先看 `summary.json` 的 measurement、correctness、monitoring、postflight、cleanup 和 lease，再核对 `benchmark-result.json` 与 `artifacts/metric-rank-N.json`、`correctness-rank-N.json`。`control/host-context.json` 和 `case-assets.json` 固定设备与配置；`code-identity.json` 固定源码身份。Base 不把 Toolkit 的 diagnosis 列等同于资格判定。

退出码：`0=passed/skipped`、`1=failed`、`2=partial`（CLI 参数/授权拒绝也可能为 2，须读错误信息）。单 run `passed` 不代表五次资格或健康通过。容量和短 smoke 的 partial 有明确窗口原因。

```bash
python3 -B base/run.py report --result-root base/result --run-id EXISTING_RUN
python3 -B base/qualification.py RUN1 RUN2 RUN3 RUN4 RUN5 --output NEW_STATS.json
```

对于 `.../base-runs/FP32-q1`，`--result-root` 必须指向该 `base-runs` 父目录，`--run-id FP32-q1`。报告命令不重跑设备，但会更新派生报告和 `summary.report_generation` 元数据；证据审计应在副本重建，不在已冻结原目录写入。

严格资格要求五个独立 run、相同源码/配置/UUID/镜像、各 rank ≥15 秒、有效监控、正确性和清理、健康空闲状态及样本 CV≤5%（ddof=1）。`MODE=measured` 的单 run 成功不强制 15 秒，但 `qualification.py` 强制。因此用户对三组 D2H 窗口的接受记录在交付层，不能改写成脚本返回 passed。

`--allow-shared` 只能把满足其他前置条件的共享/健康受限结果输出为 exploratory，不能绕过计时/重复数检查，不能把 degraded 变成正式资格。容量单次、七卡三次不输入五次资格脚本。

## 7. 回归、故障处理与交接

```bash
base/result/p800-pr1-20260920-162724/cpu-test-venv/bin/python -B -m unittest discover -s base/tests -p 'test_*.py'
python3 -S -B -m unittest discover -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py'
python3 -B -m unittest discover -s base/toolkits/_common/ascend/A3/tests
```

klx 的既有 CPU venv 含 PyYAML、CPU PyTorch；系统 Python 缺 torch 时不能用其全量失败推断代码回归。2026-10-08 本轮结果为 252 / 32 / 80 项通过。换机器需准备相同测试依赖；这是 CPU 回归，不是 Ascend 实机验收。

| 现象 | 排查与处置 |
|---|---|
| static plan skipped | 读 applicability.reason；五个已知 skip 属当前能力范围 |
| host-preflight 句柄拒绝 | 查看目标卡占用和 holder 分类；已知只读 agent 需显式授权，不停止他人任务 |
| runtime / UUID 不匹配 | 核对镜像锁、minor 和 UUID join；不要改锁绕过检查 |
| BKCL socket 无可用网卡 | 检查 internal bridge；不默认改成 host network |
| 同步 timeout | 读取本 run 原始错误和内核证据，确认本轮容器清理；禁止 reset 掩盖故障 |
| run passed 但组资格拒绝 | 检查时长、五次身份一致性、CV、监控、健康；项目例外须单独写明 |
| cleanup 不确定 | 依据精确 CID 复核本轮容器与 lease，未确认前不扩大测试 |

当前无物理卡的操作员排除政策；全部选卡均执行身份、占用、健康和租约门禁。卡 2 历史不可纠正 ECC=8 继续 degraded。XPULink 反差、BF16 吞吐限制和 INT8 算子语义保留。更换镜像/驱动、dtype、shape、卡数或拓扑后重新验证对应范围。


验收与审计见 [交付复核](p800-base-acceptance-20261008.md) 和 [可提交审计附件](../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/base-doc-audit-20261008/README.md)。
