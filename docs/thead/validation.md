# 平头哥 PPU 验证与维护

本文区分可执行范围、历史实测和当前代码的离线检查。当前锁定 runtime 保持
`release_stage:candidate`、`validated:false`：全部既有实测在显式
`--allow-candidate-runtime` 授权下完成。与昆仑芯 P800 的"已验证但范围受限"约定不同，
PPU 的 validated 复评条件是 16 rank 同机通信资格在独占窗口补齐后进行；在那之前，
候选状态如实保留，不因文档或例外自动晋级。

Operation 锁定镜像为 `flagperf-ppu-operation:654e2a3a`
（image ID `sha256:b3246f3fa6171007f66655aa5d5d1cfd679c1d4d94d34ce7e1ea08f087467b75`），
其 Base image 是 `flagtree-ppu-3.6-dev:202608`
（image ID `sha256:be1606e81fb24ca3da57a6d4a990280369b89146c2999e88eb8cd07c661dcc79`）。
每轮有界 run 都在容器预检查中核对 image ID，并在容器内回读设备 UUID 完成
三层身份验证；历史 run 的 `case_assets_sha256` 与 `context_sha256` 原样保留，
不因后续配置修订重签。原始 evidence 与资格审计（`*-qualification.json`）保存于
各轮结果目录，本文不复制、不改写其数字。

## 已实测范围（2026-10-14 至 2026-10-15，全部五次独立 run 资格审计 passed）

| 范围 | 组 | 结果（五 run 中位） | 审计 CV |
|---|---|---|---:|
| 计算 FP32 / FP16 / BF16 / TF32 / FP64 | A | 34.535 / 135.612 / 135.018 / 67.675 / 5.432 TFLOPS | 0.004–0.4% |
| 计算 INT8（`torch._int_mm` K-major 路径） | A | 269.249 TOPS | 0.041% |
| H2D/D2H × pinned/pageable × 基线/跨 NUMA 8 变体 | B | pinned 29.0（H2D）/ 28.5（D2H）GB/s；pageable H2D 11.7–12.1 / D2H 11.5–11.9 GB/s（基线–跨 NUMA 区间） | 0.007–3.48% |
| 设备显存带宽（4 GiB 拷贝） | C | 2052.36 GB/s | 0.004% |
| 同机 AllReduce（256 MiB，8 rank） | C | 25.907 GB/s algbw（busbw 45.3 GB/s） | 0.048% |
| 同机 P2P 单向（256 MiB，2 rank，staged） | C | 4.832 GB/s | 0.31% |
| 容量（高风险单次验证，不入五次资格） | — | held 98,263 MiB（103.04 GB，21 次分配，free-memory 边界） | — |

INT8 的 K-major 前置条件（`b.t().contiguous().t()`，双操作数 K 维连续）是该栈当前
唯一有效路径：naive 布局实测 0.176 TOPS（回退证据保留），转置视图直达调优内核。
FP16 真实调用形态（`F.linear`、`mm(x, W.t())`）与裸 mm 同档满速，无布局陷阱。

### P2P 扫点矩阵（p2p-single-node-v1，2026-10-19）

协议、入口与发布记录见 [P2P 协议文档](p2p.md)；本节只保留结果表。

单卡双芯拓扑下，同卡两芯与跨卡两芯的通信路径带宽不同。对标 ascend 同名扫点协议，
以两对代表设备 × 4 个消息规格建 8 cell 矩阵（每 cell 3 次重复，方向钉死 rank0→rank1）：
ITERS 由 64 GiB 单点校准钉扎（目标 ~60 s 窗口；共享契约 ITERS ≤ 200000，故同卡 4 MiB
cell 约 25 s）。size 曲线在 monitor on 下采集（与 ascend monitor-off 曲线的偏离及
原因记录在计划文件）；monitor A/B（off,on,on,off,off,on，钉在同卡 64 MiB cell）
中位数偏差 0.07 %（≤ 5 %）。cell 中位 CV：0.18%, 0.15%, 0.06%, 0.05%, 0.40%, 0.45%, 0.39%, 0.33%。

| cell（单向，staged 口径） | 4 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---:|---:|---:|---:|
| 同卡两芯（14→15） | 33.23 | 42.18 | 44.64 | 42.37 |
| 跨卡 staged（7→8） | 4.78 | 4.8 | 4.81 | 4.8 |

同卡两芯约为跨卡 staged 的 9 倍，两条路径均为主机转运口径（`P2P=disabled`）；
数字只覆盖钉死方向与实测设备对，不外推到反向、其他设备对或直连互联。
cell 明细、逐 run 值与验收判定见
[资格计划](../../base/benchmarks/interconnect-P2P_intraserver/thead/PPU/p2p-qualification-plan.json)。

## 无设备检查

使用 Python 3.11 或更新版本，先安装[宿主依赖](../../base/requirements-host.txt)。
下列检查不需要 Docker、设备或历史结果目录：

```bash
python3 -m pip install -r base/requirements-host.txt
python3 -B -m unittest discover -s tests -p test_public_distribution.py
python3 -B -m unittest discover -s base/tests -p 'test_*.py'
```

`base/tests` 全量套件在锁定镜像容器内验证过（Python 3.12，1 项可选历史 smoke
重放测试在无归档时明确 skipped，应单独记录，不计入通过数）。公开分发检查覆盖
三家厂商的文档链接与入口约定，含 `docs/thead`。Toolkit 合同与失败路径由
`base/tests/test_thead_toolkit.py` 无设备守护，与 P800 的 `test_p800_toolkit.py` 同构。

缺少依赖、跳过测试或仅通过 dry-run 时，不得记录为完整回归通过。
正式测量还需 Git 工作副本记录源码身份。

## Toolkit（thead-ppu-toolkit）

2026-10-19/20 落地 12 Case 官方工具 Toolkit：sailbandwidth/p2pBandwidthLatencyTest 原值直采、
acblas 隔离子进程（正确性回读）、ppu-smi 静态容量；宿主执行器、容器内 runner 与确定性
报告和昆仑芯 P800 同构。计时点支持 `--minimum-seconds` 窗口下限（与 P800 同语义）：
acblas 子进程按样本数与下限双条件计时，官方闭源工具在同一测量点内重复调用并取
各次原值中位数；参考监控范围 15 秒。工具链无 DMI 等价阈值诊断，诊断层为合规校验
语义；computation-INT8 因 `acblasGemmEx` int8 输出错误与 `acblasLtMatmul` 厂商 JIT 崩溃
被排除（probes/day9/toolkit T8 证据）；缺陷详情、复现参数与复测协议见 [acblas 已知问题](acblas-issues.md)；computation-BF16 保持注册 case，厂商内核缺陷
见下。参数、协议与运行方式见
[Toolkit 使用说明](../../base/toolkits/_common/thead/PPU/README.md)。

守护与回归（2026-10-20）：

- `base/tests/test_thead_toolkit.py` 31 项无设备守护测试通过：合同表与点结构锁定、
  测量点路由完备性、官方工具输出 fixture 重放（含方向门与字节 token）、候选镜像
  显式授权门、监控窗口计数、报告确定性。
- 锁定镜像容器内 `base/tests` 全量 379 OK（1 项既有历史 smoke 重放 skipped）；
  公开分发检查 10 OK。

硬件冒烟与缺陷修复（双卡 14/15，`--smoke --allow-candidate-runtime`）：

- 首轮冒烟（10-19）暴露三类缺陷：acblas 列主序操作数序错误（正确性 2.329）、
  `d2d-local`/`access` 测量点无路由、FP16/BF16 正确性回读下载尺寸按 4 字节元素
  覆盖 2 字节宿主缓冲（堆损坏 SIGABRT）。三类均修复并留原始失败记录
  （20261019T130626Z 与 20261020T032250Z 两轮 run 目录原样保留）。
- 修复后冒烟（20261020T033547Z）：12 Case 测量层与诊断层全部通过；监控层按设计
  保持 partial——`--smoke` 固定 `minimum-seconds 0`，窗口低于 10 样本监控下限，
  与 P800 的 smoke 语义一致。
- 下限验证（20261020T035226Z，`--case interconnect-h2d --repeat 1 --minimum-seconds 15`）：
  三层全部 passed，每目标窗口内监控样本 26–29。
- 首个全量 run（20261020T065417Z）暴露第五类缺陷：报告重复组表把全部卡与方向归并
  为一组（metric 记录缺 point 字段，标签退化为 `card ?`）；修复为 metric 携带完整
  point（evidence_runner），重复组按卡/方向独立分组。

全量资格口径 run（repeat 5、`--minimum-seconds 15`、双卡 14/15）：三轮（20261020T065417Z 暴露上述缺陷 5；修复后 20261020T081237Z 与
20261020T090026Z 为 11/12 Case 通过且数值复现；后者 result-root 因根文件系统
对非 root 写满改挂 /tmp tmpfs，测量路径无磁盘 I/O）：FP32 34.5 / FP64 5.53 /
FP16 135.1 / TF32 67.6 TFLOPS（TF32=FP16/2），本地 D2D 1051 GB/s（×2≈HBM
2052 GB/s），H2D CE 29.0 / SM 26.9 与 D2H CE 28.5 / SM 26.4 GB/s，主机-设备
访问时延 1.09 µs，P2P 时延 1.25–1.27 µs，P2P 单向 46.2–46.9 GB/s，显存容量
98304 MiB。未通过项均不归因 toolkit 实现：computation-BF16 两轮 10/10 目标
失败为 `acblasLtMatmul` 厂商内核上下文腐蚀（probes/day10/toolkit 定性：
heuristic 在 n≥1024 选中 algo 0x1700000001，每次调用成功但 teardown 非确定
报 error 700/挂起/正常三种结局，n=256 的 0xC0000001 干净；按约定原样移交
算子团队，不缩小矩阵或放宽容差绕过）；interconnect-P2P_intraserver 两轮各
1/30 双向目标因官方工具 stdout 矩阵块畸形被严格解析拒绝（081237Z 在
15→14/rep-4、090026Z 在 14→15/rep-3，签名相同），按 P800 严格失败语义原样
记失败、不重试。该畸形后定位为 mpirun 双 rank stdout 行内交错撕裂矩阵行（probes/day11/toolkit 取证：
同点 10 次调用中 8 次仅校验行序抖动、1 次行中间交错），修复为每 rank 独立捕获并按秩
拼接（解析与无重试语义不变，mpirun 原流与 rank 文件全量归档引用），case 级验证 run
20261020T121430Z-4f3e0b0e 30/30 目标通过；双向值的间歇波动为独立环境现象，仍按原值
与稳定性判定记录。
BF16 全量失败于 2026-10-20 深夜改判：090026Z 子进程记录（AssertionError、零样本、
hggcMemPoolDestroy code=700）与 day-12 纯 torch 对照（8192×2000 次/进程 17/17 干净、
130.8 TFLOPS）定案为 harness 自身缺陷——BF16 Lt correctness 预检以测量规模 layout
跑 256 规模缓冲，越界毒化上下文，非厂商内核腐蚀；修复 commit 41d329a1（每规模独立
管线），冒烟与 33 项单测绿后全量复测 20261020T140938Z-aea8308d：12/12 全部通过（BF16 n=10 中位数 134.686 TFLOPS，134.578–135.175）。
「BF16 厂商内核腐蚀、交算子团队」的移交撤回（INT8 维持 day-9 排除）；day-10「卡 15
teardown 正常返回」与原始输出不符（两卡均完成工作后在退出阶段 EXIT=124，独立低危
退出路径现象），一并勘误；勘误与 day-12 证据存证据快照 probes/day10–12。
另如实记录：双向并发 P2P 在部分窗口不稳定（081237Z 双向两组
跨 50–90 GB/s；090026Z 的 14→15 组内 88.3→78.7→56.5→55.7 衰减、组 CV
23.3%，恰含上述畸形目标；单向四组始终 stable-observed（CV 0.15–0.62%），
065417Z 双向曾全程稳定 88.2–89.9 GB/s，88–90 GB/s 能力上限已多轮复现），
报告重复组以与 P800 同构判定（ddof=1，<5 重复 insufficient、>5% unstable、
否则 stable-observed）原样标定，不作修正。

当前锁定镜像为候选且未验证：每轮 Toolkit 运行必须显式 `--allow-candidate-runtime`；
该授权不改变 release 声明，也不把 Toolkit 结果当作已验证 runtime 的能力证明，
Toolkit 结果保持 not-qualified。

## 状态与指标判定

| 层次 | 含义 |
|---|---|
| Base 单 run | 分别记录执行、测量、正确性、监控、postflight、cleanup 和报告状态 |
| 五次资格 | 固定 case、配置、源码、镜像和 UUID 集合；各 rank 计时不少于 15 秒；监控、正确性、资源及清理门禁通过 |
| 重复稳定性 | 五个独立 run 的样本标准差（ddof=1）除以均值；CV > 5% 为 unstable |
| 容量 | 高风险搜索型单次验证：只登记实际持有的分配并核实释放；15 秒窗口规则不适用 |

单 run `passed` 不推断五次严格资格通过；`qualification.py` 独立审计。
计时窗口不足 15 秒的 run（如降级前的 8 rank AllReduce sweep）case 级可能 passed
但 verdict 为 partial，只能作为数据点登记，不构成资格形态。

## 已知限制与边界

1. **16 rank 同机 AllReduce 独占资格未执行**：已完成共享功能 smoke，16 个 rank 的
   绑定、正确性和清理通过，但计时窗口不足 15 秒且没有独占条件。窗口迭代数必须在目标
   rank 数的实测窗口内重定——AllReduce 单次迭代耗时对 rank 数**非单调**
   （2r 18.9 ms / 4r 24.8 ms / 8r 10.4 ms @256 MiB），不能外推。
2. **FP8 唯一跳过**：`torch.mm` 与 `_scaled_mm` 路由均失败（算子未实现 /
   compute-capability 门），有探针异常证据；不代表芯片其他栈无此能力。
3. **跨机 2 条不适用**：单机执行器只解析 `nnodes=1`，且配对目标机未上线；
   目标机上线不自动扩大已冻结范围。
4. **P2P 为 staged 路径**：当前服务器无卡间直连，NCCL 经主机内存转运；
   证据按 `P2P=disabled` 如实标注，不冒充直连带宽。扫点矩阵同样只覆盖
   staged 路径与钉死方向。
5. **通信数字的口径**：本轮全部同机通信吞吐基于 PCIe + 主机转运路径实测；
   若目标硬件的直连互联（厂商标称 700 GB/s 量级）在其他机型启用，
   须重新测量，不能沿用本文数字。
6. **D2H pageable 对主机 CPU 竞争敏感**：首跑曾在主机高负载下不稳定
   （CV 5.1% / 50.2%），同身份整组重跑通过；首跑组保留为证据。
7. **16 target 的 ≥10 监控样本门禁未实测**：16 rank 共享 smoke 已绑定全部设备，
   但计时窗口不足 15 秒；现有计数器矩阵为单卡 profiling，不能替代 16 target 轮询。
8. **官方 P2P 全设备工具矩阵已完成**：`official-p2p-20261026` 覆盖 16×15=240
   有向 pair 的 CE 单向读/写、并发双向和 GPU/CPU 时延矩阵，原始输出与解析摘要可追溯。
   运行条件是共享宿主，不能外推独占性能；工具输出注明无 P2P 路径时回退 memcopy，
   结果不与 Base staged/直连口径混合。

## 2026-10-26 能力补验

| 项目 | 新证据 | 结论 |
|---|---|---|
| 16 rank 同机通信 | `final-acceptance-20261026/16rank-shared-run1/`，16 个唯一 UUID/BDF、AllReduce 初始化/正确性/清理通过；20 ITERS 计时窗口不足 15 秒 | 共享功能通过；`partial-shared-host`，不构成 16 rank 独占性能资格 |
| 官方全设备 P2P | `final-acceptance-20261026/official-p2p-20261026/`，官方 `p2pBandwidthLatencyTest` 在 16 卡上完成 uni-write、uni-read、biBw 和 GPU/CPU latency 矩阵；每个矩阵 240 个非对角 cell，全部 `data matched`，BDF 身份完整 | 工具和拓扑覆盖完成；运行期间存在外部句柄，仍不构成独占性能资格；数字按 CE、4 MiB（时延 64 KiB）工具口径解释 |
| FP8 | `final-acceptance-20261026/fp8-probe-20261026/`，torch 2.10 有 dtype/API，PPU 上 `mm/matmul` 为 `NotImplementedError`，`_scaled_mm` 被 compute-capability 门禁拒绝 | 当前锁定 runtime `unsupported-by-runtime`；未伪造 FP8 case 或用其他 dtype 替代 |
| Asight 计数器 | `final-acceptance-20261026/counter-matrix-20261026/`，7 个可进入 profiling 的 native/FlagGems 组合各 5 kernel、205 metric records，五个 PPU section 均存在且无缺失；`mm/FP32/FlagGems` 因 probe 失败跳过 | PPU 原生计数器在代表矩阵可采；不映射为 Ascend 六组指标；所有结果因共享设备保持 partial |

### 九个失败组合责任分流

| 组合 | 已确认事实 | 当前责任结论 |
|---|---|---|
| `addmm/FP32` native、FlagGems | 输入/模块状态、设备绑定、参考重算及首次/缓存路由证据完整；最大绝对误差约 `6.46e-4`/`9.95e-5` | 不是已发现的 FlagPerf 绑定或调用错误；CPU FP32 对同一 FP64 参考也有少量超差，精度契约仍需算子组复核，原始 failed 保留 |
| `bmm/FP32` native、FlagGems | 同上；最大绝对误差约 `7.20e-4`/`8.82e-5`，输入和路由门通过 | 不改容差、不改算子；责任未完全闭合，交算子/精度规则复核 |
| `layer_norm/FP16` native | 原 case 输入为 Half，LayerNorm 默认 weight/bias 为 Float；worker 未改写模块状态，PPU 返回 dtype mismatch | 框架未注入错误 dtype；当前归为 PPU/runtime 对混合 dtype 的兼容性失败，保持 failed |
| `layer_norm/BF16` native | 与 FP16 相同，输入 BFloat16、模块参数 Float，失败发生在 dtype 检查 | 同上；不以强制转换参数掩盖失败 |
| `linear/FP32` FlagGems | 原始输入重放 300 秒仍停在 FlagGems autotuning/Triton PPU `make_hgbin` 子进程等待 | 只确认阶段超时，不能直接断言编译器根因；外部路径待责任团队继续定位 |
| `mm/FP32` FlagGems | `mm-fp32-stacks-20261024` 使用原始输入和参考，probe 300 秒超时，容器及租约清理通过 | 框架生命周期已完成；根因未闭合，保留 failed，不自动豁免 |
| `mm/FP32` nativetorch | Docker 矩阵与 local 续跑均为 correctness failed（`numerical-mismatch`）；Docker 记录最大绝对误差约 `6.50e-4`、1500 mismatch（atol 1e-5/rtol 1e-4），输入、设备绑定与路由门通过 | 与 addmm/bmm 同属 FP32 matmul 精度契约问题，一并交算子组复核；原始 failed 保留，不放宽容差 |

以上 9 个任务的原始报告、stderr、输入哈希和补测目录均未覆盖。责任分流只说明当前证据
能排除的框架环节，不把数值失败、编译等待或外部 runtime 行为改写为通过。

### 九个失败根因定位（2026-10-26 对照实验）

在不改 case、不改 runtime、不放宽容差的前提下，用锁定镜像
`flagperf-ppu-operation:654e2a3a` 一次性容器（`--network=none --cap-drop=ALL`、空闲卡）
做最小对照实验。证据
`final-acceptance-20261026/failure-rootcause-20261026/`（`mm-precision/`、
`layernorm-dtype/`、`flaggems-hang/`、`flaggems-hang-terminal/`，各含脚本副本、
stdout/stderr、returncode、sha256）。

| 组 | 组合 | 根因结论 | 责任方 |
|---|---|---|---|
| ① | `addmm/FP32` native+FlagGems、`bmm/FP32` native+FlagGems、`mm/FP32` nativetorch（5 项） | 原始输入重放默认设置即逐位复现原始失败（mm `6.495e-4`/1500、addmm `6.461e-4`/1457、bmm `7.195e-4`/3053；atol 1e-5/rtol 1e-4）；runtime 自报 `allow_tf32=False`、`float32_matmul_precision='highest'` 但显式设置后结果逐位不变（旋钮无效）；误差签名介于 CPU FP32（`5.46e-5`/14）与 TF32 输入仿真（`8.70e-2`/253223）、BF16 仿真（`2.94e-1`/254962）之间且与任一不符 ⇒ 设备 FP32 matmul 使用未声明的内部降精度格式 | PPU runtime/算子（BLAS）团队 |
| ② | `layer_norm/FP16`、`/BF16` native（2 项） | 输入 half/bf16 + `nn.LayerNorm` 默认 float32 参数：同版本 torch 在 CPU 正常、PPU 拒绝（`expected scalar type Half but found Float`）；参数与输入同 dtype 时 PPU 正常；上游 CPU 与 CUDA（NVIDIA 同 case 标准路径）均支持混合 dtype ⇒ PPU runtime 缺混合 dtype layer_norm kernel | PPU runtime 团队 |
| ③ | `mm/FP32`、`linear/FP32` FlagGems（2 项） | FlagPerf 之外最小复现（512³，`flag_gems.enable()`，锁定镜像）：FP16/BF16 mm 分别 60.0s/60.0s 完成（rc=0，含启动墙钟 60s/69s）；FP32 mm 620 秒观察窗无 DONE（矩阵实际窗口 300 秒），独立复跑 1500 秒硬上限仍未完成（`flaggems-hang-terminal/`，20:40:37→21:05:47，rc=137、DONE=0，收尾记录与容器清理见该目录 README）；FP32 linear 310 秒 stdout 停在 start 行、stderr 停在初始化告警，无 DONE（steps.log 记录 rc=0 与无 DONE 标记矛盾，原样保留）⇒ FlagGems/Triton-PPU 的 FP32 matmul 编译/执行病态 | FlagGems/Triton-PPU 团队 |

计数对账 5+2+2=9，与分流表一一对应。框架侧（case 构造、输入完整性、设备绑定、
首次/缓存路由、容差）在重放中逐一排除；FlagGems 路径 addmm/bmm 超差较小
（`9.95e-5`/`8.82e-5`），与 native 同属 FP32 matmul 精度家族，交算子团队按 kernel
分别复核。FlagGems 路径 layer_norm/FP16 在矩阵中通过，可作过渡路径，native API
契约仍需厂商补齐。9 项全部保持 failed，不放宽容差、不绕过执行。

## 维护边界

更换镜像、驱动或依赖版本后，对应范围的资格结论全部失效，须重新验证；
不能复制旧资格结论。`validated` 复评、16 rank 窗口补齐与跨机范围扩展
都是显式的维护者动作，需要新的实测记录；历史报告保留当时的 candidate
状态，不回写为新验证结果。
