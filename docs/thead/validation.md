# 平头哥 PPU 验证与维护

本文区分可执行范围、历史实测和当前代码的离线检查。当前锁定 runtime 保持
`release_stage:candidate`、`validated:false`：全部既有实测在显式
`--allow-candidate-runtime` 授权下完成。与昆仑芯 P800 的"已验证但范围受限"约定不同，
PPU 的 validated 复评条件是 16 rank 同机通信资格在独占窗口补齐后进行；在那之前，
候选状态如实保留，不因文档或例外自动晋级。

锁定镜像为 `flagtree-ppu-3.6-dev:202608`
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
被排除（probes/day9/toolkit T8 证据）；computation-BF16 保持注册 case，厂商内核缺陷
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
与稳定性判定记录。另如实记录：双向并发 P2P 在部分窗口不稳定（081237Z 双向两组
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

1. **16 rank 同机 AllReduce 资格未执行**：需要 16 卡独占窗口（协调未获批）。
   已按部分完成条款以 8 rank 完成组 C 通信资格；窗口迭代数必须在目标 rank 数的
   实测窗口内重定——AllReduce 单次迭代耗时对 rank 数**非单调**
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
7. **16 target 监控轮询未实测**：监控协议在 ≤8 target 下验证
   （单次 `ppu-smi -q` 全量解析，每 target ≥10 样本）；16 卡窗口落实后补验。

## 维护边界

更换镜像、驱动或依赖版本后，对应范围的资格结论全部失效，须重新验证；
不能复制旧资格结论。`validated` 复评、16 rank 窗口补齐与跨机范围扩展
都是显式的维护者动作，需要新的实测记录；历史报告保留当时的 candidate
状态，不回写为新验证结果。
