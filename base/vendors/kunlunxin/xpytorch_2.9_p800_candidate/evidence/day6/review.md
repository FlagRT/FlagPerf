# Day 6 review: memory bandwidth, capacity and two-card communication

- 日期：2026-09-23（UTC 执行记录；分析时区 Asia/Shanghai）
- 仓库：`klx:/home/kzhang519/Zhiyu/runtime-team/FlagPerf`，分支 `zhiyu/kunlunxin-p800`
- 计划编制时提交：`812fa67a`；当日代码提交 `045a23b7`、`26e68bc3`、`bce75eea`、`405f7afe`、`23bcd20e`、`65cc3def`、`1c01aff7`（每个运行的代码身份另见其 `code-identity.json`）
- 镜像：`flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`（digest sha256:cd53efa…，Day 1 锁定，未改动）
- 授权：`user-20260923-execute-day6-plan`，8 小时滚动窗口，见 `run-records/authorization-day6.json`

## 1. 执行摘要

| 阶段 | 结果 | 关键证据 |
|---|---|---|
| A 静态合同/干跑 | 通过：4 个 intraserver case 解析为 P800 vendor 入口、nproc 匹配；2 个 interserver case 在 resolver 阶段 `supported:false` 跳过（不启动 Docker、不取 lease） | `preflight/`、run-records |
| B 带宽 | 4 GiB payload qualification 通过：**2236.82 GB/s = 2083.20 GiB/s**（2×payload 公式，16.129 s 窗口，monitor 全绿） | `memory-bandwidth/memory-qualification-a02-card5` |
| C 容量 | 实测持有 **98132 MiB = 102.90 GB = 95.83 GiB**（98304 MiB 的 99.8%），32 次分配、1 MiB 粒度收敛、OOM 原样分类、释放经 free-memory 验证恢复 | `memory-capacity/capacity-full-a01-card5` |
| D FlagCX 探测 | XCCL 运行时加载；根因定位为 `--network none` 使 BKCL socket bootstrap 无网卡可用；改用 `--internal` bridge 后双 rank collective 成功 | `communication/flagcx-probe/`、`communication/contended/` |
| E 两卡正确性 | 通过：AllReduce 四点与 P2P 四点逐元素校验、逐 rank 绑定、公式重算全过；五个真实缺陷修复见 §5.1 | `communication/contended/` |
| F 曲线与收尾 | 见 §5.3；离线回归与 HEAD 基线一致（新增 18 项 day6 测试全过，182 项全量无新增失败） | `../../../../tests/test_p800_day6.py` |

失败与部分状态全部保留（§6），无一删除。

## 2. 带宽（main_memory-bandwidth）

- 语义：`destination.copy_(source)` 每 iteration 移动 2×payload（读+写）；分配、clone 等价验证、sentinel/变更输入检查、打印全部在计时窗口之外。
- 阶梯（卡 5）：smoke 16 MiB → 1457.97 GB/s（monitor 窗口不足按策略记 partial）；medium 1 GiB → 2224.37 GB/s；qualification 4 GiB → **2236.82 GB/s / 2083.20 GiB/s**，窗口 16.129 s ≥ 15 s，monitor 采样充足，postflight/cleanup/lease 全过。
- 早期失败保留：`memory-bandwidth/failed-01-sudo-ticket`（非 tty sudo ticket）、`memory-medium-a01-card5`（合同字段名漂移）、`memory-qualification-a01-card5`（9.6 s 窗口）。
- GB/s 与 GiB/s 由同一组 bytes/elapsed 推导（1e9 / 2^30），合同测试锁定。

## 3. 容量（main_memory-capacity，高风险门禁）

- 配置：`BOUND_REQUEST_BY_FREE_MEMORY=true`、`POST_TEST_WAIT_SECONDS=0`、INITSIZE 8192、MIN_MIB 1；`--allow-high-risk-case` 显式授权；分配步数上限 4096。
- 结果：bounded 与 full 两次搜索一致 —— 峰值 98132 MiB（102.899 GB，95.832 GiB）；轨迹 8192→16384→32768→(OOM)→20398→10199→…→1 MiB 粒度终止；18 次原始 OOM 事件原样保留并分类；tensor 全部释放，free memory 98140→98138 MiB 验证恢复。
- monitoring partial 说明：搜索窗口 0.11 s，1 Hz 采样仅得 1 个窗口内样本（阈值 10）。容量为搜索型用例（`MODE: capacity`），不受 15 s 带宽窗规则约束；该 partial 是采样策略的结构性属性，不是测量失败。
- 报告值只含成功持有的分配，不含 OOM 申请或查询到的容量。

## 4. FlagCX/XCCL capability probe（D 阶段）

1. 容器内自动加载验证：XCCL `libbkcl.so`（xccl 0792b03 [rdma]，2026-02-09）随 torch_xmlir 加载，SYMBOL_REWRITE 成功。
2. 统一入口 smoke `comm-smoke-allreduce-auto042727-56`：Gloo 双 rank rendezvous 成功；BKCL socket bootstrap 在 `--network none` 下找不到 ipv4/ipv6 网卡 → `flagcxComm is not fully initialized`。
3. **弯路与更正**：最初的无设备诊断曾误判为 `BKCL_SOCKET_IFNAME` 问题并加过一版补丁。用失败 run 的 container-create 参数**逐字重放**证明：(a) 无卡容器在设备初始化阶段即失败，socket 阶段根本没有执行；(b) 单 rank 的 `init_process_group` 不会创建 flagcx 通信器（惰性创建）；(c) 带真实 collective 的对照矩阵显示 `BKCL_SOCKET_IFNAME=lo` 与 `BKCL_NET_ENABLE_INTRA=0` 均**无效**。
4. 根因：BKCL/FlagCX 的 socket bootstrap 枚举 IPv4/IPv6 网卡并拒绝仅回环的网络命名空间。同一镜像、同一对卡，容器有真实网卡（默认 bridge 或 `--internal` bridge）即成功。
5. 修复：P800 基准容器改接**仅内部连通**的 `flagperf-p800-internal`（`docker network create --internal`，无网关、无外网路由）；隔离校验从字面 `'none'` 改为对照请求策略；两个 intraserver 入口删除已证伪的 BKCL 环境变量。宿主机前置建网命令记录在 candidate README。

## 5. 两卡通信（E/F 阶段）

### 5.1 修复的五个真实缺陷（均有原始证据）

| 缺陷 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 容器网络 | `flagcxComm is not fully initialized`；`no ipv4/ipv6 net card found` | `--network none` 只给回环，BKCL socket bootstrap 拒绝 | `--internal` bridge `flagperf-p800-internal` |
| rank 设备选择 | rank 1 首次 all_reduce `invalid resource handle` + XCCL `kernel check failed, error code=-400` | `initialize()` 把当前设备固定为首个绑定；rank 1 张量在 cuda:1 而当前设备是 cuda:0 | 驱动 `set_device(local_rank)`，身份阶段先切换 |
| 产物 binding | `rank artifact binding mismatch` | `evidence()` 返回首个绑定，与 rank 无关 | 驱动 `binding(local_rank)`；各 rank 记录自己的绑定 |
| stdout 结果行 | `stdout metric count/rank coverage differs`、`stdout rank metric differs` | 双 rank 并发写同一 stdout 导致结果行拼接；小数值 `%.6f` 丢有效位 | 单次原子写 + 全文扫描校验 + 9 位有效数字 |
| AllReduce 公式 | 4 MiB 消息报出 0.000241 GB/s | algbw 漏乘窗口内迭代数 | 与仓库参考实现（`datasize = ITERS × message / elapsed`）及 NCCL 约定对齐；同点修正为 25.73 GB/s |

### 5.2 消息尺寸说明（与计划的偏差）

计划建议 1/4/16/64 MiB 四点；`Melements` 单位是 2^20 个 float32（= 4 MiB），契约最小值 `Melements=1`，因此 **1 MiB 点无法表示**。实际曲线为 **4/16/64/256 MiB**（Melements 1/4/16/64）。

### 5.3 曲线与门禁结果

AllReduce（algbw = 窗口内总流量/窗口；busbw = algbw×2(ws−1)/ws，ws=2 时两者相等）：

| Melements | 消息 | 卡对/链路 | algbw (GB/s) | busbw (GB/s) | 窗口 (s) | 状态 |
|---|---|---|---|---|---|---|
| 1 | 4 MiB | (3,7) XPULink | 25.73 | 25.73 | 17.55 | passed |
| 4 | 16 MiB | (3,7) XPULink | 27.51 | 27.51 | 17.49 | passed |
| 16 | 64 MiB | (3,4) PCIe | 37.35 | 37.35 | 13.13 | partial（窗口不足；原样保留） |
| 16 | 64 MiB | (3,4) PCIe | **37.37** | 37.37 | **19.00** | passed（补跑，探针 1000 次迭代） |
| 64 | 256 MiB | (3,4) PCIe | 37.73 | 37.73 | 17.53 | passed |

P2P 单向（rank0→rank1，卡对 (3,4) PCIe，带宽不乘 2）：

| Melements | 消息 | 单向带宽 (GB/s) | GiB/s | 窗口 (s) | 状态 |
|---|---|---|---|---|---|
| 1 | 4 MiB | 25.19 | 23.46 | 15.14 | passed |
| 4 | 16 MiB | 31.18 | 29.03 | 16.89 | passed |
| 16 | 64 MiB | 32.98 | 30.72 | 17.44 | passed |
| 64 | 256 MiB | 33.60 | 31.29 | 17.46 | passed |

每点两 rank 数值一致（AllReduce 两 rank 差 <0.7%，P2P 差 <0.1%）；正确性阶段 cold/second/post-loop 逐元素通过。

超时演练（两卡各一次 cpu-wait 注入）：

| 演练 | 结果 | 证据 |
|---|---|---|
| interconnect-MPI_intraserver | container-probe 超时终止；postflight/cleanup/lease 全过 | `contended/contended-0923T0831-timeout-MPI-34` |
| interconnect-P2P_intraserver | 同上 | `contended/contended-0923T0831-timeout-P2P-34` |

观察（不构成结论）：4 MiB 点在 (3,7) XL 对取得 25.73 GB/s，64–256 MiB 点在 (3,4) PCIe 对取得 37–38 GB/s。跨卡对比较受租户负载与链路差异影响，**不据此判断 XPULink 与 PCIe 的优劣**；正式复跑需在同一对卡上完成整条曲线。

## 6. 失败与部分状态一览（63 个运行中 28 个 failed/partial，全部保留）

| 运行 | 状态 | 阶段/原因与处理 |
|---|---|---|
| memory-bandwidth/failed-01-sudo-ticket | failed | image-identity：非 tty sudo ticket 不跨进程；改 `ssh -tt` 内联 `sudo -S -v` |
| memory-bandwidth/memory-medium-a01-card5 | failed | container-probe：合同字段名漂移（iters≠iterations，显式映射修复） |
| memory-bandwidth/memory-qualification-a01-card5 | partial | 9.6 s 窗口（ITERS 2500→4200 后 a02 通过） |
| memory-bandwidth/memory-smoke-a01-card5 | partial | monitor 窗口内采样 <10（短窗口结构性属性） |
| memory-bandwidth/memory-timeout-a01-card5 | failed（预期） | container-probe：cpu-wait 注入，60 s 看门狗终止；清理全过 |
| memory-capacity/capacity-{bounded,full}-a01-card5 | partial | monitoring partial（0.11 s 搜索窗口，见 §3） |
| memory-capacity/capacity-timeout-a02-card5 | failed（预期） | cpu-wait 演练，同上看门狗链路 |
| communication/attempts/comm-smoke-allreduce-a01-cards56、a02/a03-cards57 | failed | host-preflight：卡 6/7 被其他租户占用（fail-closed，未建 lease） |
| communication/attempts/comm-smoke-allreduce-a01-cards57、auto042727-56、a05 | failed | container-probe：前两者为 torchrun PYTHONPATH 与 BKCL socket 问题；见 §4/§5.1 |
| communication/contended/contended-smoke-allreduce-34-a0{1,2}、fix-a0{2,3,4,5}、fix-a01 | failed | 修复过程中的中间尝试：socket（网络修复前）、kernel check（设备选择修复前）、binding mismatch、stdout 合并、container-inspect 网络策略（逐项修复后通过） |
| communication/contended/contended-0923T0824-qual-MPI-m1-34-a0{1,2} | failed | 公式缺陷期的产物（数值明显不合理，驱动修复） |
| communication/contended/contended-0923T0826-qual-MPI-m1-34-a0{1,2} | partial | ITERS 尺度错误（探针换算 bug）导致的 0.17 s 窗口 |
| communication/contended/contended-0923T0831-qual-MPI-m16-34-a01 | partial | 13.13 s 窗口；补跑 19.00 s 通过（§5.3） |
| communication/contended/contended-0923T0831-timeout-MPI/P2P-34 | failed（预期） | 两卡超时演练，看门狗链路全过 |

## 7. 超时与清理演练

- 单卡：带宽与容量各一次 cpu-wait，60 s 看门狗终止，postflight/cleanup/lease 全过。
- 两卡：AllReduce 与 P2P 各一次，同上（§5.3）。
- 租约与锁：全部运行以同一双协议 lease + 设备锁运行，逐次获取/释放；连续运行反复重获同一批卡锁，未出现残留锁；最终一批运行后目标卡显存回到运行前水平（见 §9 的收尾核对）。

## 8. 回归与测试

- 新增 `base/tests/test_p800_day6.py` 18 项：2×payload 带宽公式、algbw/busbw 因子（ws=2 时 busbw==algbw，无额外 ×2）、窗口内总流量公式、P2P 单向不翻倍、Melements→字节数、容量释放验证与 1 MiB 粒度、高风险集合仅含 capacity、interserver `supported:false`、artifact 防篡改（含禁止的 busbw ×2）、报告再生（sha256 一致）。
- 全量 `python3.12 -m unittest discover -s base/tests`：182 项，1 failure + 14 errors —— 与 HEAD 基线**完全一致**（torch 缺失/Ascend 专用导入等既有项），无新增失败。
- `compileall` 通过（仅旧 baseline 归档内 SyntaxWarning）。

## 9. 第三道 go/no-go：两卡门禁

**结论：GO（qualification 级证据；正式复跑待空闲卡窗口）。**

- **正确性**：AllReduce 与 P2P 四点全部逐元素通过（cold/second/post-loop）。
- **映射**：每 rank 记录并验证自己的绑定（framework_local_rank、设备节点、PCI BDF、UUID），与官方单卡证据同一校验路径。
- **公式**：algbw/busbw 与 P2P 单向公式可重算；ws=2 时 busbw==algbw；修正前的漏乘迭代数缺陷见 §5.1。
- **monitor**：qualification 级运行窗口 ≥15 s、每 rank 样本数满足策略下限，monitoring passed。
- **超时与清理**：两卡各一次看门狗演练通过。

**限制（必须随结论一起引用）**：所有通信运行在**其他租户正在使用的卡**上完成（当日用户明确授权，临时门禁覆盖见 `communication/temporary-contended-override.patch`），带宽数字受外来负载影响，只作 qualification；**正式门禁需在空出的一对卡上复跑**。在正式复跑完成之前，第七天不得依据本结论执行 8 卡 AllReduce。

## 10. 遗留与建议

1. **正式两卡复跑**（第七天第一任务）：同一 HEAD 代码、空闲卡对、monitor on、≥15 s 窗口，重跑 AllReduce 与 P2P 曲线。
2. **容器网络前置条件**：`flagperf-p800-internal` 需在宿主机预先创建；建议后续把网络存在性纳入 host preflight。
3. **消息尺寸**：`Melements` 粒度 4 MiB，1 MiB 点不可表示；如需需扩展契约。
4. **卡健康排除**：卡 1（内核异常同步超时）与卡 2（8/8 不可纠正 ECC + pending remap）全程排除；今日覆盖的卡对为 (3,4)、(3,7)、(0,3)。
5. **临时门禁补丁**：`base/vendors/kunlunxin/preflight.py` 的 contended 覆盖（`P800_ALLOW_FOREIGN_HANDLES`，默认关闭）仅用于当日授权运行，当日结束时以 `git checkout` 还原。
6. **内存/容量证据的容器网络差异**：产生于 `--network none` 时代；两类用例不使用网络（无 collective），如需完全同构可择机重跑。
7. **曲线卡对混合**：AllReduce 前两点在 (3,7)、后两点在 (3,4)（当日租户占用动态所致），正式复跑应在同一对卡上完成整条曲线。
