# PPU Operation 机制与验收

运行准备和最小命令集中在 [PPU 算子入口](../../operation/vendors/thead/README.md)，
公共输入、容差、反向协议、报告和自定义 case 规范见 [Operation 指南](../../operation/README.md)。
本页说明 PPU 接入的证据边界。Base/Toolkit 的已有验收不代表本部分已通过。

## 实现结构

```text
operation/vendors/thead/
├── adapter.py                   # 公共 worker 接口、设备绑定和状态门禁
├── bootstrap.py                 # -S 启动、包版本校验、CPU 阶段隔离
├── host.py                      # Base 物理锁、身份/占用前后检查及清理
├── execution.py                 # 输入和模块状态的位级回传校验
├── dispatch.py、routing.py      # 首次/缓存计算及 host/device 关联
├── trace.py                     # flow、设备号和完整窗口解析
├── profiling.py                 # 独立 timeline 与 Asight 原生采集
├── profiles/torch210-ppu.json    # 固定镜像、源码、包版本和候选状态
├── Dockerfile                   # 基于既有 PPU 镜像增加固定版 FlagGems
└── README.md                    # 使用入口
```

算子和输入继续使用 `operation/benchmarks/*`，不复制 52 套 PPU case。
自定义算子按公共 `build_case(conf, cfg)` 接口提供 CPU 构造的函数/模块、输入、
是否反向和工作量函数，并在公共 catalog 注册 dtype/规模；PPU 不另设容差或替换 kernel。

## 各阶段做什么

| 阶段 | PPU 行为 | 证据和边界 |
|---|---|---|
| 规划 | 标准库解析 catalog 和参数 | 不访问设备和 Docker；dry-run 成功不等于可真实运行 |
| 宿主检查 | SMI inventory、驱动、BDF/UUID/minor、全进程句柄可见性 | 无权读取或未知持有者均拒绝；不自动提权或停止进程 |
| 租约 | 复用 Base `DeviceLease`、`thead/<uuid>` 和兼容锁 | 取得锁后再次核对身份；锁不阻止非 FlagPerf 外部任务进入 |
| CPU 参考 | `-S`，先检查锁定包版本，不执行 `.pth` 代码，设备不可见 | 保存实际量化输入；CPU FP64 参考和非均匀上游梯度沿用公共协议 |
| 设备准备 | `torch.cuda` 选可见 ordinal，核对所选物理 UUID | 输入、模块 state_dict 位级回传；设备不匹配立即失败；记录 `allow_tf32=False`（不据此推断内部计算格式） |
| 路径注册 | 原生不注册 Gems；Gems worker 显式 `enable()` | 记录实际包、注册和调用；注册成功本身不能证明目标由 Gems 执行 |
| 路由 | 首次/缓存分别采集 ATen 窗口、runtime/driver launch、ac2g flow 和 device kernel | 原生路径通常为 `cudaLaunchKernel` 或 `cuLaunchKernel`，FlagGems 也可能为 `cuLaunchKernelEx`；三者须通过同一 correlation、flow、窗口和设备关联；排除 metadata、复制/分配，缺关联保持 partial |
| 计时 | 独立 measure worker，无 profiler | 公共同步主机批计时；冷/热前向和缓存零梯度反向规则均不变；无设备计时器则如实标不支持 |
| 显存 | 独立 allocator 测量 | 保存 baseline/reset/peak，仅为该 worker 的 allocator 范围 |
| Profiling | 独立子进程，原输入重放五次 | timeline 和原生 counter 与干净计时分开，核对阶段身份，不改变原任务失败 |
| 诊断 | 原异常、精度统计、CPU 检查及显式 replay 使用公共流程 | 编译器、runtime 的错误签名不是根因证明，须结合输入、调用和最小复现分流 |
| 结束 | 按运行 ID 核对并回收自有容器/进程组，随后后检查、释放租约 | 无法确认清理/健康时保留失败，停止推进，不删除他人容器或历史 evidence |

原生路径和 Gems 路径都对同一 CPU 参考检查，不相互充当黄金结果。
原有 P800 零梯度反向协议的争议保持公共历史边界；未在 PPU 实测前，不将其归因于 PPU 算子，
也不擅自改变该协议。

## Profiling 与硬件指标

`--profiling timeline` 收集五个同步目标窗口，要求每个目标的 runtime launch、flow 和
正确设备 kernel 完整对应。CPU 时间线或设备输出不能冒充设备 kernel 执行。
跨线程反向需要明确的 fwdbwd flow 与 sequence 关联；不能仅凭时间接近拼接。

`--profiling full` 在独立重放中额外调用 PPU SDK Asight `acu`。
通过 profiler-start/stop 将采集限制到 warmup 后的五次原目标调用，不采输入搬运；
采集 `SpeedOfLight`、`MemoryWorkloadAnalysis`、`Occupancy`、`WarpStateStats`、
`LaunchStats` 原生 section（其中后两个是 SDK 文件对应的 `Identifier`），保存原生报告、原始输出与每 kernel 的 metric 名称、单位、数值。
失败、缺字段、无原生报告或身份不一致均不能作为采集成功。

这些 section 的定义来自 PPU SDK 自带文件，和 Ascend 的 ArithmeticUtilization、
PipeUtilization、Memory、MemoryL0、MemoryUB、ResourceConflictRatio 不存在已确认的一一对应。
报告中的六组兼容字段因此不填伪值，独立的 `native_counters` 记录实际 PPU 指标。
自动采集使用 `--page details --csv` 输出带单位的长表，SDK 的 raw 页宽表不用于该解析器。
SDK 在长表末尾输出的进度和报告路径不作为指标行；缺值、非有限数和无单位行不计入有效指标。
原生报告和导出的每 kernel / section / metric / unit / value 一并留存；缺少请求 section
或阶段身份不一致保持 partial。Asight 专用 `libpthread.so.0` preload 仅在锁定镜像的
计数器目标阶段允许，校验映射路径与 SHA-256，并保存在该阶段的 `bootstrap.json` 中；
此阶段专用记录不混入 probe/measure/counter 必须相同的运行栈身份字段。

当前验证站点的 `pg-exporter` 占用 PCM。已在管理员授权的临时监控暂停窗口获得原生报告，
并恢复服务；这既不是硬件不支持，也不证明监控对普通计时完全无影响。2026-10-26
扩展矩阵中 7 个可进入 profiling 的组合均获得 5 个 kernel、205 条指标和五个完整
PPU section；`mm/FP32/FlagGems` 在 probe 阶段失败，因此 profiling 明确 skipped。
适配器不会自动停止任何系统服务；其他环境应按实际工具、占用和授权核对。

## 2026-10-26 补充证据（不替代 Operation 验收）

- Base 的 16 rank AllReduce 共享 smoke 绑定 16 个唯一物理身份，初始化、正确性、清理通过；20 ITERS 窗口不足 15 秒，状态为 `partial-shared-host`，不构成独占性能资格。
- 官方 `p2pBandwidthLatencyTest` 在锁定镜像内完成 16 卡 CE uni-write、uni-read、biBw 及 GPU/CPU latency 矩阵；每个矩阵均有 240 个非对角 cell、输出 `data matched`，原始 stdout 和解析摘要位于 `final-acceptance-20261026/official-p2p-20261026/`。共享条件下不形成独占性能资格。
- FP8 探针确认 torch 2.10 dtype/API 存在，但 PPU 设备 `mm/matmul` 未实现，`_scaled_mm` 被 compute-capability 门禁拒绝；记录为 `unsupported-by-runtime`，没有新增伪实现。

## 状态与资格

| 状态/字段 | 含义 |
|---|---|
| `passed` | 该任务的正确性和所要求路由证据通过；不能脱离 performance/profiling 子状态理解 |
| `partial` | 已有结果，但路由、采集或共享性能资格等至少一项证据不足 |
| `failed` | 数值超差、执行失败、身份/输入/清理错误等；原异常保留 |
| `blocked` | 命中明确的依赖/启动错误签名；不是自动认定其他团队有错 |
| `not-run` | 未实际尝试，仍留在覆盖分母中 |
| `not-applicable` | 公共 catalog 不接受的 case/dtype 组合，不产生性能结果 |
| `performance_qualification` | Docker 独占及路由资格单独判定；授权共享和外部管理的 local 不给出宿主独占证明 |
| `validated:false` | 固定 Operation 镜像仍为候选；生成报告不自动晋级 |

无 `--allow-shared-device` 时，未确认只读的系统服务持有句柄也会拒绝。
显式共享模式仍核对身份和实际持有者，但不把持有者自动列入白名单；原结果与性能资格保留 partial。
外部任务可绕过 FlagPerf 锁，因此开始/结束观察不等于证明整个时间段都独占。

## 验收记录（2026-10-24；2026-10-27 复核）

| 项目 | 当前证据 | 状态 |
|---|---|---|
| 起点 | 基于主线 `4f50e1bd`（origin/dev-1.0），独立分支 `zhiyu/ppu-operation` | 已建立 |
| 运行镜像 | 固定 Base ID 上增加官方 FlagGems `654e2a3a`；独立候选镜像 ID 写入 profile | 已构建并完成设备 smoke |
| PPU dry-run | abs/FP32/smoke 双路径成功生成计划 | 通过 |
| CPU 参考隔离 | 锁定镜像，无设备映射，reference 退出 0；site hooks 未执行，CUDA 未初始化 | 通过 |
| Operation CPU 回归 | 路由和计数器窗口校验修复后完整 167 项通过，其中 PPU provider 定向测试通过；包含大 trace、标准 driver launch、重复/错误 flow、CPU 隔离、preload、counter section/身份/窗口及 CSV 末尾进度行负例 | 通过 |
| 公开分发 | 14 项通过 | 通过 |
| Base 回归 | 381 项测试（380 通过、1 项既有 skip），无设备映射 | 通过 |
| Docker 设备 smoke | 卡 6 `abs/FP32`（`profiling-isolated-20261023/operation-c3da51580a6d`）、卡 7 `tanh/FP16`（`representatives-cpu-isolated-20261023/tanh/operation-7f3b92c4dc85`）两路径均有正确性、测量、路由与五窗口 timeline；allocator 另有采集。卡 6/7 以共享模式运行，持有者记录含 `pg-exporter`、`ai-stack-engine` | 功能证据已获得；最终状态 `partial-shared-host` |
| local 设备 smoke | 外层复用宿主 Base 物理租约，在卡 8 容器内执行 Operation local `tanh/FP16` 双路径（`local-smoke-20261023/results/operation-edc759e7b6bf`）；输出正确性与测量已记录，外层租约释放且容器确认回收 | 功能尝试已记录；共享与 local 边界下不具备独占资格 |
| PPU `mm/BF16` FlagGems smoke | 原 probe 超过 300 秒 watchdog 的记录保留。`mm-stacks-authorized-20261023` 证明先发生冷编译、随后停在框架 trace 的重复全表扫描；原始 824,457 事件 / 83,593 kernel 经索引解析约 2.15 秒。修复后 `mm-indexed-20261023/operation-d1d339284fe6` 双路径走完正确性、测量、路由与补充采集 | 框架解析缺陷已修复；旧失败保留；共享性能仍 partial |
| Docker/local 完整 daily 矩阵 | Docker 与 local 均已覆盖计划中的 308 个适用组合；Docker 728 项计划含 420 个不适用，local 通过 28 个独立补测 run 补齐首轮 watchdog 后的未尝试项。两种执行各保留 299 个 `partial`、9 个 `failed`，原始报告和重跑目录均未改写 | 尝试覆盖完成；共享条件下仍不具备性能资格 |
| 首次/缓存、前反向、输入/模块绑定 | `abs`、`tanh`、`group_norm`、NLL/CrossEntropyLoss、`mv`、`prod` 等代表及补测组合保留 payload、首次/缓存 trace、输入/模块和路由证据；全矩阵结果按任务目录留存 | 代表性证据已保存；矩阵缓存路由仍有缺口，见 PPU-TRACE-001 |
| allocator / timeline | allocator 和五目标窗口的 timeline 已在上述共享 smoke 采集；覆盖仅限记录中的 case/dtype/输入 | 部分实测；不等于全量验收 |
| Asight 原生 counter | 历史 PCM 占用、raw 页格式失配和 details 页尾部日志异常均保留。修复后的 `counter-details-fixed-20261023/operation-d414221c112c` 自动生成 `native_counters.status=measured`：`abs/FP32` 原生路径 5 kernel、205 指标、5 section、无缺失 section、身份一致；服务约 39.4 秒后恢复 active | 自动链路已在该输入实测；full 和共享资格整体仍 partial，未外推全部 case/路径 |
| 中断/超时后的设备健康与租约释放 | 真实 watchdog 和 SIGINT 场景均完成自有 worker/容器回收、后检查和租约释放；失败记录保留 | 通过 |
| Ascend/P800 | 公共 CPU 回归通过；本轮无对应硬件回归，历史对照仅用于状态/覆盖审查 | 仅静态/CPU 范围；不作跨设备性能比较 |

共享模式的 `processes_empty=true` 只表示 `ppu-smi` 没有报告计算任务，不表示设备节点没有持有者；
应同时检查同一记录中的 `holders`。卡 6 的 preflight 已列出 `pg-exporter`、`ai-stack-engine`，
驱动日志还证明其中 `pg-exporter` 占用 Asight 需要的 PCM 资源。

冷编译 trace 的解析改用 flow 与 CPU op 索引，保留重复候选及精确 timestamp、线程、
correlation、设备和窗口核对；没有预热替代首次执行，也没有放宽路由门禁。
小 trace 与旧实现输出逐项一致，大 trace 的离线解析和哈希记录见 `trace-index-review-20261023`。

### daily 矩阵覆盖与路由缺口

2026-10-27 逐任务复读历史 `summary.json`，未重跑设备任务。Docker 全矩阵的保存源码
为 `00202323`；local 首轮及续跑按原运行身份保留，不将后续代码或文档提交冒充执行版本。
每入口原规划为 52 case × 7 dtype × 2 路径 = 728 项，其中 308 适用、420 类型不适用。

| 批次 / run | passed | partial | failed | not-run | not-applicable |
|---|---:|---:|---:|---:|---:|
| Docker `operation-c8cccff934e7` | 0 | 299 | 9 | 0 | 420 |
| local 首轮 `operation-eee0accf05f0` | 0 | 135 | 7 | 390 | 196 |
| local 续跑 `daily-local-continuation-20261023`（000–027，28 个独立 run） | 0 | 164 | 2 | 0 | 0 |

local 首轮的 390 个 not-run 包含 166 个适用项和 224 个类型不适用项；续跑任务的
case/dtype/oplib 集合恰好等于 166 个适用缺口，逐项 case_config、seed 和 profile
与原 Docker 计划一致。首轮+续跑合计覆盖 308 个唯一适用组合：299 partial、9 failed。
420 是原计划的类型不适用分母，**不是** local 首轮实际记出的 not-applicable 数；
原首轮 390 个 not-run 不回写。跨 run 只合并覆盖清单，不拼接两路径加速比或独占资格。

两入口各 299 个 partial 的 correctness 均 passed，均保存有限 median_us 与
`memory.status=measured`；routing 则各为 290 passed、9 partial，而非全部通过。

**PPU-TRACE-001（开放）**：FlagGems `cross_entropy_loss` 及原生 `mv`、`prod` 的
FP32/FP16/BF16 共 9 个组合，在每入口原始矩阵中均缺少完整的缓存调用 host/device
关联（`cached_launch_proven=false`）。首次目标窗口完整不能替代缓存路由证明；
可能涉及 trace、观察器规则或后端语义展开，现有记录不能区分。共享宿主下其余 290 项
也无独占性能资格。独立补测成功不消除这些历史缺口，修复须另留新 run 与关联证据。

无 Ascend 硬件，本轮未做 Ascend 实机回归；已有历史对照仅用于状态/覆盖审查，
不作跨设备性能比较。原失败、受干扰、中断和重跑记录均保留。

### 2026-10-23 设备 smoke 证据

共享验证结果目录：

- `ppu-operation-results/smoke-shared/operation-dc7ff4dd8132/`：首次设备运行。FlagGems 的 PPU `cuLaunchKernelEx` 尚未被旧解析器识别，保留为 partial。
- `ppu-operation-results/smoke-driverfix/operation-3935a99aae77/`：修复后重跑。原生与 FlagGems 的 correctness、measurement、first/cached routing 均为 `passed`；FlagGems kernel `abs_func_kernel_rank_1` 通过 correlation=48 和 `ac2g` flow 关联。由于显式共享设备且 smoke 只有 1 轮，最终状态保持 `partial`，不构成独占性能资格。

解析器现在严格接受三类 PPU launch 记录：原生常见的
`cuda_runtime/cudaLaunchKernel`、原生库可能使用的 `cuda_driver/cuLaunchKernel`，以及 FlagGems 常见的
`cuda_driver/cuLaunchKernelEx`。三类记录都必须满足相同的窗口、correlation、`ac2g` flow、设备号和 kernel 关联门禁。

### daily 首轮失败与定位边界

| 组合 | 观察与证据 | 处理 |
|---|---|---|
| `addmm/FP32` 原生 / FlagGems | 两入口正确性均失败，最大绝对误差约 `6.46e-4` / `9.95e-5`，不匹配 `1457` / `13` 个；输入与模块回传、参考重算、首次/缓存路由均有通过证据 | 保留原容差与失败。CPU FP32 对同一 FP64 参考也存在少量超差，因此数值大小不单独证明实现缺陷；需结合算子团队的精度契约复核 |
| `bmm/FP32` 原生 / FlagGems | local 首轮最大绝对误差约 `7.20e-4` / `8.82e-5`，不匹配 `3053` / `44` 个；参考重算、设备路由通过 | 同上，不将原生路径替代 CPU 真值，不修改 kernel |
| `layer_norm/FP16、BF16` 原生 | local probe 报输入 Half/BFloat16 与 Float 不匹配；payload 证明输入为低精度、模块 weight/bias 为原 case 构造的 FP32，并未被 PPU worker 改写 | 混合 dtype 与主线 case 一致，CPU FP16 对照可执行；当前 PPU 原生路径失败保留，兼容性根因待核实，不通过转换参数精度掩盖失败 |
| `linear/FP32/FlagGems` | local probe 超过 300 秒。`linear-stacks-20261023` 用原保存输入、同模块状态和默认 probe 复现，周期堆栈落在 FlagGems 自动调优、Triton PPU 编译的 `make_hgbin` 子进程等待；300 秒结束仍未返回 | 只确认编译/调优阶段耗时超出预算；不等同编译器错误或正确性失败。原失败保留，诊断容器已回收、后检查通过、租约释放 |

补测以新结果目录运行，不把原 `not-run` 改成已运行，也不抹除超时。跨 run 覆盖需要按
case/dtype/oplib、输入配置与运行身份核对；单次报告仍只代表自己的任务和资格。

### 失败对照实验与归因边界（2026-10-26；2026-10-27 复核）

对照实验保留在 `final-acceptance-20261026/failure-rootcause-20261026/`，详细数值与
探针局限见 [失败对照实验](validation.md#九个失败的对照实验与归因边界)。九项仍为
`failed`；复现失败或交由后端团队排查，不等于已证明全部根因或排除全部框架问题。

- `mm/addmm/bmm FP32`（5 项）：公共容差下的数值失败可复现。原生路径的误差统计与
  原矩阵一致，FlagGems 的 addmm/bmm 也保留超差。探针中 `allow_tf32` 默认即为
  `False`，重复设为 `False` 后统计不变；只读取了 `float32_matmul_precision`。
  这不能证明开关无效、结果逐位相同或内部采用未知降精度格式。CPU FP32 相对 FP64
  参考也有少量超差，需联合核查后端精度约定、累加方式与公共判定规则。
- `layer_norm/FP16、BF16` 原生（2 项）：设备 functional 调用中，低精度输入与 FP32
  参数组合被拒绝，同 dtype 参数可执行；CPU 对照只直接验证了 FP16 混合参数可执行。
  当前定位为所测 PPU 原生路径的混合 dtype 兼容性差异，交 runtime 团队复核具体原因。
  没有本轮 CUDA 对照；探针中未搬运模块的调用和跨设备误差计算不能作为有效对照。
- `mm/linear FP32 FlagGems`（2 项）：原矩阵 300 秒 probe 超时保留。FlagPerf 外
  最小脚本的 FP32 mm 在 620 秒及独立 1500 秒上限内均无完成标记；但 linear 脚本
  **完成**（`DONE ... in 301.2s`，rc=0，启动至退出墙钟 310 秒）。该脚本未验证
  输出正确性，不能替代原任务验收；证据支持首次调用延迟问题，不能仅据超时判定
  编译器错误或永久死锁。交 FlagGems/Triton-PPU 团队进一步区分编译、调优及执行阶段。

原生 FP32 数值机制、混合 dtype 的具体实现原因和 FlagGems 长延迟的根因仍待确认。
历史文件及失败不改写；尤其不能用 linear 最小脚本完成覆盖原矩阵失败。

**runtime 晋级前置条件**：当前 `operation/vendors/thead/adapter.py` 的
`runtime_identity()` 仍返回 `validated=False`。失败须按原协议修复复测，或由维护者
明确批准、限定范围并登记正式豁免；Operation 所要求的路由、测量、profiling 与独占
证据须逐项复核，Base 的 16 rank 五次资格与 16 target 监控仍需整机独占窗口补验。
补充 Base 结果不能替代 Operation 验收。维护者在完整审查新证据后才能显式修改标志；
生成报告、明确排查团队或记录例外都不会自动晋级。

验收必须在最终代码下逐项完成，保留所有失败和重新运行目录；只有完整矩阵尝试及能力项
均有实测证据后，才可声明本适配达到完整交付状态。
