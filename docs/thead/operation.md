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
| 设备准备 | `torch.cuda` 选可见 ordinal，核对所选物理 UUID | 输入、模块 state_dict 位级回传；设备不匹配立即失败；FP32 matmul TF32 关闭 |
| 路径注册 | 原生不注册 Gems；Gems worker 显式 `enable()` | 记录实际包、注册和调用；注册成功本身不能证明目标由 Gems 执行 |
| 路由 | 首次/缓存分别采集 ATen 窗口、runtime/driver launch、ac2g flow 和 device kernel | 原生路径通常为 `cudaLaunchKernel`，FlagGems 也可能为 `cuLaunchKernelEx`；两者须通过同一 correlation、flow、窗口和设备关联；排除 metadata、复制/分配，缺关联保持 partial |
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
采集 `SpeedOfLight`、`MemoryWorkloadAnalysis`、`Occupancy`、`WarpStateStatistics`、
`LaunchStatistics` 原生 section，保存原生报告、原始输出与每 kernel 的 metric 名称、单位、数值。
失败、缺字段、无原生报告或身份不一致均不能作为采集成功。

这些 section 的定义来自 PPU SDK 自带文件，和 Ascend 的 ArithmeticUtilization、
PipeUtilization、Memory、MemoryL0、MemoryUB、ResourceConflictRatio 不存在已确认的一一对应。
报告中的六组兼容字段因此不填伪值，独立的 `native_counters` 记录实际 PPU 指标。
当前原生采集接口尚待设备实测，不能因发现工具和定义就宣称计数器已完成验收，
也不能因尚未采成而断言硬件不支持。

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

## 验收记录（2026-10-23）

| 项目 | 当前证据 | 状态 |
|---|---|---|
| 起点 | 最新已取得主线 `4f50e1bd`，独立 `zhiyu/ppu-operation` | 已建立 |
| 运行镜像 | 固定 Base ID 上增加官方 FlagGems `654e2a3a`；独立候选镜像 ID 写入 profile | 已构建，设备未验收 |
| PPU dry-run | abs/FP32/smoke 双路径成功生成计划 | 通过 |
| CPU 参考隔离 | 锁定镜像，无设备映射，reference 退出 0；site hooks 未执行，CUDA 未初始化 | 通过 |
| Operation CPU 回归 | 145 项通过，含 PPU 相关负例 | 通过 |
| 公开分发 | 13 项通过，后续文档修订需复跑 | 阶段通过 |
| Base 回归 | 381 项通过、1 项既有 skip，无设备映射 | 通过 |
| Docker 最小设备运行 | 普通账号句柄不可见时拒绝；授权账号发现系统服务持有句柄后仍拒绝，未启动目标 | 门禁有真实拒绝证据，执行待验收 |
| Docker/local 完整 daily 矩阵 | 尚未执行；不能用计划分母代替运行数 | 未核实 |
| 首次/缓存、前反向、输入/模块绑定 | 代码和合成负例已覆盖，实际设备执行待确认 | 未验收 |
| allocator / timeline / 原生 counter | 接口已实现，尚无设备采集结论 | 未验收 |
| 中断/超时后的设备健康与租约释放 | 清理接口与公共回归覆盖；PPU 实际故障注入未做 | 未验收 |
| Ascend/P800 | 公共 CPU 回归通过；无对应硬件回归 | 仅静态/CPU 范围 |

验收必须在最终代码下逐项完成，保留所有失败和重新运行目录；只有完整矩阵尝试及能力项
均有实测证据后，才可声明本适配达到完整交付状态。
