# P800 Operation 机制与维护

[返回 P800 Operation](../../operation/vendors/kunlunxin/README.md) · [验证与已知问题](operation-validation.md)

本页说明现行执行机制和维护命令；历史批次与底层失败单列，不把覆盖数当作通过认证。

## 镜像取得与身份

无公共 registry 地址时，可由获准的镜像提供方用 `docker image save --output p800-runtime.tar IMAGE` 离线交付，
接收方执行 `docker image load --input p800-runtime.tar`，再用 `docker image inspect IMAGE --format '{{.Id}}'`
核对独立锁中的完整 image ID。`IMAGE` 取锁文件中的 tag；传输前后另核对归档 SHA-256。
执行器只接受锁定 ID，换 tag 或导入成功不代表包、设备或算子验收通过；本项目不附带镜像归档。

精确镜像及包版本见 [Operation 独立锁](../../operation/vendors/kunlunxin/profiles/p800-xpytorch29.json)。

## 全类型规划

全类型双路径 daily 规划（仍不访问设备）：

```bash
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 --oplib both --workload daily \
  --dtype FP32 FP16 BF16 INT32 INT16 BOOL INT64 \
  --result-root operation/result/p800-daily --dry-run
```

去掉 `--dry-run` 前仍需满足上面的运行栈、权限和设备授权条件。
308 个适用组合表示计划规模；运行结束后按实际 failed/blocked/partial 等状态读取结果，不能据覆盖数认定通过。

## 引导、身份与生命周期

- Docker 使用锁定 Conda Python 的绝对 entrypoint 和 `-S`。执行 site hooks 前检查 Python、包元数据与 Triton literal version。
  CPU 的 reference/diagnose/reference-check 不映射 XPU、不执行 `.pth` 代码，结束时核验没有导入设备后端或初始化 CUDA 兼容 API。
  local CPU worker 处于调用者现有命名空间，仍不导入/初始化 XPU。
- probe/measure/trace/memory 及独立 profiling 目标执行设备阶段。Docker 只映射选中实际节点和 `/dev/xpuctrl`，网络关闭。
  先观察完整可见框架 UUID 集合并与宿主 UUID 匹配，再选逻辑设备；不猜测物理号、minor 与框架编号相同。
- 两条路径均设置 `USE_FLAGGEMS=0`、`XRAY_DEBUG_ENABLE=0`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；
  FlagGems worker 只在 CPU 输入/参数构造后显式调用公开的 `flag_gems.enable()`。
  跨 case 复用时检查构造张量仍在 CPU。两种 oplib 和不同阶段的 worker 独立。
- trace 设置 `XPU_ENABLE_PROFILER_TRACING=1`，采集公共 CPU 事件诊断；它不等价于设备 kernel 时间线。
  probe/profiling/profile-target 同样设置为 1，并通过 `XPU_CUPTI_ENABLE_DEVICE` 选择已绑定的逻辑设备；
  Docker 单设备命名空间为 0，local 为调用者选择的逻辑编号。measure/memory tracing 为 0，CUPTI 设备选择为空，保持计时进程独立。
- probe/measure 对比解释器、包、hook 哈希、关键环境、输入、源码、profile 和注册状态；不同则 failed。
  只允许 probe 的 tracing=1、CUPTI=已绑定逻辑号到 measure 的 tracing=0、CUPTI=空串这两项预期转换，
  原始身份不改写，转换记录在 `phase_consistency.expected_profiler_environment`；其他环境或运行身份变化仍失败。
  计时进程没有 Python 跟踪，计时后另做同输入正确性检查。memory 按公共 allocator-window 协议报告，不是整卡 HBM 使用量。
- 宿主默认复用 Base provider 的严格占用检查。锁前观察，持有 `/tmp/flagperf-base-device-leases` 中的 UUID 锁及
  `/tmp/flagperf-p800-<UUID>.lock` 兼容锁后重查；这些路径是 Base/Toolkit 共用协议，不能更换以绕过互斥。
  共享模式复用同一组身份/健康解析器，记录占用，不声称空闲。`P800_ALLOW_FOREIGN_HANDLES=1` 仍明确拒绝。
- 仅清理本轮 run ID 标签匹配的容器，确认消失后再继续；local 仅回收本轮子进程组。
  watchdog、SIGTERM、启动失败均进入清理，postflight 在释放租约前执行；清理/健康未确认则停止后续执行。
  Docker 自动删除尚未完成时，对已确认身份的容器最多再等待 5 秒并查询消失状态；超时、查询失败或仍存在均不算清理成功。

宿主 root 启动和 `P800_KNOWN_READONLY_MONITORS=barad_agent` 是 klx 的部署记录，均非通用要求。
只读观察者须按 Base 规则显式授权并验证进程身份；不能将任意持卡 Python 进程列为只读观察者。
P800 不接受 `--allow-privileged-root`、额外 `--container-device` 或 `--mount`。

## 输入核验与失败结果

对齐 Ascend 的公共测试机制，框架适配与算子通过率分别报告。P800 在每个设备阶段、目标调用及计时之前，
回传并核验保存输入、模块参数/缓冲区的 shape、dtype、requires_grad 和逻辑元素字节哈希，并检查设备绑定。
`<phase>-payload.json` 保存比较结果；不一致直接失败，不继续调用目标。
这是逻辑值核验，不证明 stride、存储别名或完整的目标路由/无 CPU fallback。
首个 FlagGems 任务在审计后显式注册；复用 worker 可能已经注册，仍逐任务审计。
probe 保存 `output.pt`（反向含梯度）；测量进程在计时外校验的实际结果保存为 `measurement-output.pt`。
测量中途失败时可能没有该文件，须按阶段原始异常读取，不能补造输出。

只修复已确认的 FlagPerf 数据准备、调用或记录错误。算子数值超差、编译和运行失败保持原始状态、
输入、输出（如已生成）、异常及调用栈，交算子/编译器团队处理；不替换算子、改 workload、改变数学模式、
放宽容差或关闭注册来取得通过。框架执行正确并不意味着算子通过，failed/blocked 不自动晋级。
路由、计数器等框架能力缺口仍单独验收，不用“算子问题”豁免。

## 显存与设备 profiling

memory 使用独立 worker，按公共 `allocator-window-v1` 协议采集预热后基线、重置后计数、保留目标输出时的峰值及增量。
它描述当前进程分配器，不是整卡 HBM、物理总流量或其他进程占用。原始计数和归一化规则保存在 `memory.json`。
输入、源码、包、注册、设备绑定与 clean measurement 不一致时，采集保持 partial，保留原始观察，不展示错配的归一化指标。

正确性及 probe/measure 身份一致、无已观察回退或执行错误时，P800 允许路由/共享性能为 partial 的任务进行独立诊断采集。
结果写明 `qualification.scope=diagnostic-only`；采集成功不改变原任务、路由或性能资格。failed/blocked、正确性失败、身份不一致、
已观察回退或清理/健康异常不会通过此门禁。其他厂商继续使用原有附加阶段门禁。

```bash
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke --profiling timeline \
  --result-root operation/result/p800-timeline --dry-run
```

权限、设备授权和运行栈满足前述条件后去掉 `--dry-run`；已获共享授权才加 `--allow-shared-device`。local 使用同一个 profiling 参数。
`--profiling` 默认 off；timeline 请求设备时间线，full 请求时间线及硬件指标覆盖检查。

- 时间线使用锁定栈的 XPU CUPTI CUDA 兼容接口，在独立子进程恢复同一保存输入及模块参数，预热后记录 5 次原调用及完成同步。
  主机目标窗口只用于选择主机启动；用 correlation 和 `ac2g` 流关联设备 kernel，不直接用主机/设备时间戳包含关系推断归因。
  反向在其他 CPU 线程启动时，额外要求唯一 `fwdbwd` 流、匹配的前反向 Sequence number 和目标窗口内的完整 CPU 调用；仅时间重叠不能关联。
  五次调用及每个启动都必须有唯一、完整的相关事件；缺事件、错误设备、重复/歧义关联、身份不符或目标错误保持 partial。
- 报告分别显示 kernel 累计、执行区间并集及首尾跨度，时间单位为 μs；原始 Chrome trace、采集日志、输入身份、解析器 SHA256 和兼容工具版本保留。
  CUDA/GPU 字样是该运行栈的接口命名，不表示测试使用 NVIDIA 硬件；实际 P800 身份由 UUID 绑定和运行栈记录确定。
- CUDA 兼容 trace 中的 grid/block、occupancy、寄存器、SM 等字段尚未在 P800 验证，实测存在异常值，不能作为 P800 硬件计数器。
  full 的六组硬件指标目前没有可信采集器及语义映射，均记录 not-supported，整体 profiling 保持 partial；不以这些字段、逻辑带宽或主机时间代替。
- 设备 timeline 成功只证明关联事件已采集，不证明所有复合/反向语义都在设备执行，也不自动证明没有 CPU fallback。
  它与原有路由、正确性、独占性能门禁独立。超时终止并回收子进程；local 子进程保留在 worker 的进程组中，Docker 受同一容器清理覆盖。

解析器在关联不完整时保留预期与实际的 call/correlation 集合，并列出没有设备 kernel 的 host launch；
这些诊断字段不改变 `partial` 判定。full 模式还探测已加载 CUPTI 库的输出契约，成功返回但输出保持哨兵值时按
`not-supported` 处理，避免把兼容接口空实现误报为硬件计数器能力。

宿主还发现 `xprofiler` 工具及信号配置入口，但信号定义、计数单位、共享卡进程隔离和目标归因尚未验证；
工具存在不等于六组指标已经可用。未用整卡观测或兼容 CUDA 字段填充缺失的计数器。
已核对的 XRE 5.0.21.47 工具包未提供事件配置生成脚本。硬件计数器接入还需要适配该工具版本的配置、
各信号的含义/单位与取样规则、丢样检测及进程或独占窗口归因；工具帮助中的 API/SSE 采集不是这些指标的替代品。
不同芯片的本地存储和执行单元没有天然一一映射关系，不能仅因 Ascend 有同名组便填入推测值。

## 路由证据与结果边界

P800 在独立 probe 的首次/缓存原调用中观察 ATen 派发、主机 launch、`ac2g` 与设备 kernel，
不替换数学实现，不在 measure 中启用 Python dispatch 或 profiler。
保存 `route-first-trace.json`、`route-cached-trace.json` 及对应 `*-dispatch.json`。
每个有设备输入的非元数据计算都须在自己的标记中关联正确设备 kernel；每个 kernel 都须归于计算或显式准备操作。
只有复制、分配、view/transpose、注册、编译、另一线程的邻近 kernel 或不完整关联均不能通过。
CPU 梯度权重准备单列记录；设备输入产生 CPU 张量或最终输出不在目标设备时失败。

锁定 XPyTorch 的 `linear` 使用不经 Python dispatch 的 C++ `LinearFunction`。该特例要求唯一、完整的
CPU 记录及同线程 launch，FP32/FP16 的 XBLAS FC kernel，或 BF16 的两组 findmax/cast 加 FC 完整有序链。
辅助分配单独归类，缺转换、仅辅助 kernel、错误类型/线程/设备或重复记录均不能套用此规则。
`aten::_has_same_storage_numel` 仅在该目标中作为存储元数据检查；其他目标不据此豁免。

目标日志 begin/end、首次及缓存证据、同步和正确设备输出须同时成立。新 dispatch 证据存在而不完整时保持 partial，
不能被旧 abs/add/sum 的 Python launcher 白名单覆盖；旧证据仍按原规则解释，不回写历史。
目标抛错时保留原异常，并尽可能导出失败 trace；导出失败不能覆盖算子异常。
路由通过表示上述目标设备执行证据成立，不保证所选 Gems 路径一定对应专用 Gems kernel，
也不构成覆盖所有 C++ 后端内部指令和 CPU fallback 的通用证明。注册 Gems 后走原生设备实现不自动等同 CPU fallback。


## 证据、复放与回归

每个 run 保存源代码、原始 case YAML、解析后配置、profile、使用的 Base 纯检查文件及其静态传递依赖，按原字节封存。
`source-manifest.json` 和公共 `artifacts.json` 覆盖这些文件；任务稳定身份包含公共源码、独立 case 源码/默认配置、profile 及宿主 UUID。
随机 run ID、Git HEAD 和文档不混入稳定指纹；失败和不适用项也保留身份。
worker 的 `bootstrap.json`、每阶段 identity/退出记录、probe 的首次/缓存窗口和 cleanup 记录用于追溯初始化、编译尝试及清理耗时。
首次目标窗口可能使用磁盘缓存，不声称是清空缓存后的纯编译耗时。

```bash
python3 -B operation/run.py report --run-dir /absolute/path/to/operation-run
python3 -B operation/run.py diagnose --source-task /absolute/path/to/case-result --dry-run
# 明确需要设备复放时增加 --replay；已获共享授权时再加 --allow-shared-device。
python3 -B -m unittest discover -s operation/tests -p 'test_p800*.py' -v
```

报告中的 P800 Docker 复放命令不使用 `--allow-privileged-root`；共享来源会带 `--allow-shared-device`，
执行前仍须确认当前设备授权。旧版报告若含不适用的特权参数，应按当前模板修正命令，或在新目录汇总原始 run；
不要改写封存数据来修正命令文字。


单元测试不占用设备。全套 Operation 回归需要 CPU PyTorch 与 PyYAML；标准库门禁与实际硬件验证范围分开记录。
聚合要求同主机、镜像及执行模式；不同 UUID/profile/源码/协议保留独立行。
源编排、清理、postflight 或租约释放错误会拒绝聚合；算子 failed/blocked/partial 保留。
公共 schema、报告列布局、原数值容差和 Ascend 命令保持公共契约。
