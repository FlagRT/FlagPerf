# Kunlunxin P800 Operation

使用 `--vendor kunlunxin --runtime-profile p800-xpytorch29` 选择 P800 专用入口。
支持 Docker/local、原生 PyTorch 与显式 FlagGems，以及公共参考、取证、计时和诊断流程。
Operation 独立锁保持 **`validated:false`**；Base 的发布声明不适用于算子结果。
省略 runtime profile 时仍走原有昆仑芯兼容入口。

## 环境与最小运行

从 FlagPerf 根目录执行。规划只需 Linux/Python 3.10+ 标准库，不访问 Docker 或设备。
真实 Docker 执行需要镜像、驱动、设备权限、`xpu-smi`、`fuser`，以及完整的宿主 `/proc` 句柄读取权限。
执行器不自动 sudo、下载镜像、安装依赖或清理他人进程。
镜像、Python 和包的精确版本见[独立锁](profiles/p800-xpytorch29.json)；tag 或 `--image` 别名必须解析为同一 image ID。
镜像没有验证匿名 registry pull，需由使用者取得。

```bash
# 0 仅为示例：先确认实际设备授权与占用。
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --case add --dtype FP32 --oplib both --workload smoke \
  --result-root operation/result/p800-smoke --dry-run
```

确认计划、权限和资源后去掉 `--dry-run`。每次执行创建新的 `operation-<id>` 子目录。
默认要求独占；获得共享授权后可显式增加 **`--allow-shared-device`**，允许现有占用，但仍持有 UUID 租约、
记录真实占用/句柄并检查身份及健康。共享运行的正确性和路由分别判定，性能资格保持 partial，
报告不产生合格的性能比值；正常完成也可能退出 2。这不是资源预约工具，不应干预占用者。

`--runtime-profile` 选择运行栈，`--workload smoke|daily` 选择工作量（默认 daily）；
旧 `--profile` 是 workload 的兼容别名，不能与 `--workload` 同时指定。
case、dtype、size、迭代、容差与退出码遵循[公共入口](../../README.md)，不套用 Base 的资格门槛。
全类型双路径单卡规划为 728 项，其中 308 适用、420 类型不适用；默认类型双路径为 104 项。
Docker 物理编号须唯一且在 0..7，没有固定排除卡。

## local 运行

由调用者在宿主或自己的容器中准备相同版本的 Python、包和动态库，并在外部预约宿主设备。
`OP_LOCAL_PYTHON` 是示例变量，须指向该环境的 Python 绝对路径；CPU 测试 venv 不适用。
锁文件中的 Conda 路径仅是锁定镜像的内部路径，不能直接套用于另一台宿主。

```bash
: "${OP_LOCAL_PYTHON:?Set the verified Python absolute path}"
"$OP_LOCAL_PYTHON" -B -S operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --execution local --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke \
  --result-root operation/result/p800-local --dry-run
```

准备好运行栈后去掉 `--dry-run`，共享设备另加 `--allow-shared-device`。
`-S` 让 CLI 在执行 site hooks 前启动；所有阶段 worker 继承同一解释器，并由专用 bootstrap 检查包版本。
local 保持调用者的设备 visibility，编号是当前框架可见编号（非负且唯一），不是宿主物理号。
local 只有当前命名空间的逻辑锁，记录 externally-managed，不宣称完成宿主预约或镜像核验；
多个 local run 不支持公共 Docker 聚合，只能分别重建报告。

## 引导、身份与生命周期

- Docker 使用锁定 Conda Python 的绝对 entrypoint 和 `-S`。执行 site hooks 前检查 Python、包元数据与 Triton literal version。
  CPU 的 reference/diagnose/reference-check 不映射 XPU、不执行 `.pth` 代码，结束时核验没有导入设备后端或初始化 CUDA 兼容 API。
  local CPU worker 处于调用者现有命名空间，仍不导入/初始化 XPU。
- probe/measure/trace/memory 执行设备阶段。Docker 只映射选中实际节点和 `/dev/xpuctrl`，网络关闭。
  先观察完整可见框架 UUID 集合并与宿主 UUID 匹配，再选逻辑设备；不猜测物理号、minor 与框架编号相同。
- 两条路径均设置 `USE_FLAGGEMS=0`、`XRAY_DEBUG_ENABLE=0`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`；
  FlagGems worker 只在 CPU 输入/参数构造后显式调用公开的 `flag_gems.enable()`。
  跨 case 复用时检查构造张量仍在 CPU。两种 oplib 和不同阶段的 worker 独立。
- trace 仅在诊断 worker 设置 `XPU_ENABLE_PROFILER_TRACING=1`，采集公共 CPU 事件诊断；
  probe/measure/memory 设为 0。trace 不等价于目标设备 kernel 证明；timeline/full profiling 尚不支持并在入口拒绝。
- probe/measure 对比解释器、包、hook 哈希、关键环境、输入、源码、profile 和注册状态；不同则 failed。
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

## 路由证据与结果边界

P800 通过观察现有调用取证，不替换或包装数学库/后端。路由通过范围目前仅为已核验的 abs/add 简单前向目标：
实际 kernel 名称须分别为 `abs_func_kernel_rank_1` / `add_func_kernel_rank_1`，最近一层 FlagGems ops/fused 调用祖先须为对应目标实现。
其他名称、维度实现及尚无成功目标证据的 mm 保持 partial；新增通过范围须另行审阅实际实现与启动证据。
首次和缓存窗口均须观察到
真实 `self.launch` 的 C 调用正常返回，并完成同步。目标区间须有完整 begin/end。
仅 Python 调用、编译、注册、辅助复制/分配/transpose 或 CPU profiler 事件均不能使路由通过。
原生、复合及反向目标缺少充分证据时保持 partial；注册 FlagGems 后使用原生实现不自动等同于 CPU fallback。
CPU 目标输出判 failed；尚未建立覆盖所有原生实现的通用 CPU fallback 检测器。

2026-10-13 的共享卡最小验证范围：

| 组合 | 数值链路 | 目标路由 |
|---|---|---|
| Docker/local abs、add FP32，原生 | probe/measure 通过 | partial：缺少具体原生目标派发证据 |
| Docker/local abs、add FP32，FlagGems | probe/measure 通过 | 首次及缓存 XPU launcher 证据通过；共享性能仍 partial |
| Docker/local mm FP16，原生 | probe/measure 通过 | partial |
| Docker/local mm FP16，FlagGems | 编译受阻，未产生有效测量 | 锁定栈的 `OutOfResources: uni_sram PassManager::run failed` |

上述 mm 异常只有同时匹配异常类型、FlagGems mm 与 Triton XPU `make_ttxir` 的真实调用栈才分类 blocked；
仅错误关键词不能分类，未知异常仍 failed。没有修改库、缩小 smoke 规模或改变容差来绕过失败。
以上 2026-10-13 最小验证不代表全量验收；后续覆盖范围见下节，仍不据此晋级 Operation runtime。

## 已执行的覆盖范围（2026-10-14）

以下为锁定栈上 Docker / 单卡共享授权模式的实际评估；local 仍以上述最小验证范围为准。

| 范围 | 组合数 | 结果 |
|---|---:|---|
| 52 case 默认类型 smoke | 104 | 2 blocked、8 failed、94 partial |
| 低精度代表补测 | 6 | 1 blocked、2 failed、3 partial |
| 10 个异常 case 独立双路径复测 | 20 | 2 blocked、8 failed、10 partial |
| 29 case daily，七种 dtype 双路径 | 406 | 232 not-applicable、174 partial |

daily 的 406 项包含 174 个适用组合和 232 个 catalog 类型不适用项，涵盖算术、激活、比较及位运算 29 个 case。
使用公共默认 warmup=5、iters=20、rounds=10、seed=2026；其余 daily case 尚未完成。
这 174 个 daily 适用项的 probe 与 measure 正确性均通过；执行清理也已确认。
partial 需结合逐项正确性和路由读取；共享计时不用于独占性能资格，运行栈仍为 validated:false。

10 个 smoke 异常 case 经独立双路径 run 复现，原状态和失败阶段一致。除 mm/mv 的 uni_sram 编译阻塞外，
FlagGems addmm/bmm 出现 LLVM bad signature 断言，cross_entropy_loss 出现 kernel -714 异常，
layer_norm 出现 Triton autotune 计时除零，log_softmax/native_dropout/native_group_norm/softmax 出现
invalid device function。log_softmax 还存在 probe 数值门禁失败。低精度补测保留 mm FP16、layer_norm BF16、
native_dropout FP16 的对应异常。以上记录错误发生位置，不把它们当作已查明的最终根因。
复现可使用上方 run 模板，明确指定对应 --case/--dtype/--oplib，并使用新的结果目录；不降低尺寸或容差来改写结论。

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

单元测试不占用实际设备。全套 Operation 回归还需 CPU PyTorch 与 PyYAML。
聚合要求相同主机、镜像与执行模式；不同 UUID/profile/源码/协议保留独立行。
源编排、清理、postflight 或租约释放错误会拒绝聚合；算子本身的 failed/blocked/partial 保留。
不改公共 schema、报告列布局、原数值容差与 Ascend 命令。
