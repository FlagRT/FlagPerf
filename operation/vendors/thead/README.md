# 平头哥 PPU 算子评测

PPU 使用公共 Operation catalog、输入生成、CPU 参考、容差、前反向协议和报告。
`--vendor thead --runtime-profile torch210-ppu` 选择 PPU SDK 的 CUDA 兼容接口；这里的
`torch.cuda` / `cuda:N` 不表示 NVIDIA 硬件。Docker 的 `--device-ids` 是宿主物理编号，
容器内 `cuda:0` 是本次所选卡的可见编号，必须结合 UUID、PCI BDF 和设备节点查看。

**验收状态：**接口及 CPU 回归已验证；PPU 已完成少量共享条件下的 Docker/local 双路径 smoke。
`abs/FP32` 与 `tanh/FP16` 在各自 smoke 输入上正确性、测量路径和设备 timeline 均有实测记录，
但最终结果保持 `partial-shared-host`，不能作为性能资格或完整验收。全量 daily 矩阵尚未执行。
Asight 原生计数器已在获准暂停监控的窗口采到 5 个 kernel 的原生报告，离线导出得到 205 条带单位指标；
最终自动采集链路仍需复测。此前 `mm/BF16` FlagGems probe 超时暴露了 trace 解析的重复扫描问题，
已修复并完成双路径复测；历史失败及其他验收边界见
[Operation 机制与验收](../../../docs/thead/operation.md)。
不能将 Base 验收或 CPU 测试当作本部分通过。运行栈保持 `candidate / validated:false`，运行不会自动晋级。

## 1. 运行栈准备

规划入口需要 Linux、Python 3.10+，不要求 PyTorch；Docker 宿主身份检查还需 Base 的
Python 3.11+ 和 [host 依赖](../../../base/requirements-host.txt)、Docker、PPU 驱动、
`ppu-smi`、`fuser`，以及能读取全部进程句柄的已授权账号。权限不足时直接拒绝，
不会将无法读取的占用当作空闲，也不会自动使用 sudo。

Base 镜像 `flagtree-ppu-3.6-dev:202608` 未预装 FlagGems。Operation 使用独立镜像
`flagperf-ppu-operation:654e2a3a`，在该 Base 镜像上安装固定源码的 FlagGems，原 Base 镜像不变。
完整身份及包版本在 [profile](profiles/torch210-ppu.json) 中：Python 3.12.3、PyTorch 2.10.0、
FlagTree `0.6.1+ppu3.6`；FlagGems 来源提交为 `654e2a3a7956cc675886d71defbc1ea5aa6f69ba`，
包版本 `0.0.0+g654e2a3a` 是源码快照标签，不是官方正式版本号。

运行前核对已取得镜像的 ID 与 profile 一致：

```bash
docker image inspect flagperf-ppu-operation:654e2a3a --format '{{.Id}}'
```

本仓库不分发厂商镜像，也未验证匿名 registry pull。维护者可按以下步骤构建独立镜像：

```bash
# 先取得 Base 镜像，并核对其 ID 与 profile 的 base_image_id 一致。
BUILD_DIR=$(mktemp -d)
git init "$BUILD_DIR/FlagGems"
git -C "$BUILD_DIR/FlagGems" remote add origin https://github.com/flagos-ai/FlagGems.git
git -C "$BUILD_DIR/FlagGems" fetch --depth 1 origin 654e2a3a7956cc675886d71defbc1ea5aa6f69ba
git -C "$BUILD_DIR/FlagGems" archive --format=tar FETCH_HEAD > "$BUILD_DIR/flaggems-source.tar"
sha256sum "$BUILD_DIR/flaggems-source.tar"
docker build -f operation/vendors/thead/Dockerfile \
  -t flagperf-ppu-operation:654e2a3a "$BUILD_DIR"
```

Dockerfile 核对源码归档哈希，仅增加 FlagGems 及其指定 Python 依赖，不替换 PyTorch、
FlagTree 或 SDK。重新构建不承诺镜像字节相同；若生成新 ID，维护者需更新候选 profile、
记录构建身份并重新验收，不能只换 tag 继承旧结论。

## 2. 最小案例

从 FlagPerf 仓库根目录执行。示例卡号不代表设备可用。

```bash
python3 operation/run.py list --names-only
python3 operation/run.py list --case abs

# 只生成计划，不访问 Docker 或设备。
python3 operation/run.py run --vendor thead --runtime-profile torch210-ppu \
  --device-ids 6 --case abs --dtype FP32 --oplib both --workload smoke --dry-run

# 确认镜像、权限和空闲卡后执行。结果应写到持久盘。
python3 operation/run.py run --vendor thead --runtime-profile torch210-ppu \
  --device-ids 6 --case abs --dtype FP32 --oplib both --workload smoke \
  --result-root operation/result
```

默认要求独占，执行前后读取 SMI 与进程句柄，使用与 Base 相同的物理资源锁，
再核对容器中观察到的 UUID。只映射所选 `/dev/alixpu_ppuN` 和公共控制节点 `/dev/alixpu`；
不使用 privileged 容器。CPU 阶段不映射设备，不执行 site hooks，不注册 FlagGems。

若管理员明确授权共享测试，可在上述运行命令中加 `--allow-shared-device`。
所有真实持有者会被记录，既不把它们自动认定为只读服务，也不停止它们；
测量与性能资格保持 partial。默认门禁仍拒绝未经确认的监控/管理服务。

local 入口需在已准备相同 Python/包版本、可见卡已预留的环境中执行：

```bash
python3 operation/run.py run --vendor thead --runtime-profile torch210-ppu \
  --execution local --device-ids 0 --case abs --dtype FP32 --oplib both --workload smoke
```

local 不会替调用者启动容器、安装依赖或预留宿主设备；其逻辑锁只保护当前命名空间，
报告明确保留外部设备预留边界，不给出宿主独占证明。

## 3. 扩大范围与查看结果

`--oplib nativetorch` 为默认值；`flaggems` 显式调用 `flag_gems.enable()`；`both` 分别运行两条路径，
共享相同的保存输入和 CPU 参考。两条路径都独立检查正确性，不用原生路径替代 CPU 真值。

```bash
# 全 catalog、全 dtype、双路径 daily 计划；先审查分母和资源再去掉 --dry-run。
python3 operation/run.py run --vendor thead --runtime-profile torch210-ppu \
  --device-ids 6 --dtype FP32 FP16 BF16 INT32 INT16 BOOL INT64 \
  --oplib both --workload daily --dry-run

# 正式计时外，独立重放采集五次目标时间线及 PPU 原生 Asight 指标。
python3 operation/run.py run --vendor thead --runtime-profile torch210-ppu \
  --device-ids 6 --case abs --dtype FP32 --workload smoke --profiling full --dry-run
```

输出结构与 Ascend/P800 相同：`operation/result/operation-<ID>/report.md`、`summary.json`、
`artifacts.json`，以及逐任务 `result.json`、`correctness.json`、`measurement.json`、
`memory.json`、`profiling.json`、保存的输入输出和原始日志。
离线报告入口为 `python3 operation/run.py report --run-dir operation/result/operation-<ID>`。

首次/缓存调用分别保存 dispatch 和 trace；只有相应计算关联到 host launch、flow 和
正确设备 kernel 才能通过路由门禁。PPU 原生 ATen 通常记录
`cuda_runtime/cudaLaunchKernel`，FlagGems 可能记录
`cuda_driver/cuLaunchKernelEx`；两者都必须通过相同的 correlation、`ac2g` flow、窗口和设备
kernel 关联检查。输出在设备上、复制内核或仅编译不能替代证明。
显存来自 worker allocator，性能来自同步主机批计时，二者均不冒充整卡指标或设备计时器。
原生 Asight 指标保留 PPU 的名称、单位和 kernel 归属，不伪装成 Ascend 的 L0/UB 等指标。
`--profiling full` 使用 SDK 的 section Identifier 和 `--page details --csv`，
保存 section 名称、指标单位及原生 `.acurep`；缺少请求 section 或阶段身份不一致保持 partial。
SDK 的 raw CSV 是宽表，不作为带单位长表解析。Asight 重放指标不能替代独立计时结果。

在当前验证站点，`pg-exporter` 会占用 Asight 所需的 PCM 资源。
这是站点监控配置冲突；普通运行命令不会暂停服务，也不会将它自动列为只读白名单。
维护者需另行取得管理员授权和安全采集窗口，采集后恢复监控并记录恢复证据。

失败、超时、精度超差及证据不足均保留，不修改算子、不放宽容差、不缩小输入掩盖失败。
详细状态、机制、待验收项及自定义 case 入口见 [运行与验收说明](../../../docs/thead/operation.md)。
