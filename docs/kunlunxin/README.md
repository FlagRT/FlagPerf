# 昆仑芯 P800 适配指南

本适配提供 FlagPerf Base 的 PyTorch Benchmark、独立原生 Toolkit，以及 Operation 算子评测适配。
所有命令从 FlagPerf 仓库根目录执行。Base 与 Operation 分别声明运行栈验证范围，不能互相沿用资格。

## 按任务选择入口

- **设备基础能力**：[Base 使用指南](../../base/README.md)；本页下文说明 Base/Toolkit 的配置、权限与运行栈。
- **单算子正确性、路由与性能**：[P800 Operation 最小运行](../../operation/vendors/kunlunxin/README.md)。
  必须显式选择 `--vendor kunlunxin --runtime-profile p800-xpytorch29`，不能套用 Base 命令。
- **Operation 机制、结果目录及复放**：[机制与维护](operation-design.md)。
- **Operation 实测范围与限制**：[验证与已知问题](operation-validation.md)。Operation 保持 `validated:false`；
  Docker/local 全矩阵及受干扰范围补测已完成；六组硬件计数器仍未完成，间歇 trace 缺失与精确比较统计缺口暂时保留。
  各批次资格分开记录，不能从覆盖数或 Base 发布状态推断算子已完整验收。


## 入口和支持范围

| 能力 | 入口 | 当前边界 |
|---|---|---|
| Base Benchmark | `python3 base/run.py benchmark run ...` | 15 项已分类：10 项可执行，5 项启动前跳过；每次运行一个 `CASE:P800` |
| P800 Toolkit | `python3 base/run.py toolkit run ...` | 12 项独立 C++ microbenchmark，通过 XBLAS/XRE 执行；可重复指定 `--case` |
| 身份预检查 | `python3 base/run.py benchmark preflight ...` | 单卡身份、微小张量回读、监控和清理检查，不给出性能指标 |
| 报告重建 | `python3 base/run.py report ...` | 读取已有证据，重建 Markdown/SVG，不重新测量硬件 |
| 重复性检查 | `python3 base/qualification.py ...` | Base 五次独立 run 的严格资格检查，与 Toolkit 诊断状态分开 |

厂商名为 `kunlunxin`，芯片名为 `P800`。Benchmark 中计算、传输和内存为单 rank；
同机 AllReduce 接受 2–8 rank，P2P 恰好 2 rank。FP64、FP8、TF32 和两个跨机 Case
在当前锁定栈下提前跳过，不产生性能值。该范围不代表芯片在其他软件栈下没有相应能力。

Toolkit 覆盖 FP32/FP16/BF16/INT8 算力、H2D/D2H 带宽及各自时延、设备内存带宽、
容量查询和同机 P2P 带宽/时延；不提供 AllReduce 或跨机测试。
本页 Base/Toolkit 适配不把结论扩大到 Operation、Training、Inference 或 Generate。
Operation 是独立的算子评估入口，当前运行栈保持 `validated:false`；
[最小运行](../../operation/vendors/kunlunxin/README.md)、[执行机制](operation-design.md)和[验证/失败/责任边界](operation-validation.md)分别维护。

## 环境准备

### 宿主

使用 Linux、Python 3.11 或更新版本、Git、Docker、可用的昆仑芯驱动、`xpu-smi` 和 `fuser`。
宿主仅需轻量规划依赖；PyTorch 和厂商库在锁定容器中运行：

```bash
python3 -m pip install -r base/requirements-host.txt
python3 base/run.py --help
```

宿主配置为 [kunlunxin_p800_xpytorch29.yaml](../../base/configs/kunlunxin_p800_xpytorch29.yaml)，
内容采用 JSON 语法。移植到另一宿主时复制到仓库外，通过 `--config` 传入：

| 字段 | 用途 |
|---|---|
| `vendor` / `chip` / `runtime_profile` | `kunlunxin` / `P800` / `xpytorch_2.9_p800_candidate` |
| `image` | 必须与 runtime 清单中的镜像身份一致，仅修改 tag 不构成验证 |
| `expected_device_ids` | 宿主完整物理 inventory；随附配置为 0–7，本轮选卡另由 CLI 指定 |
| `result_root` | 相对于 `base/` 的结果根或绝对路径；CLI 可另指定结果位置 |
| `required_devices` | 公共控制节点 `/dev/xpuctrl`；计算节点由实时身份检查确定 |
| `runtime_environment` | 锁定 `USE_FLAGGEMS=0`，使用原生 XPYTORCH 路径 |

### 锁定镜像

| 项目 | 值 |
|---|---|
| 镜像 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` |
| image ID | `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975` |
| 平台 | `linux/amd64` |
| 容器 Python / PyTorch | `3.10.18` / `2.9.0+cu129`；XPYTORCH、`torch_xmlir` |
| 运行状态 | `validated:true`、`release_stage:validated`、`validation_status:partial`，仅覆盖声明范围 |

profile 的 `_candidate` 后缀为兼容旧配置保留，不表示当前发布状态。
当前镜像无需 `--allow-candidate-runtime`；既有实测、限制及范围见 [验证说明](validation.md)。

依赖版本和运行路径以 [runtime 目录](../../base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/README.md)
中的锁文件为准。宿主 Driver、SMI Runtime、头文件版本及镜像 XRE 元数据来自不同层，不能混作同一个版本号。
锁定环境存在 XRE/XHPC 依赖元数据差异；既有实测限定于该组合，不等于厂商兼容性认证。

本仓库不分发该镜像，也没有验证匿名 registry pull。
使用者需要先取得相同镜像及所需授权，核对实际 image ID 和清单中的 RepoDigests。
代码开源不代表镜像已经公开分发。更换镜像、驱动或依赖后需重新验证对应范围，不能复制旧资格结论。

### 设备、权限和通信网络

P800 使用 `--physical-device-ids` 显式选卡。SMI 物理编号、`/dev/xpuN` 的 minor、
容器内逻辑编号和 rank 不是同一个编号；执行器校验 PCI/UUID/minor，并在容器内重新绑定。
当前没有固定物理卡排除清单，所选设备均经过身份、占用、健康及租约检查。

宿主执行者必须能调用 Docker、检查设备句柄，并读写本轮结果及共享锁。
`--privilege-command 'sudo -n'` 只为 Docker/fuser 等宿主命令增加权限，不会改变 Python
进程本身的文件权限。它也不是 `--allow-privileged-root`：P800 容器不使用 privileged 模式。
Toolkit 容器丢弃全部 capabilities，结果目录必须允许容器实际身份写入；必要时由有权限的
宿主账户启动整个 runner。root 启动是部署方式之一，不是所有环境的固定要求。

Base 的容器策略使用已有的 `flagperf-p800-internal` 内部 bridge，包括单卡 Benchmark。
先确认网络存在且为 internal；不存在时，由宿主管理者按本机网络规划创建：

```bash
docker network inspect flagperf-p800-internal
docker network create --internal flagperf-p800-internal
```

创建命令仅用于尚未创建该网络的宿主；如需固定子网，应先确认地址段无冲突。
执行器不自动创建网络，单次清理也不删除这项共享设施。Toolkit 使用 `--network=none`，不需要该 bridge。

## 最小静态检查

下面的 4、7 仅为示例，不表示设备可用。dry-run 不查询设备、不启动容器或采集监控。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP32:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --dry-run

python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4,7 --dry-run
```

真实执行还需符合 [Base 指南](../../base/README.md) 中的资源预约、权限及输出目录条件。
Toolkit 的原生 API、载荷、计时、参数和证据格式见 [Toolkit 使用说明](../../base/toolkits/_common/kunlunxin/P800/README.md)。

## 身份预检查与故障处理

`benchmark preflight` 的默认 `--probe-mode identity` 仅检查单卡身份、微小张量回读和观察窗口。
真实运行需要带时区的资源窗口和全新结果目录；当前锁定镜像无需 candidate opt-in；timeout 为 30–180 秒。
与性能测试一样，资源截止时间需要覆盖执行预算和至少 30 秒清理，并在启动前重新检查。

```bash
python3 base/run.py benchmark preflight \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4 --probe-mode identity \
  --timeout 120 --dry-run
```

故障演练 `--probe-mode timeout-check` 会故意触发宿主 watchdog，实验状态应保持失败；
清理成功只能证明演练中的清理步骤，不应改写为性能通过。

| 现象 | 检查与处理 |
|---|---|
| 参数或 runtime 不匹配 | 检查厂商配置、`:P800` selector和镜像锁；未验证候选版本另需 candidate 授权 |
| 设备占用或句柄拒绝 | 重新选择已预约空闲设备；只读监控例外必须显式配置并通过进程身份校验 |
| `already leased` | 等待持有者释放；不删除共享锁或修改其权限 |
| 容器无法写入结果 | 检查宿主创建目录的所有者、权限与容器身份，不能只给 Docker 增加 sudo 前缀 |
| BKCL 缺少可用接口 | 检查 internal bridge；不自动切换 host network |
| 有数值但结果 partial | 检查真实计时窗内的监控样本、postflight 和 cleanup |
| 超时或 cleanup 不确定 | 保留错误，核实本轮精确 CID、设备空闲状态和租约；不 reset 设备或重试掩盖失败 |

租约是遵守相同协议的 FlagPerf 进程之间的协作锁，不能替代团队资源预约。
健康和性能以本轮证据为准，不沿用某次历史选卡结论。

验证方法、历史结果范围和已知限制见 [验证与维护](validation.md)。
