# 平头哥 PPU 适配指南

本适配提供 FlagPerf Base 的 PyTorch Benchmark。所有命令从 FlagPerf 仓库根目录执行；
完整运行步骤见 [Base 使用指南](../../base/README.md)。

## 入口和支持范围

| 能力 | 入口 | 当前边界 |
|---|---|---|
| Base Benchmark | `python3 base/run.py benchmark run ...` | 21 个 case×变体条目：18 项可执行，3 项启动前跳过；每次运行一个 `CASE:PPU` |
| 身份预检查 | `python3 base/run.py benchmark preflight ...` | 单卡身份、微小张量回读、监控和清理检查，不给出性能指标 |
| 报告重建 | `python3 base/run.py report ...` | 读取已有证据，重建 Markdown/SVG，不重新测量硬件 |
| 重复性检查 | `python3 base/qualification.py ...` | 五次独立 run 的严格资格检查，与单 run passed 分开 |

厂商名为 `thead`，芯片名为 `PPU`。Benchmark 中计算、传输和内存为单 rank；
同机 AllReduce 接受 2–16 rank（本轮实测资格完成于 8 rank，16 rank 见验证说明），
P2P 恰好 2 rank。FP8（算子路由失败）和两个跨机 Case（单机执行器与目标机未上线）
在当前锁定栈下提前跳过，不产生性能值。该范围不代表芯片在其他软件栈下没有相应能力。
本次适配不扩大 Operation、Training、Inference 或 Generate 的验证范围。

## 环境准备

### 宿主

使用 Linux、Python 3.11 或更新版本、Git、Docker、可用的平头哥驱动和 `ppu-smi`。
宿主仅需轻量规划依赖；PyTorch 和厂商库在锁定容器中运行：

```bash
python3 -m pip install -r base/requirements-host.txt
python3 base/run.py --help
```

宿主配置为 [thead_ppu_torch210.yaml](../../base/configs/thead_ppu_torch210.yaml)，
内容采用 JSON 语法。移植到另一宿主时复制到仓库外，通过 `--config` 传入：

| 字段 | 用途 |
|---|---|
| `vendor` / `chip` / `runtime_profile` | `thead` / `PPU` / `torch210_ppu_candidate` |
| `image` | 必须与 runtime 清单中的镜像身份一致，仅修改 tag 不构成验证 |
| `expected_device_ids` | 宿主完整物理 inventory；随附配置为 0–15，本轮选卡另由 CLI 指定 |
| `result_root` | 相对于 `base/` 的结果根或绝对路径；CLI 可另指定结果位置 |
| `required_devices` | 公共控制节点 `/dev/alixpu`；计算节点由实时身份检查确定 |

### 锁定镜像

| 项目 | 值 |
|---|---|
| 镜像 | `flagtree-ppu-3.6-dev:202608` |
| image ID | `sha256:be1606e81fb24ca3da57a6d4a990280369b89146c2999e88eb8cd07c661dcc79` |
| 平台 | `linux/amd64` |
| 容器 Python / PyTorch | `3.12.3` / `2.10.0+cu130`；PPU_SDK 2.1.0 CUDA 兼容层（`MODE=CUDA`） |
| 运行状态 | `validated:false`、`release_stage:candidate`；每轮 run 须显式 `--allow-candidate-runtime` |

当前镜像保持 candidate 状态：全部既有实测在该状态下完成，16 rank 资格与 validated
复评条件见 [验证说明](validation.md)。workload 通过 `torch.cuda` API 使用设备
（PPU_SDK 为 CUDA 兼容层）；不要求 FlagGems、FlagCX 或 Torch-FL 插件。

本仓库不分发该镜像，也没有验证匿名 registry pull。使用者需要先取得相同镜像及所需
授权，核对实际 image ID 与清单一致。更换镜像、驱动或依赖后需重新验证对应范围，
不能复制旧资格结论。

### 设备、权限和通信网络

PPU 使用 `--physical-device-ids` 显式选卡。SMI 物理编号、`/dev/alixpu_ppuN`
（minor 为 N+1）、容器内逻辑编号和 rank 不是同一个编号；执行器校验 BDF/UUID/minor，
并在容器内重新绑定。设备身份以 UUID 为主键（`resource_key=thead/<uuid>`），
UUID join（torch `get_device_properties(i).uuid` 与 SMI UUID）已在 Day-1 探针验证。

容器以非特权方式运行：丢弃全部 capabilities、只读根文件系统、pids 上限、
结果目录由容器实际身份写入（必要时 `--group-add` 宿主组）。
`--privilege-command` 只为 Docker 等宿主命令增加权限，不改变容器特权状态。

Base 的容器策略使用已有的 `thead-internal` 内部 bridge，包括单卡 Benchmark。
先确认网络存在且为 internal；不存在时，由宿主管理者按本机网络规划创建：

```bash
docker network inspect thead-internal
docker network create --internal --subnet 172.31.253.0/24 thead-internal
```

创建命令仅用于尚未创建该网络的宿主；如需固定子网，应先确认地址段无冲突。
执行器不自动创建网络，单次清理也不删除这项共享设施。

## 最小静态检查

下面的选卡仅为示例，不表示设备可用。dry-run 不查询设备、不启动容器或采集监控。
同机通信 case 至少需要 2 rank（P2P 恰好 2）；容量 case 是高风险单次验证，
真实执行需要 `--allow-high-risk-case`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case computation-FP32:PPU --physical-device-ids 7 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run

python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-MPI_intraserver:PPU --physical-device-ids 7,8 \
  --nproc-per-node 2 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行还需符合 [Base 指南](../../base/README.md) 中的资源预约、权限及输出目录条件：
带时区的资源窗口、全新结果目录、timeout 30–600 秒，且资源截止时间覆盖执行预算
和至少 30 秒清理。

## 身份预检查与故障处理

`benchmark preflight` 的默认 `--probe-mode identity` 仅检查单卡身份、微小张量回读和
观察窗口。`--probe-mode timeout-check` 会故意触发宿主 watchdog，实验状态应保持失败；
清理成功只能证明演练中的清理步骤，不应改写为性能通过。

```bash
python3 base/run.py benchmark preflight \
  --config base/configs/thead_ppu_torch210.yaml \
  --physical-device-ids 7 --probe-mode identity \
  --timeout 120 --allow-candidate-runtime --dry-run
```

| 现象 | 检查与处理 |
|---|---|
| 参数或 runtime 不匹配 | 检查厂商配置、`:PPU` selector 和镜像锁；candidate 镜像必须显式授权 |
| 设备占用或句柄拒绝 | 重新选择已预约空闲设备；占用判定含进程表与已分配显存底线 |
| `already leased` | 等待持有者释放；不删除共享锁或修改其权限 |
| 容器无法写入结果 | 检查宿主创建目录的所有者、权限与容器身份，不能只给 Docker 增加 sudo 前缀 |
| `Failed to open /dev/alixpu` | 设备映射缺少公共控制节点 `/dev/alixpu`；逐卡节点 `alixpu_ppuN` 由执行器映射 |
| 通信 case 报 rank 数不符 | AllReduce 至少 2 rank（上限 16），P2P 恰好 2 rank |
| 有数值但结果 partial | 检查真实计时窗内的监控样本、postflight 和 cleanup；计时窗不足 15 秒的 run 不构成资格形态 |
| 超时或 cleanup 不确定 | 保留错误，核实本轮精确 CID、设备空闲状态和租约；不 reset 设备或重试掩盖失败 |

租约是遵守相同协议的 FlagPerf 进程之间的协作锁，不能替代团队资源预约。
健康和性能以本轮证据为准，不沿用某次历史选卡结论。

验证方法、历史结果范围和已知限制见 [验证与维护](validation.md)。
