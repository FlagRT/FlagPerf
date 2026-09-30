# BKCL 传输层甄别日志（XPULink vs PCIe 回退）

日期：2026-09-24　主机：P800 服务器（klx）　镜像：sha256:cd53efa40eb7（锁定 flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608）

## 背景

§5.4/§5.5 实测发现 XPULink 直连对（smi4=84:00.0 + smi7=BB:00.0）AllReduce 带宽
（约 28.9 GB/s）反而低于非 XPULink 对照对（smi3=2E:00.0 + smi4=84:00.0，约 37.9 GB/s）。
本目录用 `BKCL_DEBUG=1` 的 BKCL 初始化日志甄别两对卡各自实际使用的物理传输。

方法：两进程 torch.distributed（backend `cpu:gloo,cuda:flagcx`，即 FlagCX→BKCL 栈），
256 MiB float32 AllReduce 循环 5 s 计时；对照对加 `BKCL_DEBUG=1` 采集 BKCL 初始化与
通道建立日志。FlagCX 同构场景下所有集合通信均委托 BKCL 执行
（`xccl_adaptor.cc` 直接把 FlagCX bootstrap 句柄传给 `bkcl_init_rank`），
因此 BKCL 日志即底层传输的判据。

## 甄别结果

| 对 | XL ring（kl3_init.cpp:1926） | PCIe ring（kl3_init.cpp:1987） | 通道数 | 实测 AllReduce |
|---|---|---|---|---|
| smi4 + smi7（`topo -m` 标 XL） | **6** | 0 | 6 × 4 MiB ring buf | 28.85 GB/s |
| smi3 + smi4（`topo -m` 标 SYS） | **0** | **12** | 12 × 4 MiB ring buf | 37.88 GB/s |

两对均运行同一算法 `xlink_ring_all_reduce_single_node`（kl3_all_reduce.cpp:46；
"xlink_ring" 为 KL3 单节点 ring 算法名，不区分物理传输）。
`log2phy` 映射：dbg1（4,7）设备 0→物理 4、1→物理 7；dbg134（3,4）设备 0→物理 3、1→物理 4。

## 对照实验

| 实验 | 环境变量 | 结果 |
|---|---|---|
| cr1-47（4,7） | `BKCL_CLUSTERS_PER_RING=1` | 仍 6 rings，28.21 GB/s（无变化） |
| mc24-47（4,7） | `BKCL_MAX_CLUSTERS=24`（日志确认 set to 24） | 仍 6 rings，28.21 GB/s（无变化） |
| fx-34（3,4） | `BKCL_P2P_FORCE_XLINK=1` | 仍 0 XL rings → 回退 12 PCIe rings，37.77 GB/s |

结论：
1. XPULink **已被正确启用**——4↔7 对 BKCL 探测到 6 条 XL 硬件 ring 并实际使用。
2. 3↔4 对即使强制 XL 探测也是 0 ring，与 `xpu-smi topo -m` 的 SYS 分类一致：
   两卡之间硬件上不存在 XPULink 链路，PCIe（BAR2 P2P）回退是唯一路径。
3. "XL 对慢于 PCIe 对"是真实的物理层现象，非配置错误：每 ring 带宽 XL ≈ 4.8 GB/s
   反而高于 PCIe ≈ 3.2 GB/s，但 XL 仅 6 条 ring（PCIe 12 条），聚合带宽被 ring 数限制。
4. XL ring 数不受 `BKCL_CLUSTERS_PER_RING` / `BKCL_MAX_CLUSTERS` 影响，
   判定为硬件资源上限。→ 是否符合设计预期，建议向厂商确认（见 review.md §10 第 8 条）。

## 文件

- `dbg1-r{0,1}.log` — (4,7) 基线 + `BKCL_DEBUG=1`（6 XL rings）
- `dbg134-r{0,1}.log` — (3,4) 基线 + `BKCL_DEBUG=1`（0 XL / 12 PCIe rings）
- `cr1-47-r0.log` — (4,7) + `BKCL_CLUSTERS_PER_RING=1`
- `mc24-47-r0.log` — (4,7) + `BKCL_MAX_CLUSTERS=24`
- `fx-34-r{0,1}.log` — (3,4) + `BKCL_P2P_FORCE_XLINK=1`

## 附带发现的厂商问题

`BKCL_DEBUG=INFO`（或 `TRACE`）会使首次 AllReduce 崩溃（Python 侧
`ValueError: invalid literal for int() ... stoi`）——BKCL_DEBUG 期望数值，
非数值直接 `std::stoi` 未捕获异常。本文所有日志均使用 `BKCL_DEBUG=1`。
