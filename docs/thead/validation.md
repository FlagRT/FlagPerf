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
三家厂商的文档链接与入口约定，含 `docs/thead`。

缺少依赖、跳过测试或仅通过 dry-run 时，不得记录为完整回归通过。
正式测量还需 Git 工作副本记录源码身份。

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
   证据按 `P2P=disabled` 如实标注，不冒充直连带宽。
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
