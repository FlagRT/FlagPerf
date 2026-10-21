# 单机 P2P 协议

正式协议名为 `p2p-single-node-v1`。目录属于
`base/benchmarks/interconnect-P2P_intraserver/thead/PPU/`，原 Case 的 `main.py` 保持不变。
公共 distributed backend 为 `cpu:gloo,cuda:nccl`（PPU 上 vendor NCCL，staged 数据面）。

## 工作负载与范围

| 项目 | 固定约定 |
|---|---|
| 节点与 rank | 单节点、恰好两 rank |
| workload | FP32、单向 blocking `send/recv`、20 次 warmup |
| 原结果公式 | `one-way payload × iterations / elapsed`（从不加倍）；这是单向 payload 带宽口径 |
| 拓扑与方向 | 同卡两芯：Device 14→15；跨卡 staged：Device 7→8（两对代表性设备的 BDF 见计划 `representative_pairs` 与 `matrix[].bus_ids`） |
| 消息大小 | 4、16、64、256 MiB |
| 迭代数 | 由 64 GiB 单点校准钉扎，固定写入各配置，不随实测速度自动扩展 |
| 规模曲线 | 每对拓扑×大小三次 monitor **on**，共 24 run |
| 监控 A/B | 同卡 64 MiB，off/on/on/off/off/on，共 6 run；off 腿按设计为 partial（失真探针） |
| 单 run 测量窗 | 最短 15 秒（共享契约地板）、目标 60 秒；timeout 300 秒；同卡 4 MiB cell 因契约 `ITERS ≤ 200000` 为 ~25 秒 |
| 判定 | 单元 CV ≤5%，监控中位数差绝对值 ≤5%，完整双 rank、postflight 通过 |

与 ascend 同名协议的**一处已记录偏离**：本 harness 的判定门
（`executors/bounded_benchmark.py`）把无监控的 qualification run 判为 partial，
因此规模曲线在 monitor on 下采集（`size_curve_monitor_note` 记录在计划文件）；
ascend 曲线当年以特权脚本在 monitor off 下采集，绕过了同一判定门。

固定配置由校准选定：每 cell 单向 64 GiB、monitor off、1 次重复，实测 per-iteration
同卡 0.1264/0.4013/1.5126/6.3522 ms、跨卡 0.8764/3.4952/13.9561/55.5549 ms（4/16/64/256 MiB）。
首次运行可能受初始化与宿主状态影响；全矩阵约 28 分钟纯测量。
反向通信、其它设备对、跨机和长稳属于这份资格范围之外，不用单机结果推断它们已通过。
全设备对（16×15 有向 pair）带宽/时延扫点的工具缺口与补齐条件登记在
[验证说明](validation.md)已知限制第 8 条。
本机 `P2P=disabled`，两条路径均为主机转运（staged）口径，数字不描述直连互联。
新主机的编号和拓扑必须重新确认；更换设备范围、工作量或阈值需要新计划和验证记录。

## 入口与命名

| 用途 | 文件 |
|---|---|
| 正式运行配置 | [thead_ppu_torch210.yaml](../../base/configs/thead_ppu_torch210.yaml) |
| 单次功能检查 | [case_config.smoke.yaml](../../base/benchmarks/interconnect-P2P_intraserver/thead/PPU/case_config.smoke.yaml) |
| 扫点与校准配置（12+4） | `case_config.p2p-{intracard,intercard}-{4m,16m,64m,256m}.yaml`、`case_config.p2p-calibration-<size>.yaml` |
| 配置 sha256 名单 | [runtime_requirements.json](../../base/benchmarks/interconnect-P2P_intraserver/thead/PPU/runtime_requirements.json)（`requires_case_override`，15 条） |
| 校准计划 | [p2p-calibration-plan.json](../../base/benchmarks/interconnect-P2P_intraserver/thead/PPU/p2p-calibration-plan.json) |
| 正式计划 | [p2p-qualification-plan.json](../../base/benchmarks/interconnect-P2P_intraserver/thead/PPU/p2p-qualification-plan.json) |
| 资格发布记录 | [p2p-qualification.json](../../base/vendors/thead/torch210_ppu_candidate/p2p-qualification.json) |
| 校准记录 | [p2p-calibration-record.json](../../base/vendors/thead/torch210_ppu_candidate/p2p-calibration-record.json) |
| 证据清单 | [p2p-evidence-manifest.json](../../base/vendors/thead/torch210_ppu_candidate/p2p-evidence-manifest.json) |
| 一致性守卫 | `base/tests/test_thead_p2p_scan.py`（名单↔目录、契约界、计划↔配置、A/B 序、窗口地板） |

正式扫点必须显式传 `--case-config` 且其 resolved-path sha256 在名单内（executors/benchmark.py
既有 allowlist 机制强制）；直接运行默认 `case_config.yaml` 会被拒绝。
规模曲线与 A/B 共 30 个 run 由站点编排器按计划顺序执行；每个 run 是标准
`base/run.py benchmark run` 调用，带设备预约引用。失败或不完整的 run 目录按原名保留、
不覆写；协议修订前的 monitor-off 首链 17 个 run 目录同样保留为偏离证据。

```bash
# 只输出计划，不触碰硬件。
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_intraserver:PPU \
  --case-config base/benchmarks/interconnect-P2P_intraserver/thead/PPU/case_config.p2p-scan-intracard-64m.yaml \
  --physical-device-ids 14,15 --nproc-per-node 2 --monitor on \
  --allow-candidate-runtime --dry-run

# 已完成设备预约后执行单个 cell（乘 3 次重复；8 cell × 3 + A/B 6 = 30 run）。
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_intraserver:PPU \
  --case-config base/benchmarks/interconnect-P2P_intraserver/thead/PPU/case_config.p2p-scan-intracard-64m.yaml \
  --physical-device-ids 14,15 --nproc-per-node 2 --monitor on --timeout 300 \
  --allow-candidate-runtime --result-dir <results-root>/dayN-p2pscan2-intracard-64m-01 \
  --reservation-end <UTC> --reservation-reference <ref>
```

## 证据和公开分发

[资格发布记录](../../base/vendors/thead/torch210_ppu_candidate/p2p-qualification.json)绑定
原计划与镜像 ID 的 SHA-256、30 个完成 run 的证据清单（另保留 17 个 monitor-off 首链）、
声明范围（单节点两 rank、钉死方向、staged 路径、4 档消息）、验收观测
（最大 cell CV 0.4499%，监控中位数差 0.0739%，fallback 计数为 staged 即声明路径故为 0）
与决策（formal execution eligible；production 晋级仍由镜像 manifest 的
`validated:false` 把守，本记录不自动晋级 runtime）。
[校准记录](../../base/vendors/thead/torch210_ppu_candidate/p2p-calibration-record.json)
是维护者 attestation：钉扎迭代数来源、8 个校准点的观测值与 summary 哈希，
`formal_baseline_eligible:false`——校准不是性能基线。

原始 run 目录在站点结果根，不随本仓分发；清单内每 run 的 `summary_sha256` 允许在
独立克隆中静态核对记录与计划的一致性，执行仍需实时 preflight、拓扑与资源回收检查。
published_contract 中的矩阵、监控 A/B 与验收阈值为计划原文投影，记录与计划的漂移
由 `test_thead_p2p_scan.py` fail closed。
