# 平头哥 PPU Base — interconnect-P2P_intraserver

## 实现和范围

厂商 NCCL send/recv，恰好两个 rank，方向 rank 0→rank 1。本机型实测没有卡间直连 P2P 路径，NCCL 经主机 staged 转发：对外数字描述 staged 路径，运行证据保留 `P2P=disabled` 标注，不作为直连带宽呈现。rank 不等于物理卡号。正确性检查 payload/序列，外层超时和清理必须完整。

## 配置

消息 256 MiB（Melements 64 float32），warmup 10，ITERS 340，恰好 2 rank。需 internal bridge。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-P2P_intraserver:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 扫点协议（p2p-single-node-v1）

本机为单卡双芯（每张物理卡两个 PPU 芯片），同卡两芯与跨卡两芯的通信带宽不同。对标 ascend 的 p2p-single-node 扫点协议，本目录维护同一协议形状的 thead 版本：

- **代表对**：`intracard-die-pair`（同卡两芯，实测对 14,15）与 `intercard-staged-pair`（跨卡 staged，实测对 7,8）；方向钉死 rank0→rank1（`14-to-15-only` / `7-to-8-only`），不声明反向。
- **校准**：4 个规格（4/16/64/256 MiB），每点 64 GiB 单向 payload，monitor off、单次（`case_config.p2p-calibration-*.yaml`），只用于钉扫点 ITERS，不入正式账目。
- **扫点矩阵**：8 cell（2 对 × 4 规格），每 cell 3 次重复（`case_config.p2p-scan-*.yaml`，ITERS 由校准钉扎，目标 ~60 s 窗口，monitor on——本 harness 将无监控的 qualification run 判为 partial，与 ascend monitor-off 曲线的这一偏离记录在计划文件），验收 CV ≤ 5%；monitor A/B（off,on,on,off,off,on）钉在 intracard-64m，中位数偏差 ≤ 5%，off 腿按设计为 partial（失真探针，不作资格声明）。共享契约 ITERS 上限 200000，故 intracard-4m cell 窗口约 25 s（契约下限 15 s）。
- **配置钉扎**：本 case 开启 `requires_case_override`，每次运行必须显式传 `--case-config` 且其 sha256 在 `runtime_requirements.json` 名单内（正式、烟囱、校准、扫点共 15 条）。
- cell 明细、中位数、CV 与 monitor A/B 记录见 `p2p-qualification-plan.json`（校准观测见 `p2p-calibration-plan.json`）；两份计划与名单的一致性由 `base/tests/test_thead_p2p_scan.py` 守护。

扫点运行示例（cell 与设备对的对应关系以计划文件为准；卡号仅为示例）：

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_intraserver:PPU \
  --case-config base/benchmarks/interconnect-P2P_intraserver/thead/PPU/case_config.p2p-scan-intercard-4m.yaml \
  --physical-device-ids 7,8 --nproc-per-node 2 --timeout 300 \
  --monitor off --allow-candidate-runtime --dry-run
```

## 指标和验收

单向带宽 = payload_bytes×ITERS/elapsed；不乘 2。每 run 取两个有效 rank 的较小带宽，再对五次取中位数。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_intraserver:PPU --physical-device-ids 7,8 \
  --nproc-per-node 2 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
