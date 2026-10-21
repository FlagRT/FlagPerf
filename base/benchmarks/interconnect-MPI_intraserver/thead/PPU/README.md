# 平头哥 PPU Base — interconnect-MPI_intraserver

## 实现和范围

厂商 NCCL all_reduce SUM，后端 `cpu:gloo,cuda:nccl`（FlagCX 是 P800 路径）；provider 接受 2–16 rank，实际设备由 UUID join 绑定。非零输入用于计时外正确性，计时循环用零和系数避免多 rank 溢出。单 iter 用时对 rank 数不是单调关系，8 rank 及以上的资格须在各自窗口内重新校准 ITERS；本轮实测资格完成于 8 rank，16 rank 复评见验证说明。

## 配置

消息 256 MiB（Melements 64 float32），warmup 10，ITERS 1700（8 rank 资格计时窗）。需 internal bridge。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-MPI_intraserver:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

algbw = message_bytes×ITERS/elapsed；busbw = algbw×2×(world_size−1)/world_size。逐 run 取最慢 rank，不求和，不加额外展示系数。全部同机通信数字为该机实测路径口径；其他机型或互联配置须重新测量。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-MPI_intraserver:PPU --physical-device-ids 7,8 \
  --nproc-per-node 2 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
