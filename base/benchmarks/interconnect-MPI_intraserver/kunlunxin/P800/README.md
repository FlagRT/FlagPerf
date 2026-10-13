# 昆仑芯 P800 Base — interconnect-MPI_intraserver

## 实现和范围

FlagCX/BKCL all_reduce SUM，后端 cpu:gloo,cuda:flagcx；provider 接受 2–8 rank，实际设备由 UUID join 绑定。非零输入用于计时外正确性，计时循环用零和系数避免多 rank 溢出。

## 配置

默认 YAML 为 4 MiB、2 iterations 的 smoke。性能组使用独立 override，真实计时窗口须满足资格门槛；需 internal bridge。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-MPI_intraserver:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

algbw = message_bytes×ITERS/elapsed；busbw = algbw×2×(world_size−1)/world_size。逐 run 取最慢 rank，不求和，不加额外 NVIDIA 展示系数。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case interconnect-MPI_intraserver:P800 --physical-device-ids 4,7 \
  --nproc-per-node 2 --timeout 300 --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
