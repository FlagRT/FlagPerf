# 平头哥 PPU Base — main_memory-capacity

## 实现和范围

实际保留成功分配的 tensor；1 MiB 搜索粒度、按空闲显存约束请求量（BOUND_REQUEST_BY_FREE_MEMORY）、有界分配/OOM trail、释放后 free bytes 与恢复检查。容量不是 free-memory 查询。

## 配置

单卡，需要 `--allow-high-risk-case`；INITSIZE 8192 MiB，POST_TEST_WAIT_SECONDS 为 0，MODE capacity。仅独占授权窗口执行；本 Case 是单次高风险验证，不在五次资格组内。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `main_memory-capacity:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

只统计实际成功持有字节数；OOM 原文保留，非 OOM 错误必须传播，free-memory hint 不作为容量结果。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case main_memory-capacity:PPU --physical-device-ids 7 \
  --nproc-per-node 1 --timeout 300 --allow-high-risk-case \
  --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
