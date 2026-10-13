# 昆仑芯 P800 Base — interconnect-d2h

## 实现和范围

预分配设备源和主机目标，循环 Tensor.copy_；完整同步、计时外内容检查。Base 没有 Toolkit 的 1 MiB XRE 分块实现。

## 配置

512 MiB；四份 case_config.measured-*.yaml。pageable/nonblocking 为 ITERS 500、MODE measured，实际有短于 15 秒样本。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-d2h:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case interconnect-d2h:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
