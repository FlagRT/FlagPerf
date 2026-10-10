# 昆仑芯 P800 Base — interconnect-h2d

## 实现和范围

预分配主机源和设备目标，循环 Tensor.copy_；完整同步、计时外内容检查。四种 pageable/pinned × blocking/nonblocking 请求独立统计，nonblocking 不代表已证明重叠。

## 配置

512 MiB；四份 case_config.measured-*.yaml。迭代数和 host memory 类型以合并配置为准。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-h2d:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case interconnect-h2d:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
