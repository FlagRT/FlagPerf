# Kunlunxin P800 Base — interconnect-h2d

## 实现和范围

预分配主机源和设备目标，循环 Tensor.copy_；完整同步、计时外内容检查。四种 pageable/pinned × blocking/nonblocking 请求独立统计，nonblocking 不代表已证明重叠。

## 配置

512 MiB；四份 case_config.measured-*.yaml。迭代数和 host memory 类型以合并配置为准。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `interconnect-h2d:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

按实际字节量和完整同步墙钟计时输出 GB/s 与 GiB/s；单位分别除以 1e9 和 2³⁰。

2026-10-08 结果：四变体均有五次严格通过组；pinned-nonblocking CV 4.163%，接近门槛。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
