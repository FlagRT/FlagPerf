# Kunlunxin P800 Base — main_memory-capacity

## 实现和范围

实际保留成功分配的 tensor；1 MiB 搜索粒度、有界分配/OOM trail、释放后 free bytes 与恢复检查。容量不是 free-memory 查询，也不是 Toolkit 标称容量 query。

## 配置

单卡，需要 --allow-high-risk-case；POST_TEST_WAIT_SECONDS 为 0。仅独占授权窗口执行。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `main_memory-capacity:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

只统计实际成功持有字节数；OOM 原文保留，非 OOM 错误必须传播，free-memory hint 不作为容量结果。

2026-10-08 结果：98,132 MiB = 102.898860032 GB = 95.83203125 GiB；32 次分配。测量/释放/清理通过，短窗口 monitor partial 原样保留；不套用吞吐五次资格。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。
