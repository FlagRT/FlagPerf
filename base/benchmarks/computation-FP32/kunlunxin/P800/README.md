# Kunlunxin P800 Base — computation-FP32

## 实现和范围

XPYTORCH torch.mm；FP32 输入/输出。TF32 compatibility flag 关闭不证明内部严格 IEEE 算术。

## 配置

8192³，warmup 100，ITERS 2000。历史 4096³/20,000 次归 Day 4。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `computation-FP32:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

GEMM ops = 2×M×N×K×ITERS，除以 elapsed_seconds 和 1e12 得 TFLOPS/TOPS。

2026-10-08 结果：118.4729 TFLOPS，CV 0.3242%，五次严格通过。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。

正确性包括小矩阵完整 reference 与大矩阵固定行列采样/输出有限性，不能描述成大矩阵逐元素完整 CPU 对照。计时采用 perf_counter_ns 与目标设备完整同步；CUDA Event elapsed 为零的历史限制保留。
