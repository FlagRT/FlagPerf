# Kunlunxin P800 Base — computation-BF16

## 实现和范围

XPYTORCH torch.mm；BF16 输入/输出，reference 使用量化输入的 CPU FP64 值。锁定栈约 FP32 吞吐；历史算子对照支持该路径限制，不外推所有硬件或未来 runtime。

## 配置

8192³，warmup 100，ITERS 5000。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `computation-BF16:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../docs/p800-base-handoff.md)。

## 指标和验收

GEMM ops = 2×M×N×K×ITERS，除以 elapsed_seconds 和 1e12 得 TFLOPS/TOPS。

2026-10-08 结果：118.9696 TFLOPS，CV 0.0010%，五次严格通过。 这些数字仅属于固定镜像、设备集合与配置，不能当作所有设备/尺寸的保证。详情和逐组状态见 [验收复核](../../../../docs/p800-base-acceptance-20261008.md)。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。

正确性包括小矩阵完整 reference 与大矩阵固定行列采样/输出有限性，不能描述成大矩阵逐元素完整 CPU 对照。计时采用 perf_counter_ns 与目标设备完整同步；CUDA Event elapsed 为零的历史限制保留。
