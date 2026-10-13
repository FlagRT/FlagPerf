# 昆仑芯 P800 Base — computation-BF16

## 实现和范围

XPYTORCH torch.mm；BF16 输入/输出，reference 使用量化输入的 CPU FP64 值。锁定栈约 FP32 吞吐；历史算子对照支持该路径限制，不外推所有硬件或未来 runtime。

## 配置

8192³，warmup 100，ITERS 5000。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `computation-BF16:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

GEMM ops = 2×M×N×K×ITERS，除以 elapsed_seconds 和 1e12 得 TFLOPS/TOPS。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。

正确性包括小矩阵完整 reference 与大矩阵固定行列采样/输出有限性，不能描述成大矩阵逐元素完整 CPU 对照。计时采用 perf_counter_ns 与目标设备完整同步；CUDA Event elapsed 为零的历史限制保留。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-BF16:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
