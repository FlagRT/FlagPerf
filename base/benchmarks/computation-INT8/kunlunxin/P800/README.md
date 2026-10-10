# 昆仑芯 P800 Base — computation-INT8

## 实现和范围

设备算子 xtorch_ops.gemm_I8_I8_bf16_nt；INT8 输入、缩放、BF16 输出。对外结果为 (A@B)*SCALE_A*SCALE_B 后转 BF16；driver 传入 127×scale 配合内核 /127 约定。不是 Toolkit 的 XBLAS fc_fusion。历史 torch._int_mm 主机回退数字已撤回。

## 配置

8192³，warmup 10，ITERS 8000；TOPS 将乘加计作两次操作。

配置合并顺序为 generic < vendor < P800 < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `computation-INT8:P800` 与 P800 host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

GEMM ops = 2×M×N×K×ITERS，除以 elapsed_seconds 和 1e12 得 TFLOPS/TOPS。

历史实测与当前限制见 [公开验证说明](../../../../../docs/kunlunxin/validation.md#历史实测范围)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。候选 manifest 不因本 case 通过自动晋级。

正确性包括小矩阵完整 reference 与大矩阵固定行列采样/输出有限性，不能描述成大矩阵逐元素完整 CPU 对照。计时采用 perf_counter_ns 与目标设备完整同步；CUDA Event elapsed 为零的历史限制保留。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-INT8:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
