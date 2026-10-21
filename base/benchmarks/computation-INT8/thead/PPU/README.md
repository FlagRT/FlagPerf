# 平头哥 PPU Base — computation-INT8

## 实现和范围

torch._int_mm；INT8 输入、int32 累加输出。PPU 调优内核只在两个操作数 K 维连续时启用，driver 在计时外一次性把 (K, N) 操作数转为 K-major 布局；行主序回退路径实测停留在 naive 内核量级，相关证据保留在受控结果根，不作为对外数字。TOPS 将乘加计作两次操作。

## 配置

8192³，warmup 2，ITERS 4400；ATOL/RTOL 为 0——int8 操作数的 int32 累加是精确的。其他矩阵规模需单独验证。

配置合并顺序为 generic < vendor < PPU < 显式 override；每次运行保存配置和源码哈希。选择 case 时使用 `computation-INT8:PPU` 与 PPU host profile。共同的完整命令、权限/窗口/选卡、报告再生和故障处置见 [Base 使用手册](../../../../README.md)。

## 指标和验收

GEMM ops = 2×M×N×K×ITERS，除以 elapsed_seconds 和 1e12 得 TFLOPS/TOPS。

历史实测与当前限制见 [公开验证说明](../../../../../docs/thead/validation.md)。
既有结果仅属于固定镜像、设备集合和配置，不作为所有设备或尺寸的性能保证。

每 run 保存 metric、correctness、设备 binding、配置/code identity、原始监控、cleanup/postflight/lease 和 report。原始失败不得覆盖。单次结果不改变 runtime 声明或扩大验证范围。

正确性以零容差参考对照为准（int32 累加精确），大矩阵仍用固定行列采样与输出有限性检查；另有对 float64 参考的精确性核验记录。计时采用 perf_counter_ns 与目标设备完整同步；torch.cuda.Event 计时在本锁定栈实测可用。


## 静态计划

从仓库根目录执行；卡号仅为示例。锁定 runtime 处于 candidate 状态，dry-run 与真实执行均显式 `--allow-candidate-runtime`。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case computation-INT8:PPU --physical-device-ids 7 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
