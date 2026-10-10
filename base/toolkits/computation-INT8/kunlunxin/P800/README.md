# 昆仑芯 P800 INT8 算力实验

本 Case 通过独立 C++ microbenchmark 调用锁定镜像内的 XBLAS 库测量 INT8
吞吐。实现不导入 PyTorch，不运行 Base workload，也不读取历史 Base 测量。

## 测量协议

`native_bench.cpp` 调用 `fc_fusion<int8_t,int8_t,float,int8_t,float,float>`
（`xblas_legacy_deprecated_api.h`，API 模式取自厂商
`samples/cublas_legacy/deprecated/sample_fc_fusion.cpp`）：INT8 输入与 TGEMM、FP32
输出，数据 maxima 127、alpha=1、beta=0、无 bias、LINEAR 激活。

- 默认规模 8192³（`--matrix-size` 可改）；超时按失败点保留，不静默缩小。
- 指标：`2*M*N*K / 计时区间`，单位 TOPS；每卡 1 点。
- 本实现明确不是 INT8→INT32 GemmEx；后者与 Lt 路线在本锁定镜像中失败，历史失败
  记录保留，不据文档例外改写。
- CPU 参考回读在计时区间之外执行。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/kunlunxin/P800/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。
以下命令从仓库根目录执行，可先加 `--dry-run` 核对计划。

```bash
bash base/toolkits/computation-INT8/kunlunxin/P800/main.sh \
  --physical-device-ids 4 \
  --repeat 5 --samples 50 --warmup 5 --minimum-seconds 15 \
  --result-root base/result/p800-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。示例中的卡号仅为
示意；物理卡号不是 `/dev/xpuN` 或容器序号，每轮运行前须核实空闲状态、ECC、UUID 与
PCI 映射。监控默认开启并与计时同窗。

判定沿用三层状态：测量层要求退出码为零、身份一致、正值有限样本与正确性回读；
监控层要求每个计时目标至少 10 个完整位于原生计时窗口内的有效样本，
不足保持 partial；诊断层为厂商工具链合规校验。三层全部通过 Case 才为 passed。重复指独立进程；
五次重复组 CV（ddof=1）与进程内 CV（ddof=0）分开计算，报告以 CV > 5% 标记 unstable；稳定性结论不改变
测量状态，也不扩大 runtime 的验证范围。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/kunlunxin/P800/README.md)。

总览沿用简短摘要；具体设备、载荷、模式及字段的比较以详情中的独立重复组为准。
