# 平头哥 PPU TF32 算力 实验

本 Case 通过隔离子进程直接调用锁定镜像内 `libacblas` 的 `acblasGemmEx` 测量 TF32 GEMM 吞吐。
实现用 ctypes 绑定厂商库，不导入 PyTorch，不运行 Base workload，也不读取历史 Base 测量。

## 测量协议

`compute_child.py` 按上述 IO 约定调用 `acblasGemmEx`：FP32 输入/输出，compute type ACBLAS_COMPUTE_32F_FAST_TF32，algo ACBLAS_GEMM_DEFAULT。

- 默认规模 8192³（`--matrix-size` 可改）；大尺寸是独立测量口径，超时按失败点保留，
  不静默替换为更小规模。
- 指标：`2*M*N*K / median_ns / 1000`，单位 TFLOPS；每卡 1 点。
- 每卡一个隔离子进程，绑定单卡（`cudaDeviceProp` UUID 回读参与身份门）；
  子进程超时由 `--compute-timeout`（默认 900 秒）约束，含厂商 JIT 预热。
- 计时 `--compute-samples`（默认 50）次调用；预热与正确性回读在计时区间之外。
- `--minimum-seconds`（默认 0，参考监控范围 15 秒）为窗口下限：样本数达到后
  计时循环继续到下限（与 P800 native_bench 同语义），监控同窗采样才可判定。
- 正确性回读：以确定性参考实现核对输出元素，回读失败按失败点保留。
- 原始样本、stdout/stderr 与退出码逐 repeat 归档。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/thead/PPU/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。
以下命令从仓库根目录执行，可先加 `--dry-run` 核对计划。

```bash
bash base/toolkits/computation-TF32/thead/PPU/main.sh \
  --physical-device-ids 14,15 \
  --allow-candidate-runtime --repeat 5 \
  --result-root base/result/thead-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。示例中的卡号仅为
示意；物理卡号不是容器序号，每轮运行前须核实空闲状态、ECC、UUID 与 PCI 映射。
监控默认开启并与计时同窗。

判定沿用三层状态：测量层要求退出码为零、身份一致、正值有限样本与正确性回读通过；
监控层要求每个计时目标至少 10 个完整位于真实测量窗口内的有效样本，不足保持 partial；
诊断层为厂商工具链合规校验。三层全部通过 Case 才为 passed。重复指独立子进程；
五次重复组 CV（ddof=1）与进程内 CV（ddof=0）分开计算，报告以 CV > 5% 标记 unstable；
稳定性结论不改变测量状态，也不扩大 runtime 的验证范围。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/thead/PPU/README.md)。
