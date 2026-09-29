# Ascend Operation 适配

[返回 Operation 入门](../../README.md) · [环境准备](../../../docs/ascend/README.md)

当前运行路径为 CANN 9 / PyTorch 2.10 / Torch-FL，设备 API 是 `torch.flagos`。
厂商适配负责设备初始化、同步、物理映射、运行身份、资源检查、路由解释和独立 profiling；
通用 Case 继续负责输入构造和算子调用。不要把 Inference 使用的 torch_npu 镜像直接作为此处默认镜像。

## 运行前

准备环境说明中的锁定镜像、可用驱动和 Docker 权限，选择空闲设备。示例从仓库根目录执行：

```bash
python3 operation/run.py run --vendor ascend --device-ids 0 --case abs \
  --dtype FP32 --workload smoke --dry-run
python3 operation/run.py run --vendor ascend --device-ids 0 --case abs \
  --dtype FP32 --oplib both --workload smoke --allow-privileged-root
```

`--dry-run` 不证明镜像或设备可用；正式运行会核对实际镜像、设备租约、前后健康状态，并限制每个阶段的执行时间。
异常后的清理/健康检查失败会阻止继续运行。程序不修补底层库，也不把依赖受阻改判为算子通过。

## 正确性、路由与性能

52 个案例不代表所有 dtype/实现组合均可用；使用 `list --case NAME` 查看输入类型，实际运行按组合报告。
原生与 FlagGems 路径分别检查 CPU 参考。默认数学模式来自所选运行栈，不额外宣称 IEEE 严格模式。

正式测量阶段关闭逐调用 dispatch 日志；路由在独立 probe 中取证。
`[flagos cpu_fallback]` 等提示用于解释观测到的回退，不是硬件 kernel 覆盖率。
当前 Ascend 路径不把 CUDA Triton 的计时器用于 NPU；无法取得的计时指标保持 N/A。

`--profiling timeline` 采集目标调用的设备时间线；`full` 另尝试厂商指标组。
目标窗口和设备事件关联充分才报告可归因结果；CPU 回退、缺失 trace 或关联不足会显示 partial。
这些采集与原性能测量分开。

失败后的公共检查、离线诊断与显式 replay 见 [诊断指南](../../../docs/operations/diagnosis.md)。
错误分类只是线索；例如注册缺失、随机状态布局不匹配或编译签名不兼容，需要结合对应日志和依赖身份判断。
历史覆盖记录见 [覆盖摘要](../../../docs/ascend/operation-coverage.json)，其日期和源码边界保持原义，不自动代表当前代码全量复验。

## 代码职责

- `adapter.py`：运行栈、设备生命周期、路由和依赖错误分类。
- `profiling.py`：独立设备采集与目标窗口关联。
- `report_metrics.py`：报告中的 Ascend 指标名称、单位和范围。
- 通用诊断与报告位于 `operation/runtime/`；旧 schema 1 报告读取由 `operation/legacy/report_v1.py` 保留。
