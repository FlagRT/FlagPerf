# Ascend Operation 支持说明

使用方法、计时边界和阈值见 [Operation 指南](../../README.md)，环境见
[Ascend 适配指南](../../../docs/ascend/README.md)。公共规划与调度位于 `operation/runtime/`，
Ascend 的初始化、映射、同步、占用检查、运行身份和路由取证集中在 `adapter.py`。

52 个 Case 已进入统一 CLI，接入数量不等于全精度、双路径全部通过。
[公开覆盖记录](../../../docs/ascend/operation-coverage.json)是 2026-09-09 的历史跨批次汇总：
308 个适用组合中 277 passed、24 blocked、5 failed、2 partial，另有 420 个不适用项。
验证设备为逻辑 Device 14（物理 NPU 7），锁定 operator 镜像 ID 见该记录。
原始证据在整理时校验过哈希，公开文件仅保留状态与来源摘要；此次集成未重跑完整设备矩阵。

| 组合 | 已观察到的限制 |
|---|---|
| nativetorch：isnan、rsub | Torch-FL backend 注册缺失 |
| nativetorch FP32：addmm、bmm、linear、mm、mv | 固定阈值下数值检查失败；唯一根因未确认 |
| nativetorch FP16/BF16：linear | 数值通过，目标矩阵计算路由证据不足 |
| flaggems：amax、outer | squeeze.dims 或 mul.out 注册缺失 |
| flaggems：dropout、native_dropout | Philox RNG 状态布局与运行栈不匹配 |
| flaggems：mm | Triton 不接受 SPLIT_K 参数 |
| flaggems：mul | 原 Case 的标量输入进入 Tensor 重载 |

依赖异常分类为 `blocked` 仍表示未解决。未通过组合保留错误和诊断，不自动切换实现，
不放宽数值阈值，也不把 CPU profiler API 事件当作设备 kernel 证明。
缓存命中路径只有观察到实际 launcher 执行状态才可通过路由门禁。
纯 kernel 时间仍为 `not-supported`；等效 FLOPS 与设备峰值利用率的限制见公共指南。
