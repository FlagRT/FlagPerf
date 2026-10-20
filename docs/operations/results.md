# 如何读 Operation 指标与报告

[返回 Operation 入门](../../operation/README.md)

按“执行 → 正确性 → 路由 → 性能”的顺序阅读。`partial` 可能表示路由证据不足，
不一定是数值错误；底层实现不可用会保留 blocked/failed 及原因，工具不暗中改用另一后端。

## 正确性与比较对象

浮点案例以实际量化输入的 CPU FP64 运算为参考；整数和布尔按精确规则比较。
梯度案例同时检查相应输入梯度，随机算子沿用自身结构、mask 和梯度契约。
原生与 FlagGems 两侧各自经过参考检查，不把两侧接近当作正确性的充分条件。

`--oplib both` 的性能比较仅在本次运行内配对相同 Case、dtype、设备与工作量。
跨运行合并报告保留各来源比较，不重新拼出跨运行的 off/on 比值。

## 指标边界

| 内容 | 含义与边界 |
|---|---|
| 首次、预热、正式耗时 | 分开记录；同步后的批次计时包括 Python 提交、分配与设备执行，不等同纯 kernel 耗时 |
| 吞吐与有效带宽 | 按算子工作量及逻辑读写量计算；不是硬件总线计数器 |
| FLOPS 与利用率 | 沿用 Case 运算量公式；利用率需要操作者提供对应峰值，缺失显示 N/A |
| 轮次波动 | 对已有测量轮次给出统计；样本不足的统计不伪造为零 |
| 显存 | 在独立阶段记录 allocator 指标，不把整卡占用与算子增量混为一谈 |
| 路由 | 注册、调用和厂商 dispatch 证据；CPU profiler 调用链本身不能证明设备 kernel 执行 |
| `profiling timeline/full` | 独立设备采样；只有关联充分的目标窗口用于归因，缺口明确报告 |

profiler、显存和失败诊断均不插入普通性能计时区间，但会增加任务总耗时。
`full` 尝试额外性能指标组，可用范围由设备、依赖与采集能力决定；不可用项不会当作测得零值。

P800 在独立 probe 中，分别把首次/缓存的 ATen 设备计算关联到同线程主机 launch、`ac2g` 流及正确设备 kernel。
`route-*-trace.json` 与 `route-*-dispatch.json` 保存原始事件、逐计算关联和缺失原因；
锁定栈的 C++ `LinearFunction` 另用明确的 CPU 记录及经核对的 XBLAS kernel 链证明。
仅复制/分配、邻近 kernel 或不完整关联不能通过；观察到设备输入转到 CPU 时失败。
未观察到回退不等于已经排除所有 C++ 内部回退；路由通过也不保证 Gems 路径采用专用 Gems kernel。
probe 与 measure 只允许厂商指南记录的两项 profiler 环境转换，其他身份变化仍失败；计时进程不开启该观察器。
共享卡的性能资格始终 partial，即使正确性、路由、memory 或 timeline 各自通过，也不生成合格性能比值。
完整 P800 协议和硬件计数器限制见[厂商指南](../../operation/vendors/kunlunxin/README.md)。

## 结果文件

终端 `Report:` 指向阅读入口，`Summary:` 指向总状态。每个任务子目录保存自己的配置、输入/参考、
probe、测量、显存、profiling、诊断和日志；并非每个任务都会产生所有文件。
根目录 `artifacts.json` 记录封存摘要；`source/` 保存运行代码，便于解释结果对应的版本。

普通运行退出码 0 表示 passed，2 表示 partial，1 表示 failed；诊断命令的退出码按
[诊断说明](diagnosis.md) 解读，不能混用。

```bash
# 多个普通评测结果汇总到一个新目录。
python3 operation/run.py report --run-dir /absolute/run-a --run-dir /absolute/run-b \
  --output-dir /absolute/new-aggregate
```

汇总要求同主机、厂商、执行模式与实际镜像身份；来源必须是原始运行而不是再次汇总。
相同工作量采用所给顺序中最后一次已尝试记录，不让未运行/中断记录抹去已有完成结果。
不同诊断、性能和 profiling 协议保持区分；诊断包不混入普通评测覆盖统计。
