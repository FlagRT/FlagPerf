# 性能运行监控

`performance` 默认启动独立采集进程，约每秒观察设备和 worker 状态。采集不增加模型重放或设备同步。
打开结果目录的 `monitor.html`，按设备、阶段和时间范围查看；`report.md` 提供入口。

```bash
python3 run.py performance --config config/local.yaml --monitor on \
  --monitor-interval-ms 1000 --output result/monitored
python3 run.py performance --config config/local.yaml --monitor off --output result/unmonitored
```

采样间隔至少 100 ms。高频轮询并不改变硬件传感器的刷新频率；秒级曲线用于趋势和阶段关联，短 batch 的时延与归因仍以计时及 profiler 为准。

| 指标 | 来源与边界 |
|---|---|
| 设备利用率、HBM | Ascend 管理面 provider；按实际能力记录，不作为某个层的独占消耗 |
| 功率、频率、温度 | 管理面传感器；瞬时值、有效点数和刷新能力分开解释 |
| worker CPU | `/proc` CPU 时间增量 / 经过时间，单位 cores；1 表示一个满载核心 |
| worker RSS | `/proc` 常驻页；共享页不去重，不等于 Python 堆或设备显存 |
| host 可用内存 | Linux MemAvailable；与框架 allocator 指标分开 |

当前设备 provider 为 Ascend A3/910C，自动选择可用 DCMI 接口。其他厂商可通过 `vendors` 扩展统一 provider。
provider 能力不适用、访问失败、缺少样本或数据冲突均显式记录，缺失值不补零。
执行、分析、监控及页面生成分别保存状态；一个 completed 状态不保证每个监控指标完整。

先看有效点数和指标状态，再看曲线。功率或温度波动不能单凭时间接近就归因给某个算子；需要结合阶段事件和 profiler。
采集文件与统计位于结果目录的 `monitor/`；确切入口由报告列出。扩展接口见[目录与扩展](inference-case-doc.md)。
