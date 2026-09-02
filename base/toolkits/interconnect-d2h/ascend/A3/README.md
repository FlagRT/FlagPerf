# Ascend A3 Device→Host 带宽实验

执行 `ascend-dmi --bw -t d2h -s 536870912 --et 50 -q --fmt json`，测量每次
512 MiB、50 次的 Device 内存到 Host 内存传输。显式选卡时逐逻辑 Device 执行；
默认模式在全设备输出覆盖不足时逐设备补测。保存 GB/s 指标、原始 stdout/stderr、
退出码、设备选择、拓扑和 DMI bandwidth 诊断；不把结果等同于 HBM 或 HCCL 带宽。

默认同期采集 `npu-smi info -t hccs-bw -time 1000` 的逐链路及 `total` Rx/Tx
原值，以样本区间与 D2H 进程墙钟区间重叠建立作证关系。每个目标至少要求 10 个有效
workload 样本和 1 个主窗口样本；不支持或采样不足只降低监控/最终 Case 状态，不改变
DMI D2H 原值，也不派生平均值或峰值利用率。
