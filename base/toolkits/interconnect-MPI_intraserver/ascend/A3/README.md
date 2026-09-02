# Ascend A3 单节点 MPI/HCCL AllReduce 性能实验

本 Case 使用 MPICH 启动 CANN 随包 `hccl_test/bin/all_reduce_test`，测量单节点所选
逻辑 Device 上的 HCCL FP32/SUM AllReduce。产品默认消息范围为 8 KiB–1 GiB、倍率
2、预热 10 次、测量 20 次并启用正确性校验；HCCL 是实际通信库，MPI 负责进程启动。

权威性能值是 `hccl_test` 报告的 `alg_bandwidth(GB/s)`；同时保存消息大小、
`aveg_time(us)`、正确性、rank→Device 映射、命令、环境和原始输出。v1 不派生 bus
bandwidth。任何正确性失败均使测量失败；缺点或超时保留已产生的证据并标记覆盖不足。

该 Case 必须显式 `--case interconnect-MPI_intraserver` 选择，不属于默认 toolkit 套件。

运行 AllReduce size sweep 时默认并行采集全部 rank 所在目标的 `hccs-bw` 逐链路
Rx/Tx 原值，并在 `report_monitor.md` 展示 timeline、逐样本 UTC/原值和 topo 路由覆盖。
该证据只说明相同 workload 窗口内链路活动，不换算 bus bandwidth 或理论峰值利用率。
若所选路由含 SIO、命令不支持或样本不足，HCCL 测量及正确性结果原样保留，监控和最终
Case 为 partial。
