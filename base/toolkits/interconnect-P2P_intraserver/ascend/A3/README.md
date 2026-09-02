# Ascend A3 单机 P2P 带宽实验

本 Case 测量同一服务器内 Ascend 设备之间的 P2P 带宽。旧 README 的“不同服务器”描述错误，已经按 `intraserver` 和 DMI card-mode 实际语义修正。

## 测量协议

```bash
ascend-dmi --bw -t p2p -m card -q
```

DMI 26.1.0 的默认 card-mode P2P 可能忽略 size、execute-times 和 JSON format 参数，因此这里保留 normal 输出，并按以下语义解析：

1. 识别 `Unidirectional Peer to Peer Test` 与 `Bidirectional Peer to Peer Test` 区段；
2. 按矩阵的源设备行、目标设备列解析全部设备对；
3. 每条指标绑定 source、destination、direction 和 `GB/s`；
4. 参与设备少于两个时返回 `partial`，不再读取第 14/22 行。

对应厂商诊断为 `signalQuality --lt hccs`。诊断、原始 P2P 输出和 `npu-smi info -t topo` 必须一起解释，单个 GB/s 不能代表所有设备对。

```bash
python3 base/run.py toolkit run --case interconnect-P2P_intraserver --npu-ids 1 \\
  --allow-privileged-root --allow-disruptive-dmi
```

本轮只处理单机 P2P；跨机 P2P、MPI/HCCL、压力测试和复位均不在范围内。

作证监控与权威命令同粒度：显式选卡时每个无序 Device 对形成一个监控单元，默认模式
整张 P2P 矩阵形成一个单元。报告同时展示 topo 路由和目标两端 `hccs-bw` 逐链路 Rx/Tx
原值。`HCCS/HCCS_SW` 可由该计数器覆盖；若路由为 `SIO`，当前锁定 CANN 9 栈没有动态
SIO 带宽计数，因此明确标记 partial，绝不以 HBM 使用率冒充链路利用证据。
