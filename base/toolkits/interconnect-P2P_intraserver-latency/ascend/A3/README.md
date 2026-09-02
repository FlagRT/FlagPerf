# Ascend A3 单节点 P2P 时延实验

显式选卡时，对每个有向 Device 对和 512 B、4 KiB、64 KiB、1 MiB 四个负载分别
执行 DMI latency 命令，A→B 与 B→A 均实测且绝不镜像推导。默认全设备模式仅执行
512 B card-mode 矩阵。每个点保存 ns、源/目标 Device、负载大小及原始证据；
signalQuality 只作为独立厂商诊断，不能替代时延测量。
