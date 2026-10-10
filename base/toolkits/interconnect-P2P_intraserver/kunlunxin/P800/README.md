# 昆仑芯 P800 单机 P2P 带宽实验

本 Case 测量同一主机内两卡之间的点对点拷贝带宽，调用锁定镜像内 XRE 的
`xpu_memcpy_peer`。N 张可见卡产生 N(N-1)/2 个无序 pair（如 4 卡 6 pair），每 pair
2 个逻辑点。

## 测量协议

- 升序单向：固定由编号较小的卡拷向编号较大的卡，独立计时，报告 GB/s。
- 并发双向：两个方向在独立线程同时提交，以实际两份载荷为分子、两方向共同完成
  时间为分母计算带宽（`2*payload_bytes/median_elapsed_ns`，GB/s）；该口径包含
  线程协调开销；逐点结果和重复组与单向分别统计，不互推。
- 默认载荷 32 MiB；`--p2p-payload-bytes 268435456` 显式选择 256 MiB，两种载荷是
  独立测量口径。
- pair 覆盖必须与计划一致；未测 pair 不推导。

仅限单机；跨机通信不属于本 Case。本 Case 不判定物理链路类型（XPULink/PCIe），
链路证据以当轮拓扑记录为准。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/kunlunxin/P800/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。
以下命令从仓库根目录执行，可先加 `--dry-run` 核对计划。

```bash
bash base/toolkits/interconnect-P2P_intraserver/kunlunxin/P800/main.sh \
  --physical-device-ids 4,7 \
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
