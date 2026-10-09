# 昆仑芯 P800 单机 P2P 时延实验

本 Case 测量同一主机内两卡之间的点对点拷贝时延，调用锁定 candidate 镜像内 XRE 的
`xpu_memcpy_peer`。N 张可见卡产生 N(N-1)/2 个无序 pair，每 pair 1 个逻辑点。

## 测量协议

- 固定 64 KiB、升序单向（编号较小的卡拷向编号较大的卡）拷贝 + 完成等待，单位 ns。
- 口径与 H2D/D2H 时延一致：一次 API 调用自提交至完成同步的主机 monotonic 耗时；
  不是往返时间（RTT），也不是纯 kernel 或 DMA 时间。
- 不推导反向结果；pair 覆盖必须与计划一致。

仅限单机；本 Case 不判定物理链路类型，链路证据以当轮拓扑记录为准。

## 运行与判定

```bash
bash base/toolkits/interconnect-P2P_intraserver-latency/kunlunxin/P800/main.sh \
  --physical-device-ids 4,7 --allow-candidate-runtime \
  --repeat 5 --samples 50 --warmup 5 --minimum-seconds 15 \
  --result-root base/result/p800-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。示例中的卡号仅为
示意；物理卡号不是 `/dev/xpuN` 或容器序号，每轮运行前须核实空闲状态、ECC、UUID 与
PCI 映射。监控默认开启并与计时同窗。

判定沿用三层状态：测量层要求退出码为零、身份一致、正值有限样本与正确性回读；
监控层要求每个计时目标至少 10 个完整位于原生计时窗口内的有效样本，
不足保持 partial；诊断层为厂商工具链合规校验。三层全部通过 Case 才为 passed。重复指独立进程；
五次重复组 CV 与进程内 CV 分开计算，报告以 CV > 5% 标记 unstable；稳定性结论不改变
测量状态，也不晋级 candidate runtime。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/kunlunxin/P800/README.md)。
