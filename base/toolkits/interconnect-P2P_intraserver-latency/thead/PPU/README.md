# 平头哥 PPU 单机 P2P 时延实验

本 Case 调用锁定镜像内官方 `p2pBandwidthLatencyTest` 的 GPU 时延矩阵测量同机
两卡间 P2P 传输时延。实现不导入 PyTorch，不运行 Base workload，也不读取历史 Base 测量。

## 测量协议

- 每个无序设备对的每个方向各 1 点；时延取 `GPU P2P Transmission Latency Matrix`，
  单位 us→ns 只做单位换算。
- CPU 侧指针追踪矩阵（`CPU P2P Transmission Latency Matrix`）不是设备时延证据，
  明确排除，不进入指标。
- 单元格按 BDF→序号定位；对角线（同卡）不构成测量点。
- 每点 `--p2p-samples`（默认 15）次 repeat；`P2P test finished` 完成标记参与判定。
- `--minimum-seconds`（默认 0，参考监控范围 15 秒）为窗口下限：达到下限前同一
  测量点重复调用该官方工具，指标取各次调用原值的中位数，每次输出独立归档。
- 完整矩阵、stdout/stderr 与退出码逐 repeat 归档。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/thead/PPU/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。

```bash
bash base/toolkits/interconnect-P2P_intraserver-latency/thead/PPU/main.sh \
  --physical-device-ids 14,15 \
  --allow-candidate-runtime --repeat 5 \
  --result-root base/result/thead-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。P2P 至少需要两卡；
单卡选择时该 Case 无测量点（保持 not-run）。监控默认开启并覆盖两张参与卡。

判定沿用三层状态：测量层要求退出码为零、GPU 矩阵存在且单元格为正值有限数、
身份一致；监控层要求每个计时目标至少 10 个完整位于真实测量窗口内的有效样本；
诊断层为工具链合规校验。三层全部通过 Case 才为 passed。五次重复按完整测量点分组，
重复组 CV（ddof=1）>5% 标记 unstable；稳定性结论不改变测量状态。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/thead/PPU/README.md)。
