# 平头哥 PPU 单机 P2P 带宽实验

本 Case 调用锁定镜像内官方 `p2pBandwidthLatencyTest` 测量同机两卡间的 P2P 带宽。
实现不导入 PyTorch，不运行 Base workload，也不读取历史 Base 测量。

## 测量协议

- 每个无序设备对的两个方向都作为有序对独立测量：`uni-write`、`uni-read`、
  `bi-write`（并发双向）各 1 点；双向值不由单向推导，也不得把单向乘二当作双向。
- day-9 矩阵显示真实不对称（如 42.83 对 46.14 GB/s），两个方向都保留原值。
- 默认载荷 40000000 字节（`--p2p-bytes` 可改，须为 4 的倍数）；
  每点 `--p2p-samples`（默认 15）次 repeat。
- `--minimum-seconds`（默认 0，参考监控范围 15 秒）为窗口下限：达到下限前同一
  测量点重复调用该官方工具，指标取各次调用原值的中位数，每次输出独立归档。
- 解析按矩阵头行锁定：`(Unidirectional)`/`(Bidirectional)` 关键字、
  `(40000000 *2 Bytes)` 字节 token、WRITE/READ 方向门，单元格按 BDF→序号定位。
- `data matched` 数据校验标记参与诊断层；缺失或为假按失败点保留。
- 两个 rank 的完整矩阵、stdout/stderr 与退出码逐 repeat 归档。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/thead/PPU/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。

```bash
bash base/toolkits/interconnect-P2P_intraserver/thead/PPU/main.sh \
  --physical-device-ids 14,15 \
  --allow-candidate-runtime --repeat 5 \
  --result-root base/result/thead-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。P2P 至少需要两卡；
单卡选择时该 Case 无测量点（保持 not-run）。监控默认开启并覆盖两张参与卡。

判定沿用三层状态：测量层要求退出码为零、矩阵与方向门符合、身份一致且单元格为
正值有限数；监控层要求每个计时目标至少 10 个完整位于真实测量窗口内的有效样本；
诊断层含 `data matched` 校验。三层全部通过 Case 才为 passed。五次重复按完整
测量点分组，重复组 CV（ddof=1）>5% 标记 unstable；稳定性结论不改变测量状态。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/thead/PPU/README.md)。
