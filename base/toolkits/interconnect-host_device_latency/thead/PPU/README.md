# 平头哥 PPU 主机-设备访问时延 实验

本 Case 调用锁定镜像内官方 `sailbandwidth` 的访问时延路径
测量主机-设备访问时延。实现不导入 PyTorch，不运行 Base workload，
也不读取历史 Base 测量。

## 测量协议

- testcase 索引锁定在 v2.1.0 构建：`32 host_device_latency_sm`；期望名与单位（ns）
  由导入期断言锁定。
- 这是该工具提供的访问时延口径（SM 路径），不是四档拷贝 API 时延扫描；
  该范围差异登记在计划的 `excluded_capabilities`，报告原样引用。
- 每点 `--transfer-samples`（默认 25）次迭代；解析工具 JSON stdout。
- 工具 JSON 原生以 ns 报值（`bandwidth_description` 带 "(ns)" 后缀，锁定镜像
  实测核对），不做换算；每卡 1 点。
- 原始样本、stdout/stderr 与退出码逐 repeat 归档。

## 运行与判定

先按 [Toolkit 总说明](../../../_common/thead/PPU/README.md#从宿主运行) 准备镜像、宿主权限与结果目录写权限。
以下命令从仓库根目录执行，可先加 `--dry-run` 核对计划。

```bash
bash base/toolkits/interconnect-host_device_latency/thead/PPU/main.sh \
  --physical-device-ids 14,15 \
  --allow-candidate-runtime --repeat 5 \
  --result-root base/result/thead-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。示例中的卡号仅为
示意；物理卡号不是容器序号，每轮运行前须核实空闲状态、ECC、UUID 与 PCI 映射。
监控默认开启并与计时同窗。

判定沿用三层状态：测量层要求退出码为零、testcase 索引/期望名/单位三重符合、
身份一致且样本为正值有限数；监控层要求每个计时目标至少 10 个完整位于真实测量
窗口内的有效样本，不足保持 partial；诊断层为厂商工具链合规校验。
三层全部通过 Case 才为 passed。五次重复按完整测量点分组，重复组 CV（ddof=1）
>5% 标记 unstable；稳定性结论不改变测量状态。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/thead/PPU/README.md)。
