# 平头哥 PPU Base — interconnect-P2P_interserver

当前锁定 runtime 下为 **启动前 skipped**，不启动 Docker、不获取设备 lease、不生成性能数字。

技术依据：统一执行器只支持单宿主 nnodes=1，且本轮适配没有接入配对目标机，不能运行跨机点对点通信。

该结论属于当前 runtime/case/执行器范围，不是对芯片所有未来软件栈的能力判断。禁止通过换精度或单机伪执行将其标成通过。`--case interconnect-P2P_interserver:PPU` 的 dry-run 会在 `applicability.reason` 输出原因。

完整支持矩阵和启用新能力的要求见 [Base 使用手册](../../../../README.md)，已验证范围见 [验证说明](../../../../../docs/thead/validation.md)。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/thead_ppu_torch210.yaml \
  --case interconnect-P2P_interserver:PPU --physical-device-ids 7 \
  --nproc-per-node 1 --timeout 300 --allow-candidate-runtime --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
本项当前不提供真实测量；移除 dry-run 后仍应保留 skipped 原因，不能强制绕过 requirements。
