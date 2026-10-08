# Kunlunxin P800 Base — computation-FP8

当前锁定 runtime 下为 **启动前 skipped**，不启动 Docker、不获取设备 lease、不生成性能数字。

技术依据：FP8 mm and scaled-mm APIs returned NOT IMPLEMENTED on the locked stack

该结论属于当前 runtime/case/执行器范围，不是对芯片所有未来软件栈的能力判断。禁止通过换精度或单机伪执行将其标成通过。`--case computation-FP8:P800` 的 dry-run 会在 `applicability.reason` 输出原因。

完整支持矩阵和启用新能力的要求见 [Base 使用手册](../../../../docs/p800-base-handoff.md)，本轮复核见 [验收报告](../../../../docs/p800-base-acceptance-20261008.md)。
