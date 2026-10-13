# 昆仑芯 P800 Base — computation-FP64

当前锁定 runtime 下为 **启动前 skipped**，不启动 Docker、不获取设备 lease、不生成性能数字。

技术依据：锁定栈探针输入 FP64 却输出 FP32，未证明符合本 Case 的原生 FP64 输出。

该结论属于当前 runtime/case/执行器范围，不是对芯片所有未来软件栈的能力判断。禁止通过换精度或单机伪执行将其标成通过。`--case computation-FP64:P800` 的 dry-run 会在 `applicability.reason` 输出原因。

完整支持矩阵和启用新能力的要求见 [Base 使用手册](../../../../README.md)，已验证范围见 [验证说明](../../../../../docs/kunlunxin/validation.md)。


## 静态计划

从仓库根目录执行；卡号仅为示例。

```bash
python3 base/run.py benchmark run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --case computation-FP64:P800 --physical-device-ids 4 \
  --nproc-per-node 1 --timeout 300 --dry-run
```

真实执行的环境、权限、预约窗口和全新结果目录参数见 [Base 使用指南](../../../../README.md)。
本项当前不提供真实测量；移除 dry-run 后仍应保留 skipped 原因，不能强制绕过 requirements。
