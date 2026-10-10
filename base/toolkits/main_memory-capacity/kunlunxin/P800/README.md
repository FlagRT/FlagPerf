# 昆仑芯 P800 显存容量属性实验

本 Case 通过锁定 candidate 镜像内的 `xpu-smi` 读取显存容量属性，不通过逐步分配
寻找 OOM 边界，因此不是"当前可分配显存"测试。

## 测量协议

- 每卡 1 点：Memory Usage 的 Total/Used/Free，单位 MiB，原值进入 manifest。
- 读取前后核对设备 PCI 地址与 UUID；物理卡号不当作设备节点或容器序号。
- 容量是静态查询：监控层使用静态查询观察，不套用计时负载的最短窗口要求，也不
  生成伪造的利用率 timeline。
- 不报告 allocator 碎片、业务可用容量或跨卡求和；更大 scope 必须基于逐卡原值
  显式另行求和。

## 运行与判定

```bash
bash base/toolkits/main_memory-capacity/kunlunxin/P800/main.sh \
  --physical-device-ids 4 --allow-candidate-runtime \
  --repeat 5 \
  --result-root base/result/p800-toolkit
```

`main.sh` 将参数原样转发给宿主 facade `base/run.py toolkit run`。示例中的卡号仅为
示意；物理卡号不是 `/dev/xpuN` 或容器序号，每轮运行前须核实空闲状态、ECC、UUID 与
PCI 映射。监控默认开启；容量按静态查询口径观察，不涉及计时窗口。

判定沿用三层状态：测量层要求查询退出码为零、Total/Used/Free 等字段按 MiB 成功解析，
且 PCI 地址与 UUID 匹配所选设备；Used 为零是有效值。本 Case 不产生原生计时样本，
不要求计时样本正值或设备数据回读。监控层使用静态查询观察，不套用计时负载的最短
窗口，也不生成伪造的利用率 timeline；诊断层结合查询身份校验与宿主健康观测。
三层全部通过 Case 才为 passed。重复指独立查询进程；报告按设备和字段分别统计五次
查询值，CV > 5% 标记 unstable，均值为零时 CV 为 N/A。稳定性结论不改变测量状态，
也不晋级 candidate runtime。`--samples`、`--warmup` 和 `--minimum-seconds` 不控制静态查询；
Toolkit 容量查询不需要 Base 分配/OOM case 的 `--allow-high-risk-case`。

完整契约、工具来源、证据目录与权限路径见 [Toolkit 总说明](../../../_common/kunlunxin/P800/README.md)。
