# 昆仑芯 P800 显存带宽实验

本 Case 测量设备内存内部的 D2D 拷贝带宽。每卡 2 个逻辑点，两种口径分开实现、分开
报告，禁止合并或由其一推导另一个：

| 逻辑点 | 实现 | 字节计数口径 |
|---|---|---|
| `d2d` | XRE runtime memcpy | 载荷计 1× |
| `d2d-kernel` | XBLAS Scopy kernel | 读写合计计 2× |

## 测量协议

- 默认载荷 512 MiB；数据校验在计时区间之外。
- 两种口径的数值差异来自实现与计数定义，属并列指标而非矛盾；报告中以名称区分，
  不以单一数值代表显存带宽。
- 仅限单机，不含 Host↔Device 路径与跨卡拷贝。

## 运行与判定

```bash
bash base/toolkits/main_memory-bandwidth/kunlunxin/P800/main.sh \
  --physical-device-ids 4 --allow-candidate-runtime \
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
