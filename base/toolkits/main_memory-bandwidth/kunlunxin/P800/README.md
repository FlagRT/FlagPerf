# 昆仑芯 P800 显存带宽实验

本 Case 测量设备内存内部的 D2D 拷贝带宽。每卡 2 个逻辑点，两种口径分开实现、分开
报告，禁止合并或由其一推导另一个：

| 逻辑点 | 实现 | 字节计数口径 |
|---|---|---|
| `d2d` | XRE runtime memcpy | 载荷计 1× |
| `d2d-kernel` | XBLAS Scopy kernel | 读写合计计 2× |

`d2d-kernel` 是设备内存读写带宽的实测参考；`d2d` 是使用 runtime memcpy API 时的
拷贝吞吐参考，不能当作 HBM 带宽上限。例如约 11 GB/s 的 runtime memcpy 与约
2.1 TB/s 的 Scopy 结果可同时成立，不应合并成约 1.07 TB/s 的“中位数”。两者都包含
调用至完成同步的耗时，Scopy 也不是理论峰值。计数的 1×/2×差异不足以解释两者的
全部差距，具体瓶颈须另行诊断。

`xpu_memcpy` 进入的是 runtime 的通用复制路径。它内部怎样安排传输、如何提交任务、
是否分块、怎样等待，都由 runtime 实现决定。

Scopy 则进入数学库的向量复制实现。这类设备内核通常可以让许多执行单元并行读取连续地址，
再并行写入目标地址，充分利用设备内存系统。

以上说明两条路径的职责与通常工作方式；本报告未剖析 runtime 内部实现或 Scopy 的具体调度，
不据吞吐差距断言主机中转、PCIe 路径或内部回退。

报告保留原有 Case 行、列名、详情和图表格式，仅在“指标摘要”单元格内分别标注
“D2D设备内存读写带宽”（Scopy，读写 2×）与“XRE runtime memcpy 拷贝吞吐”（载荷 1×）。
两项仍按设备、载荷、字段和单位分别统计，不合并中位数；状态、范围、耗时与限制/错误
均沿用整个 Case 的原始值。双卡五次重复为 20 个执行目标，耗时为整个 Case 总耗时。
旧报告不会因代码更新自动变化；冻结结果应保留，展示重建只能在副本上进行并记录生成器版本。

## 测量协议

- 默认载荷 512 MiB；数据校验在计时区间之外。
- 两种口径使用不同实现与计数定义，属并列指标而非矛盾；报告中以名称区分，
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
