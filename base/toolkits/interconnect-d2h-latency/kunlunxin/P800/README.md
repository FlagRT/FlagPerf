# 昆仑芯 P800 Device→Host 时延实验

`d2h` 的方向是 **Device 显存 → Host 内存**。本 Case 调用锁定 candidate 镜像内 XRE
的 `xpu_memcpy` / `xpu_memcpy_async` 测量小载荷传输时延。

## 测量协议

每卡 16 个逻辑点：pageable/pinned × blocking/async 四种组合 × 512 B / 4 KiB /
64 KiB / 1 MiB 四种载荷，单位 ns。

- 口径：单方向一次 API 调用自提交至完成同步的主机 monotonic 耗时，包含提交与同步
  开销；不是往返时间（RTT），也不是纯 kernel 或 DMA 时间。
- blocking 计一次同步调用；nonblocking 为 `xpu_memcpy_async` 提交 + stream 等待。
- 默认 `--async-chunk-bytes 1048576` 下，四档时延载荷均不大于分块阈值，pinned async
  使用单次提交。若显式调小阈值，超过阈值的 pinned async 载荷将分块提交、最后统一
  等待；这会改变测量协议，须按实际 `async_chunk_bytes` 与 `api_calls_per_sample` 比较。
- 未测组合与载荷不推导补齐；逐点独立归档原始样本。

仅限单机，不含 H2D；时延数值不用于推导带宽。

## 运行与判定

```bash
bash base/toolkits/interconnect-d2h-latency/kunlunxin/P800/main.sh \
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
