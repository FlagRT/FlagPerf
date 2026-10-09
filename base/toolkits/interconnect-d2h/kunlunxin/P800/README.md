# 昆仑芯 P800 Device→Host 带宽实验

`d2h` 的方向是 **Device 显存 → Host 内存**。本 Case 调用锁定 candidate 镜像内 XRE
的 `xpu_memcpy` / `xpu_memcpy_async`，与 Base 的 PyTorch 数据通路分开运行与统计。

## 测量协议

每卡 4 个逻辑点：pageable/pinned × blocking/async 四种独立组合，各自报告 GB/s；
不同组合之间不合并、不互推。

- blocking：单次 `xpu_memcpy` 同步调用；nonblocking：`xpu_memcpy_async` 提交后
  stream wait 统一等待。
- 默认载荷 512 MiB；Host 侧分配与全量数据校验均在计时区间之外。
- pinned 内存默认 `posix_memalign` + `xpu_host_register`；`--pinned-api alloc`
  显式选择 `xpu_host_alloc`。
- 本 candidate 中 512 MiB pinned async 整块提交返回错误 -999；默认协议按
  `--async-chunk-bytes 1048576` 以 1 MiB 分块提交、最后一次等待。指标记录
  `async_chunk_bytes`、`api_calls_per_sample` 与 `submission_scope`，不把该协议
  描述成单次整块 API；`--async-chunk-bytes 0` 复现整块路径，小于分块阈值的载荷
  使用单次提交。历史失败记录保留。

带宽数值受 Host 内存、NUMA、设备互联与运行时状态共同影响；本 Case 不把结果归属到
具体物理链路，也不等于 HBM D2D 带宽。仅限单机，不含 H2D 与跨机通信。

## 运行与判定

```bash
bash base/toolkits/interconnect-d2h/kunlunxin/P800/main.sh \
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
