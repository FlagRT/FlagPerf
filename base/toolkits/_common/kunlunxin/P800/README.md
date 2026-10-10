# 昆仑芯 P800 Toolkit 使用说明

P800 Toolkit 提供 12 项独立 C++ microbenchmark，直接调用锁定镜像中的 XBLAS/XRE。
它不导入 PyTorch、不运行 Base workload，也不读取旧 Base 性能值。
环境与镜像准备见 [P800 指南](../../../../../docs/kunlunxin/README.md)，统一入口见
[Base 使用指南](../../../../README.md)。

## 从宿主运行

所有命令从 FlagPerf 仓库根目录执行，先安装 `base/requirements-host.txt`。
物理 ID 4、7 仅为示例；真实执行前确认本轮预约、空闲、健康、UUID/PCI、目录权限和存储空间。

```bash
# 只生成计划，不启动设备负载。
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4,7 --allow-candidate-runtime --dry-run

# 独立容量查询，不分配显存或探测 OOM。
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4 --case main_memory-capacity --allow-candidate-runtime \
  --result-root base/result/p800-toolkit

# 双卡全量：显式选择 15 秒计时下限及五次独立进程。
python3 base/run.py toolkit run \
  --config base/configs/kunlunxin_p800_xpytorch29.yaml \
  --physical-device-ids 4,7 --allow-candidate-runtime \
  --repeat 5 --samples 50 --warmup 5 --minimum-seconds 15 \
  --compute-monitor on --data-movement-monitor on \
  --command-timeout 600 --timeout 14400 \
  --result-root base/result/p800-toolkit
```

真实命令也可先加 `--dry-run`。不指定 `--case` 选择全部 12 项；`--case` 可重复指定。
12 个 Case 的 `main.sh` 都将参数原样转发给同一个宿主入口，遵守相同权限和选卡条件。
Toolkit 容量是静态查询，不需要 Benchmark 分配/OOM Case 的 `--allow-high-risk-case`。

宿主执行者须能读写本轮结果、共享租约文件，并运行 Docker/fuser。
容器丢弃全部 capabilities，容器 root 不等于能写宿主用户拥有的 mode-755 目录。
`--privilege-command 'sudo -n'` 只为部分命令加前缀，不提升宿主 Python 的文件权限。
需要整体 root 执行的部署可在交互 `sudo -v` 后，在整个 `python3 ...` 命令前加上
`sudo -n env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"`。
其他部署按实际账户和目录权限配置；
不修改共享锁所有者/权限，不删除他人的锁。容器始终采用非 privileged、无网络、只读源码策略。

## 参数与默认值

| 参数 | 默认值 / 范围 | 含义 |
|---|---|---|
| `--config` | 必须选择 P800 host profile | 默认通用入口为 Ascend，P800 示例始终显式指定配置 |
| `--physical-device-ids` | 必填，唯一的 0–7 ID，支持逗号和范围 | 不接受 Ascend 的 NPU/logical Device 别名 |
| `--case` | 全部 12 项 | 选择唯一、已注册 Case；P2P 至少两卡才有 pair 测量点 |
| `--allow-candidate-runtime` | 默认关闭 | 真实运行 candidate 的显式允许，不晋级 runtime |
| `--repeat` | 5；1–20 | 每个逻辑点独立启动的进程次数 |
| `--samples` | 50；25–100000 | 每个计时进程最低样本数，全 Case 参数校验下限为 25 |
| `--warmup` | 5；0–100 | 每进程计时外预热次数 |
| `--minimum-seconds` | 0；0–120 | 显式计时下限；参考监控范围使用 15 秒 |
| `--matrix-size` | 8192；8–8192，8 的倍数 | 方阵 M=N=K |
| `--payload-bytes` | 536870912；1 B–1 GiB | H2D/D2H/D2D 带宽载荷 |
| `--p2p-payload-bytes` | 33554432；1 B–1 GiB | P2P 带宽；268435456 表示 256 MiB |
| `--pinned-api` | `register`，可选 `alloc` | 注册 Host 内存或调用 `xpu_host_alloc` |
| `--async-chunk-bytes` | 1048576；0–1 GiB | pinned async 分块阈值；0 请求整块提交 |
| `--command-timeout` | 600；5–600 秒，严格大于 minimum-seconds | 每个原生命令上限 |
| `--timeout` | 3600 秒 | 整个宿主执行预算；全量运行须显式增大 |
| `--compute-monitor` / `--data-movement-monitor` | `on`，可选 `off` | 计算/数据搬运的同窗监控 |
| `--result-root` | host config 的 result_root | 每轮在该根下生成唯一 run ID |
| `--privilege-command` | 空，或 `sudo -n` | 部分宿主命令的权限前缀 |
| `--allow-busy-devices` | 默认关闭 | 显式允许共享设备探索；不绕过身份、健康和租约 |
| `--smoke` | 默认关闭 | 矩阵 128、带宽载荷 1 MiB、25 samples、2 warmup、minimum-seconds 0 |
| `--dry-run` | 默认关闭 | 只输出静态 JSON 计划 |

当前 `EXCLUDED_PHYSICAL_IDS` 为空，计划输出 `excluded_physical_ids=[]`。
占用许可不终止其他进程，不授权 reset，也不能把共享测量当作空闲资源资格。
已知只读监控进程的例外需要显式授权并通过身份核对；普通占用仍按门禁处理。

容量不受 samples、warmup、minimum-seconds 的计时语义控制。`--smoke` 不缩短固定时延载荷：
H2D/D2H 保持四档扫描，P2P 时延保持 64 KiB。它不表示参考规模或正式资格。
P800 不接受 Ascend 的 `--latency-sizes`、HCCL 范围、legacy、privileged 或 DMI 开关。

## Case、载荷与公式

| Case | 每卡 / pair 逻辑点 | 测量方式 |
|---|---:|---|
| computation-FP32/FP16/BF16/INT8 | 每精度每卡 1 | XBLAS GEMM；前三项 TFLOPS，INT8 TOPS，均为 FP32 输出 |
| interconnect-h2d / interconnect-d2h | 每方向每卡 4 | pageable/pinned × blocking/async，默认 512 MiB |
| interconnect-h2d-latency / interconnect-d2h-latency | 每方向每卡 16 | 四模式 × 512 B、4 KiB、64 KiB、1 MiB |
| main_memory-bandwidth | 每卡 2 | XRE runtime memcpy 与独立 XBLAS Scopy |
| main_memory-capacity | 每卡 1 | SMI Total/Used/Free，MiB；静态查询 |
| interconnect-P2P_intraserver | 每无序 pair 2 | 升序单向及独立并发双向，默认 32 MiB |
| interconnect-P2P_intraserver-latency | 每无序 pair 1 | 升序单向，64 KiB |

载荷与点结构以 [toolkit-contract.json](toolkit-contract.json) 和 [contract.py](contract.py) 为准。
完整计划的逻辑点数是 `47*n + 3*n*(n-1)/2`；目标数再乘 repeat。
双卡五次是 97 点／485 目标，其中容量 10 次静态查询；八卡五次是 460 点／2300 目标。
双卡 475 个计时目标在 15 秒下限下需至少约 119 分钟，另有初始化、预热、回读及归档成本。
预留数 GiB 至十余 GiB 空间，实际规模取决于保留的全部样本数量。

计算用 `2*M*N*K / median_ns / 1000` 得 TFLOPS/TOPS；带宽用
`multiplier*payload_bytes / median_ns` 得十进制 GB/s。
runtime D2D、H2D/D2H、单向 P2P 的 multiplier 为 1；Scopy 为读+写 2；
并发双向 P2P 的分子为两份实际 payload，分母为两方向均完成的独立计时，含线程协调开销。
不得将单向结果直接乘二作为双向测量。

时延为单方向 API 提交至完成同步的主机 monotonic 耗时，不是 RTT 或纯 kernel 时间。
预热、Host 分配和正确性回读在计时之外；没有测量反向就不推导反向结果。

### D2D 两种路径

报告 Case 列分别显示 `D2D设备内存读写带宽` 和 `XRE runtime memcpy 拷贝吞吐`，每项一行。
各自保留原来的 `n=数量，最小值–最大值 单位，中位数 数值` 摘要格式，不混合两种模式。
其他列沿用整个原 Case 的状态、目标数量、耗时和限制，两行的计数及耗时不能相加。
其他 Case、明细和图表保持原格式；逐设备/模式/载荷/字段的独立重复统计在详情中。

`xpu_memcpy` 进入的是 runtime 的通用复制路径。它内部怎样安排传输、如何提交任务、
是否分块、怎样等待，都由 runtime 实现决定。

Scopy 则进入数学库的向量复制实现。这类设备内核通常可以让许多执行单元并行读取连续地址，
再并行写入目标地址，充分利用设备内存系统。

Scopy 按 2×payload 计费，作为设备内存读写带宽的实测参考；runtime memcpy 按
1×payload 计费，反映调用该 API 的拷贝吞吐。两者均含同步边界，都不是理论 HBM 峰值。
约 11 GB/s 和约 2.1 TB/s 可以同时成立，1×/2×不能单独解释差距；没有剖析内部实现时，
不能据差距断言主机中转、PCIe 路径或内部回退。

### pinned async 协议与保留失败

默认 pinned 内存通过 `posix_memalign` + `xpu_host_register` 准备，`alloc` 显式选择
`xpu_host_alloc`。该 candidate 在两条路径上均出现过 512 MiB pinned async 整块提交 `-999`。
默认以 1 MiB 分块：512 MiB 逻辑载荷提交 512 次，最后统一 stream wait；整个过程计时，
包含提交开销。指标保存 `async_chunk_bytes`、`api_calls_per_sample` 和 `submission_scope`。

`--async-chunk-bytes 0` 可请求历史整块路线，不自动回退为 blocking。
默认阈值下四档时延载荷均为单次提交；显式减小阈值可能使较大时延载荷分块，比较时须核对协议。
历史失败继续保持失败，pinned 时延双峰及 CV>5% 的重复组如实保留。

## 工具来源与实现范围

以下库相对于镜像内 `$CONDA_PREFIX/lib/python3.10/site-packages/torch_xmlir`；
每轮 `provenance.json` 保存实际路径与 SHA-256。

| 实现 | 来源 | 范围 |
|---|---|---|
| FP32/FP16/BF16 | `xhpc/xblas/include/cublas_v2.h`、`libxpu_blas.so` | `cublasGemmEx`，FP32 累加/输出，无 epilogue |
| INT8 | `xhpc/xblas/include/xblas_legacy_deprecated_api.h`、`libxpu_blas.so` | `fc_fusion<int8_t,int8_t,float,int8_t,float,float>`；maxima 127、alpha 1、beta 0、无 bias、LINEAR；FP32 输出 |
| 传输、时延、runtime D2D、P2P | `xre/include/xpu/runtime.h`、`xre/so/libxpurt.so.2` | memcpy/async/peer 原生 API 加完成等待 |
| Scopy | XBLAS 向量复制 API | 独立设备 kernel 读写路径 |
| 容量 | `xre/bin/xpu-smi` | 逐卡 Total/Used/Free，校验 PCI/UUID |
| 同窗监控 | 宿主 `xpu-smi`、`monitoring/kunlunxin_usage.py` | 所选设备的利用率、内存、温度及功耗原值；共享设备数值不能全归因本测试 |

INT8 GemmEx-int32 和 Lt 路线失败，不能把现行 fc_fusion 描述成它们成功运行。
厂商 `test_dma`、`test_memcpy_peer` 是独立对照工具；`fc_effciency/run_eff.py` 和
`perf_regression` 需要匹配的 XBLAS unittest，当前未提供该依赖，不声称完成官方效率流水线。
`xprofiler` 的可用性不等于已取得 kernel trace，普通运行不自动启用全局 profiler。

锁定兼容 runtime 的 `xpu_device_list` 未返回可用更新，因此 runner 从 device count 枚举
本地 ordinal、逐 context 读取 PCI，并在分配前核对完整可见集与宿主选择。
不能把 SMI 卡号直接当作原生 ordinal。

## 证据、状态和重建

每个 run 保存 summary、resolved-plan、code/image identity、源码快照、pre/postflight、
cleanup、lease、报告、SVG 和 SHA-256 索引。`toolkit-evidence/` 包含 manifest、
合同、工具来源、身份、编译记录、健康/拓扑/诊断、监控 JSONL，以及每个目标的 command、
stdout/stderr、samples 和 metrics。

- 测量层：原生命令成功、身份正确、正值有限样本、正确性与覆盖符合合同；容量按静态查询检查，Used=0 合法。
- 监控层：每个计时目标、每个参与设备至少 10 个完整位于原生计时窗内的有效样本，不把设置、预热或回读算入窗口；容量单独观察。
- 诊断层：XBLAS/XRE 正确性、合同覆盖和宿主健康观测组成合规校验；该工具链没有 DMI 等价阈值诊断。
- 三层全部 passed 才为 Case passed；short/partial/failed 保留，缺失监控不填零。
- 五次独立进程按完整测量点、字段与单位分组，样本 CV（ddof=1）>5% 为 unstable；进程内样本 CV（ddof=0）另报。

总览中其他 Case 的简短摘要仅用于概览，性能比较应使用相同设备/载荷/模式的详情重复组。
报告读取该 run 保存的排除策略，缺失则标“未记录”，不借当前配置推断历史。
Toolkit 结果保持 not-qualified，共享资源结果为 exploratory，均不会自动晋级 candidate。

```bash
python3 base/run.py report --result-root base/result/p800-toolkit --run-id RUN_ID
python3 -B -m unittest discover -s base/tests -p test_p800_toolkit.py
```

重建只改派生报告、报告元数据及索引，不补测缺失证据；冻结结果先复制再重建，记录生成器版本。
同一证据与生成器的重复渲染保持确定性。完整限制及历史验证范围见
[P800 验证说明](../../../../../docs/kunlunxin/validation.md)。
