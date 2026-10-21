# 平头哥 PPU Toolkit 使用说明

PPU Toolkit 提供 12 项独立微基准，直接调用锁定镜像内的官方 `comm_tools`
（sailbandwidth、p2pBandwidthLatencyTest）与 `libacblas`/`libacblasLt`，
另含逐卡静态容量查询。它不导入 PyTorch、不运行 Base workload，也不读取
Base 或历史性能值。环境与镜像准备见 [PPU 指南](../../../../../docs/thead/README.md)，
统一入口见 [Base 使用指南](../../../../README.md)。

## 从宿主运行

所有命令从 FlagPerf 仓库根目录执行，先安装 `base/requirements-host.txt`。
物理 ID 14、15 仅为示例；真实执行前确认本轮预约、空闲、健康、UUID/PCI、目录权限和存储空间。
Toolkit 以非 privileged 宿主进程运行，不需要 sudo。

```bash
# 只生成计划，不启动设备负载。
python3 base/run.py toolkit run \
  --config base/configs/thead_ppu_torch210.yaml \
  --physical-device-ids 14,15 --dry-run

# 独立容量查询，不分配显存。
python3 base/run.py toolkit run \
  --config base/configs/thead_ppu_torch210.yaml \
  --physical-device-ids 14 --case main_memory-capacity \
  --result-root base/result/thead-toolkit

# 双卡全量：五次独立重复，监控默认同窗开启。
python3 base/run.py toolkit run \
  --config base/configs/thead_ppu_torch210.yaml \
  --physical-device-ids 14,15 \
  --allow-candidate-runtime --repeat 5 --minimum-seconds 15 \
  --compute-monitor on --data-movement-monitor on \
  --result-root base/result/thead-toolkit
```

真实命令也可先加 `--dry-run`。不指定 `--case` 选择全部 12 项；`--case` 可重复指定。
12 个 Case 的 `main.sh` 都将参数原样转发给同一个宿主入口，遵守相同权限和选卡条件。

当前锁定镜像 `release_stage: candidate / validated: false`，因此每轮真实运行都必须
显式携带 `--allow-candidate-runtime`；这不改变发布声明，也不把候选镜像当作已验证 runtime。
宿主执行者须能读写本轮结果目录与共享租约文件。容器丢弃全部 capabilities
（`--cap-drop=ALL`）且 `--read-only`，容器 root 经 `--group-add` 获得宿主用户组，
证据目录以组可写模式（0770）挂载；容器内以 `umask 0000` 运行，宿主无需特权即可
在运行后补全 manifest 与监控指标。容器始终无网络（`--network=none`）、私有 IPC/PID、
非 privileged；只挂载选中卡的设备节点与 `/dev/alixpu`、`/dev/alixpu_ctl`。
每轮退出码：0=passed，1=failed，2=partial。

## 参数与默认值

| 参数 | 默认值 / 范围 | 含义 |
|---|---|---|
| `--config` | 必须选择 PPU host profile | 默认通用入口为 Ascend，PPU 示例始终显式指定配置 |
| `--physical-device-ids` | 必填，唯一的物理卡 ID，支持逗号和范围 | 不接受 Ascend 的 NPU/logical Device 别名 |
| `--case` | 全部 12 项 | 选择唯一、已注册 Case；P2P 至少两卡才有 pair 测量点 |
| `--allow-candidate-runtime` | 默认关闭 | 候选镜像必选；不改变 release 声明或验证范围 |
| `--repeat` | 5；1–20 | 每个逻辑点独立重复次数（与 P800 共用同一 CLI 注册） |
| `--payload-mib` | 512；1–32768 | sailbandwidth 缓冲（H2D/D2H/D2D） |
| `--p2p-bytes` | 40000000；1024–4294967296，4 的倍数 | P2P 传输字节数，与 day-9 校准口径一致 |
| `--matrix-size` | 8192；64–16384，8 的倍数 | 方阵 M=N=K（与 P800 共用同一 CLI 注册） |
| `--compute-samples` | 50；25–100000 | acblas 计时调用次数（中位数与 CV 的样本来源） |
| `--transfer-samples` | 25；5–1000 | sailbandwidth 每点迭代数 |
| `--p2p-samples` | 15；5–1000 | p2pBandwidthLatencyTest `--repeat` |
| `--warmup` | 5；0–100 | 计时外预热次数（与 P800 共用同一 CLI 注册） |
| `--minimum-seconds` | 0；0–120，严格小于 command-timeout | 每个计时点的窗口下限：达到样本数后窗口继续保持到下限（与 P800 共用同一 CLI 注册）；参考监控范围用 15 秒 |
| `--command-timeout` | 600；5–1800 秒 | 每个官方工具命令上限（与 P800 共用同一 CLI 注册） |
| `--compute-timeout` | 900；30–3600 秒 | 单个 acblas 子进程上限，含厂商 JIT 预热 |
| `--timeout` | 3600 秒 | 整个宿主执行预算；全量运行须显式增大 |
| `--compute-monitor` / `--data-movement-monitor` | `on`，可选 `off` | 计算/数据搬运的同窗监控 |
| `--result-root` | host config 的 result_root | 每轮在该根下生成唯一 run ID |
| `--smoke` | 默认关闭 | 64 MiB 传输、GEMM 256、25/5/5 采样、minimum-seconds 0；无正式资格 |
| `--dry-run` | 默认关闭 | 只输出静态 JSON 计划 |

P800 专属参数（`--samples`、`--payload-bytes`、`--p2p-payload-bytes`、`--pinned-api`、
`--async-chunk-bytes`、`--minimum-seconds`、`--allow-busy-devices`）对 PPU 无效，
显式给出即报错，不静默忽略。PPU 不接受 Ascend 的 `--latency-sizes`、HCCL 范围、
legacy、privileged 或 DMI 开关；`THEAD_ALLOW_FOREIGN_HANDLES` 由 toolkit 永不设置，
被占用卡一律拒绝（含 uncorrectable ECC 证据的卡 fail-closed）。
`--smoke` 不缩短固定时延载荷；它表示边界正确性冒烟，不是参考规模或正式资格。

## Case、载荷与公式

| Case | 每卡 / pair 逻辑点 | 测量方式 |
|---|---:|---|
| computation-FP32/FP64/FP16/BF16/TF32 | 每精度每卡 1 | 隔离子进程调用 acblas GEMM 条目，TFLOPS |
| interconnect-h2d / interconnect-d2h | 每方向每卡 2 | sailbandwidth `host_to_device`/`device_to_host` 的 CE 与 SM 路径，GB/s |
| main_memory-bandwidth | 每卡 1 | sailbandwidth `device_local_copy`，GB/s |
| main_memory-capacity | 每卡 1 | ppu-smi Total/Used/Free，MiB；静态查询 |
| interconnect-host_device_latency | 每卡 1 | sailbandwidth `host_device_latency_sm`，ns |
| interconnect-P2P_intraserver | 每有序 pair 3 | p2pBandwidthLatencyTest uni-write / uni-read / biBw，GB/s |
| interconnect-P2P_intraserver-latency | 每有序 pair 1 | p2pBandwidthLatencyTest GPU 时延矩阵，us→ns |

载荷与点结构以 [toolkit-contract.json](toolkit-contract.json) 和 [contract.py](contract.py) 为准。
完整计划的逻辑点数是 `12*n + 4*n*(n-1)`；目标数再乘 repeat。
双卡五次是 32 点／160 目标；四卡五次是 80 点／400 目标。
P2P 点按有序对展开（每个无序对的正反两个方向都实测，day-9 矩阵显示真实不对称，
如 42.83 对 46.14 GB/s），双向值不由单向推导或乘二得出。
computation-INT8 在当前 SDK 构建上被排除（`acblasGemmEx` int8 返回成功但输出错误、
`acblasLtMatmul` 触发厂商 JIT 崩溃，day-9 T8 证据）；排除项进入计划的
`excluded_capabilities`，报告原样引用，不改变其他 Case 的状态。

计算用 `2*M*N*K / median_ns / 1000` 得 TFLOPS；带宽直接取官方工具矩阵原值
（GB/s）；时延中 sail 访问 testcase 原生报 ns，P2P GPU 时延矩阵为 us×1000。sailbandwidth/时延工具值取自锁定 v2.1.0 构建的
testcase 表（索引与期望名见 `SAIL_EXPECTED_NAMES`，导入期断言锁定两表一致）。
CPU 侧指针追踪矩阵（`CPU P2P Transmission Latency Matrix`）不作为设备时延证据。

## 工具来源与实现范围

以下路径均在锁定镜像内；每轮 `provenance.json` 保存实际路径与 SHA-256。

| 实现 | 来源 | 范围 |
|---|---|---|
| FP32/FP64/FP16 | `libacblas.so` 的 `acblasSgemm_v2`/`acblasDgemm_v2`/`acblasHgemm` | 独立 ctypes 子进程，正确性回读后计时 |
| BF16 | `libacblasLt.so` 的 `acblasLtMatmul`（compute 32F） | 同上 |
| TF32 | `acblasGemmEx`（`ACBLAS_COMPUTE_32F_FAST_TF32`） | 同上 |
| H2D/D2H/D2D/访问时延 | `comm_tools/single_process/sailbandwidth` | JSON stdout，testcase 索引+期望名+单位三重门 |
| P2P 带宽/时延 | `comm_tools/single_process/p2pBandwidthLatencyTest`（`mpirun --allow-run-as-root` 时用 multi_process） | 矩阵解析，BDF→序号定位，`data matched` 校验 |
| 容量 | 容器内 `ppu-smi -q` | 逐卡 Total/Used/Free MiB，UUID 校验 |
| 同窗监控 | 宿主 `ppu-smi -q`、`monitoring/thead_usage.py` | 所选卡利用率、显存、温度、功耗原值；共享卡数值不能全归因本测试 |

设备身份门：宿主 ppu-smi 的 UUID/PCI/minor → 只挂载选中节点 →
工具自报 BDF 与子进程 `cudaDeviceProp` UUID 回读；三层一致才计 passed。
sailbandwidth 的通道数（CE/SM）与方向由 testcase 索引锁定，头行方向（WRITE/READ）
与 `(40000000 *2 Bytes)` 字节 token 参与逐矩阵校验。

## 证据、状态和重建

每个 run 保存 summary、resolved-plan、code/image identity、源码快照、pre/postflight、
cleanup、lease、报告、SVG 和 SHA-256 索引。`toolkit-evidence/` 包含 manifest、
合同、工具来源、身份、健康/拓扑/诊断、监控 JSONL，以及每个目标的 command、
stdout/stderr、samples 和 metrics。

- 测量层：官方工具/子进程成功、身份正确、正值有限样本与数据校验标记符合合同；容量按静态查询检查。
- 监控层：每个计时目标、每张参与卡至少 10 个完整位于真实测量窗内的有效样本，
  不把启动或预热算入窗口；容量单独观察（静态查询，无并发负载要求）。
- 诊断层：acblas 子进程正确性回读 + 官方工具数据校验标记 + 契约覆盖率校验 +
  ppu-smi 健康观测组成合规校验；该工具链没有 DMI 等价阈值诊断。
- 三层全部 passed 才为 Case passed；partial/failed 保留，缺失监控不填零。
- 五次独立重复按完整测量点、字段与单位分组，样本 CV（ddof=1）>5% 标记 unstable；
  稳定性结论不改变测量状态，也不扩大 runtime 的验证范围。

Toolkit 结果保持 not-qualified；这些是单次运行的资格状态。当前锁定 runtime 为
candidate 且未验证，只覆盖 [公开验证说明](../../../../../docs/thead/validation.md)
中的范围。报告读取该 run 保存的排除策略，缺失则标“未记录”，不借当前配置推断历史。

```bash
python3 base/run.py report --result-root base/result/thead-toolkit --run-id RUN_ID
python3 -B -m unittest discover -s base/tests -p test_thead_toolkit.py
```

重建只改派生报告、报告元数据及索引，不补测缺失证据；冻结结果先复制再重建，记录生成器版本。
同一证据与生成器的重复渲染保持确定性。完整限制及验证范围见
[PPU 验证说明](../../../../../docs/thead/validation.md)。
