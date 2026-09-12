# Operation：统一算子测试入口

`run.py` 使用命令行选择厂商、设备、case、实现注册路径及规模，不读取 host.yaml，
不初始化 SSH。现有 52 个 case 的输入构造与算子调用仍由各自 `benchmarks/<case>/main.py`
提供。此工具测量和验证当前运行栈；底层库未实现的算子保留失败证据，不补写底层 kernel。

## 失败诊断（工具层）

`run --diagnostics failures` 为默认行为；`--diagnostics off` 关闭附加诊断，不关闭原正确性或路由门禁。
附加诊断只针对数值失败或路由证据不足的组合，在独立 worker 中执行；诊断耗时计入端到端耗时，
不进入性能计时区间。已知执行异常保存调用栈并分类，不重复尝试或自动更换实现路径。

- `numeric-diagnostic.json`：基于保存的相同输入，分别比较 probe／measure 输出与 CPU FP64、CPU FP32；
  包含超阈值比例、误差分位数、最大阈值归一化误差及对应元素，输出和输入梯度分别标注。原阈值不变。
- `route-diagnostic.json`、`route-trace.json`：保存独立诊断调用的 CPU profiler 事件和 ATen 父子关系。
  CPU API 事件不等于设备 kernel 证据，不能仅凭 linear/addmm 事件将 partial 升为 passed。
- 缓存 kernel 路径：probe 同时识别 Triton compiled runner 和 Ascend launcher，并记录 launcher 的
  `compile_only`／`register_tensor_only` 状态。两个状态明确为 false、设备输出与同步正常时，才将
  缓存 launch 纳入路由证据；缺失状态、仅编译、仅登记或只有分配事件都不能据此通过。
- `result.json`：增加 `failure_stage`、`diagnosis` 和附加 `diagnostics` 结果；附加诊断失败不覆盖原门禁。
  后端注册、RNG 状态布局、SPLIT_K 参数和标量重载异常必须结合调用栈识别为 dependency-blocked。
  blocked 仍然未解决，分类改变不计为适配通过。

报告同时列出异常、数值失败和路由 partial。不同诊断模式／协议及显式运行配置不合并覆盖，旧证据缺少
字段时按历史记录保留。当前未提供已证实可控制此 Torch-FL 路径的 FP32 数学模式开关；不将通用 PyTorch
精度设置冒充 Ascend 生效配置。底层库及原 case 语义保持不变。

## 使用

在 FlagPerf 目录执行：

```bash
# 列出 52 个 case 和本工具支持的输入类型，不需要设备库。
python3 operation/run.py list

# 查看默认计划：全部 case、一个路径、每个 case 一种适用 dtype。
python3 operation/run.py run --vendor ascend --device-ids 14 --dry-run

# Ascend：匹配 CANN 9 的锁定镜像；先检查租约和空闲设备，再开始测试。
python3 operation/run.py run --vendor ascend --device-ids 14 --allow-privileged-root

# 两个指定 case、两条注册路径；更小规模 smoke。
python3 operation/run.py run --vendor ascend --device-ids 14 \
  --case abs --case mm --oplib both --dtype FP16 --profile smoke --allow-privileged-root

# 相同 case 多个规模，按输入维度字段指定；打印实际 shape 和有效配置。
python3 operation/run.py run --vendor ascend --device-ids 14 --case mm \
  --size M=128,N=256,K=128 --size M=512,N=512,K=512 --allow-privileged-root

# 完整验收矩阵：类型不适用项明确记为 not-applicable。
python3 operation/run.py run --vendor ascend --device-ids 14 --oplib both \
  --dtype FP32 FP16 BF16 INT32 INT16 BOOL INT64 --allow-privileged-root

# 容器内运行：使用已准备好的运行栈，device-ids 是当前容器可见编号。
python3 operation/run.py run --vendor ascend --execution local --device-ids 0 --case abs

# 同一个公共入口也支持其他厂商；硬件验证范围见下文。
python3 operation/run.py run --vendor nvidia --image <vendor-image> --device-ids 0 --case mm

# 根据原始证据离线重建报告；文件缺失或哈希不符时拒绝生成。
python3 operation/run.py report --run-dir operation/result/<run-id>
```

`--case` 可重复，省略时选择全部 52 项；`--case all` 仅选择名为 `all` 的归约算子。
默认 `--oplib nativetorch`，在 Ascend 上对应 Torch-FL。显式 `flaggems` 会调用
`flag_gems.enable()`；两条路径各自独立进程，不能仅凭入口名称声称底层实现不同。

默认 dtype：浮点 FP32、位运算 INT32、`all` 保留原有 INT64 输入。显式 `--dtype`
不自动转换不适用类型，而是在覆盖表中注明不适用。默认不是全精度或双路径测试。

`--profile daily` 与 `smoke` 均采用短测配置；smoke 使用更小的维度／元素单位和更少迭代。
`--size` 字段对应 case 自身配置，例如 `Melements`、`M,N,K`、`bs,channel,hiddensize`。
包含原 1024 元素单位的 case 可用 `ELEMENT_UNIT` 控制这个单位；公式同步使用实际单位。
没有长规模 profile。每个组合约两分钟是软目标，超出不会自动失败。`--watchdog`
默认 300 秒／阶段，仅处理卡死；批量总耗时单列。首次 worker 初始化与首次编译会增加时间。

`--spectflops` 为当前选定设备粒度和精度的手动峰值，不自动猜测；未提供则 FU 为 N/A。
`--warmup`、`--iters`、`--rounds` 可显式覆盖，所有最终值进入 `plan.json` 和各项 `task.json`。

## 执行与测量边界

- 每种设备、注册路径分别使用 CPU reference、带跟踪 probe、无跟踪 measure worker。
  worker 可跨 case 复用；输入 seed、参数和每项日志独立。首次调用延迟不是干净编译缓存下的编译耗时。
- CPU 进程保存实际量化输入及模块参数，用 FP64 参考检查输出；测量进程复用相同输入文件。
  前向／反向的测试选择沿用原 case；反向正确性使用非均匀上游梯度，原反向计时使用每个 case 缓存一次的零梯度协议。
- 浮点检查记录固定 dtype 容差和误差，整数／布尔精确比较。dropout 检查缩放、保留率与掩码一致的梯度；零输入的掩码不可由零输出推断，按两种合法梯度检验。
  不要求 CPU 和设备共享随机序列。这里是本次输入的正确性检查，不是逐层误差归因系统。
- cold/warm latency 为原协议的前向调用；主机批次含提交、分配、同步等待，带反向 case 还含
  forward、sum 和 backward。不计入输入搬运、CPU 参考和路由跟踪。
- 当前 Ascend adapter 不提供纯 kernel 时间；报告明确 `kernel_status=not-supported`。
  原 case 的等效工作量公式保留，反向乘 3 为估计；整数／布尔不标作 TFLOPS。
- 目标区间观察到 CPU fallback 判为路由失败；测量进程中的 fallback 单独记录，因可能来自准备工作，
  将结果降为 partial。仅输出位于设备或仅观察到 transpose 等辅助调用，不足以证明目标计算路由。
- 实际目标设备编号、包版本、源码快照、输入哈希、路由日志、正确性及计时结果均保存。
  `passed` 需要正确性与目标路由证据；`partial` 表示证据不足；`blocked` 表示确认的底层注册缺口；
  `failed` 为执行或正确性失败；未执行和不适用项单独保留。

当前浮点检查阈值如下；整数／布尔输出精确比较。`passed` 表示本次输入在这些固定阈值下通过，
不代表全输入空间或全部数学模式的精度资格认证。没有因本轮 FP32 矩阵失败而放宽阈值。

| 输入类型 | atol | rtol |
|---|---:|---:|
| FP32 | 1e-5 | 1e-4 |
| FP16 | 2e-3 | 2e-2 |
| BF16 | 2e-2 | 8e-2 |

## 多批次汇总

```bash
python3 operation/run.py report --run-dir operation/result/<run-a> \
  --run-dir operation/result/<run-b> --output-dir operation/result/<new-summary>
```

按提供顺序取同一 host、镜像、设备、case、dtype、路径、seed 和规模的最新实际尝试；失败会保留，
未运行或中断项不会抹去已完成结果，不同规模也不会混成一项。每个来源校验 SHA-256，原始实验不改写。
汇总目录必须是新目录；保留来源目录后才能再次离线重建报告。跨镜像／主机及未记录镜像身份的
local 结果不混合汇总。各来源的代码快照单独保留，汇总结果不能被描述为同一次进程运行。

## 厂商扩展与文件管理

| 位置 | 职责 |
|---|---|
| `runtime/` | 无设备库依赖的规划、生命周期与报告，以及通用 worker |
| `benchmarks/<case>/main.py` | 原有算子及输入构造，`build_case` 被旧入口和新 worker 共用 |
| `vendors/<vendor>/` | 厂商初始化、环境、设备映射、同步、计时能力与路由证据 |
| `vendors/compat.py` | 现有其他厂商 Torch 路径的兼容接口 |
| `legacy/` | 旧 host.yaml / SSH 集群入口，仅显式调用 |
| `result/<run-id>/` | 每次实验独立配置、源码、原始日志和报告，禁止覆盖历史实验 |
| `tests/` | CLI、覆盖、输入、公式、资源清理和报告契约 |

增加厂商时实现 `bootstrap/synchronize/kernel_time`、运行环境、Docker 设备参数、
运行身份、租约目录、preflight 与路由判断，并登记到 `vendors`。不能在公共 CLI 中加入厂商分支。
Ascend 复用 Base 的锁定运行身份与设备租约，未通过 host 配置文件间接传参，也未改变 Base 执行器。

Docker 模式的设备编号是宿主编号，worker 只看本次映射的设备；结束时清理本次创建的容器并检查设备。
`local` 模式保留现有可见设备映射，只能约束当前命名空间内的租约。容器所有者负责宿主设备预约和
运行镜像身份；报告如实注明这项验证边界，不能将其当作宿主 preflight 通过。

其他厂商目前保留兼容能力，未进行本轮硬件验证。NVIDIA 提供宿主空闲检查；其他兼容厂商
在未实现宿主 occupancy 检查前使用已预约环境的 local 模式。未知环境不会默默启用 CUDA。
旧集群方式仍可通过 `legacy/cluster_run.py` 显式使用，与本入口的短测配置和验证状态不混用。
