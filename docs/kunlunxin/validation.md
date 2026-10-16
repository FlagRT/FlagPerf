# P800 验证与维护

本文区分可执行范围、历史实测和当前代码的离线检查。2026-10-10 经维护者确认，锁定
runtime 设为 `release_stage:validated`、`validated:true`、`validation_status:partial`，
采用与 Ascend 相同的“已验证但范围受限”约定。此声明基于下文 2026-10-08 Base 与
2026-10-09 Toolkit 的既有记录，本次发布调整没有重新运行硬件测试。

机器可读的 [验证摘要](../../base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/validation-summary.json)
绑定 image ID、范围和来源摘要哈希；manifest 记录该摘要的 SHA-256，运行时拒绝哈希或身份不符。
它是可分发的维护者验证声明，原始证据仍单独保存。profile 中的 `_candidate` 为兼容旧配置
保留；当前镜像无需 `--allow-candidate-runtime`，未验证的新镜像仍须显式授权。

`validated:true` 不表示所有重复组稳定、所有卡健康、所有维度完成验证或镜像可匿名拉取。
范围限于 native Base 的计算/传输/内存、已记录的单机通信组合，以及双卡 12-case Toolkit；
不覆盖 Training、Inference、Operation、FlagGems、跨机和未实现精度。新镜像不能沿用该声明。
单次运行仍独立判定：Toolkit 的 `not-qualified`、`exploratory` 和 `unstable` 不因发布状态改变。

### Operation 独立评估记录（硬件：2026-10-14；离线核对：2026-10-15）

Operation 不继承 Base/Toolkit 的 `validated:true`。在锁定镜像和 P800 共享授权下，三个源码身份对应的三批 Docker daily 共覆盖 52 个 case 的 728 个唯一 case/dtype/oplib 组合：308 个适用、420 个类型不适用；适用项保留 275 个 `partial`、27 个 `failed` 和 6 个 `blocked`。local 另有 7 个代表 case 的 14 项双路径 smoke：11 `partial`、2 `failed`、1 `blocked`。这些数字是评估覆盖和失败记录，不是 Operation runtime 晋级或单一源码版本的全算子通过声明。

离线检查按三个源码身份分别重建报告，核对源 evidence 哈希、清理/租约字段、公开复放参数和报告确定性；不同源码指纹没有跨组混合。公开测试覆盖 P800 专用 profile 的 Docker/local 示例、全类型分母、多设备/尺寸规划、文档链接和标准库负例；无设备回归不替代硬件复测。原生路由证据、独占性能资格、设备 timeline/full profiling 和底层失败仍有限制，Operation 保持 `validated:false`。完整入口、失败阶段和边界见 [昆仑芯 Operation 指南](../../operation/vendors/kunlunxin/README.md)；原始运行结果不随公开源码分发。

2026-10-16 新增独立 memory 诊断采集及 XPU CUPTI 设备 timeline。完成 Docker 2 项 smoke、30 项 daily、修正反向跨线程关联后的 18 项 daily，
以及 local 6 项 daily 代表验证；后两批分别有 15/5 项 memory 与 timeline measured，Gems native_dropout 的 3/1 项测量失败保留。
sum 浮点前向新增两阶段目标路由证明；原生全目标路由和通用 fallback 证明仍有缺口。
另实际执行 partial、数值失败、编译阻塞三类来源的同输入 replay，确认原生 mm FP32 数值超差和 Gems mm FP16 编译错误仍存在。
各批输入/运行身份和清理分别核验，详情见厂商指南。原任务不因附加采集晋级；full 的六组硬件指标没有已验证的采集器和语义映射，明确记录 not-supported。
CUDA 兼容 trace 中的 occupancy、grid/block、SM 等未经验证字段不作为 P800 硬件计数器。历史统计与新采集范围分开记录，不改写旧结果。
历史报告及 `qualification-record.json` 保留当时的 candidate 状态，不回写为新验证结果。

## 无设备检查

使用 Python 3.11 或更新版本，先安装 [宿主依赖](../../base/requirements-host.txt)。
下列检查不需要 Docker、设备或本机历史结果目录：

```bash
python3 -m pip install -r base/requirements-host.txt
python3 -B -m unittest discover -s tests -p test_public_distribution.py
python3 -B -m unittest discover -s base/tests -p test_p800_toolkit.py
python3 -B -m unittest discover -s base/toolkits/_common/ascend/A3/tests
python3 -S -B -m unittest discover \
  -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py'
```

Base 全量离线回归另需 CPU PyTorch 与 PyYAML，使用维护者准备的 CPU 测试环境：

```bash
python3 -B -m unittest discover -s base/tests -p 'test_*.py'
```

缺少依赖、跳过测试或仅通过 dry-run 时，不得记录为完整回归通过。
runtime 单元测试使用公开的最小测试夹具，不读取私有 evidence。Base 全量套件另有一项
可选的历史 smoke 报告重放测试；未提供该归档时会明确 skipped，应单独记录，不能计入通过数。
公开分发检查覆盖两家厂商入口、P800 Case 合同及文档链接；也应在不含 `.git`、
原始结果、私有 evidence 和相邻项目的源码导出目录运行。正式测量还需 Git 工作副本记录源码身份。

## 状态与指标判定

| 层次 | 含义 |
|---|---|
| Base 单 run | 分别记录执行、测量、正确性、监控、postflight、cleanup 和报告状态 |
| Base 五次资格 | 固定 case、配置、源码、镜像和 UUID 集合；各 rank 计时不少于 15 秒；监控、正确性、资源及清理门禁通过 |
| Toolkit 测量 | 原生命令、正值有限计时、身份、正确性和合同覆盖；容量按静态查询规则检查 |
| Toolkit 监控 | 每个计时目标、每个参与设备至少 10 个完整落在原生测量窗内的有效样本；容量单独观察 |
| Toolkit 诊断 | 厂商工具链合规校验：正确性回读、合同覆盖和健康观测；没有 DMI 等价的厂商阈值诊断 |
| 重复稳定性 | 五个独立进程/run 的样本标准差（ddof=1）除以均值；CV > 5% 为 unstable，均值为零时不定义 CV |

Toolkit 进程内样本 CV 使用总体标准差（ddof=0），不能代替五次独立进程 CV。
`passed`、重复稳定、健康和 runtime 晋级是不同结论。
`MODE=measured` 的 Base 单 run 不强制与 qualification 相同的 15 秒拒绝条件；
`qualification.py` 仍检查时间下限，不能凭单 run passed 推断五次严格资格通过。

Base 计算使用同步墙钟计时；CUDA Event 在锁定栈的早期探针中返回零，不用于 Perf 计时。
Benchmark 与 Toolkit 的 INT8 算子、输出 dtype、D2D 实现及容量含义不同，比较前必须对齐协议。
Toolkit 总览采用简短 n/范围/中位数，D2D 按两种模式分为两行；
其余 Case 的总览只用于概览，逐设备、载荷、方向和请求模式的结论应读取详情中的独立重复组。

## 历史实测范围

以下为固定镜像和配置下的历史结果摘要，原始 evidence/result 不随源码分发。
它们不是当前 checkout 的新硬件复测记录；如需独立复算，应取得对应完整证据或重新测量。

| 日期与范围 | 已确认结果 | 不能外推的范围 |
|---|---|---|
| 2026-10-08 Base 最终测量 | 114 run 的测量/正确性通过；110 passed、4 partial；22 个重复组中严格检查 15 passed、2 unstable、5 rejected | 不是全部组严格通过；其中七卡组仅三次 |
| Base 计算 | 8192³；FP32 118.4729 TFLOPS、FP16 259.1654 TFLOPS、BF16 118.9696 TFLOPS、INT8 504.5260 TOPS，均为对应五次组中位数 | BF16 吞吐限制保留；INT8 为设备算子、BF16 输出；不认证内部 IEEE 实现或硬件峰值 |
| Base 设备内存 | 预分配 copy 读写带宽中位数 2235.9549 GB/s；单次实际持有容量 98,132 MiB | 容量短观察窗为 partial，不参加吞吐五次资格；不是 Toolkit 的静态容量查询 |
| Base 通信 | 2/6 rank AllReduce 与 2 rank P2P 有严格五次资格记录 | 七卡重复不足；八卡测量和稳定性通过但历史健康 degraded，严格资格拒绝 |
| 2026-10-09 Toolkit | 双卡、12 Case、485 执行目标，测量/监控/诊断三层通过 | 101 个重复组中 81 stable、20 unstable，不能称全部稳定；结果仍 not-qualified |

Base 的三组 D2H pageable/nonblocking 历史组存在不足 15 秒的窗口，项目交付接受了这一
限定例外；其中两组 CV≤5%，初组 CV 7.158% 仍为 unstable。该决定没有修改严格脚本、
原始时长或健康门禁，也不授权未来运行忽略时间下限。

## 已知限制与保留失败

- Base 早期 `torch._int_mm` 走主机 CPU 的 INT8 数字已撤回。现行使用
  `xtorch_ops.gemm_I8_I8_bf16_nt`，INT8 输入、显式缩放、BF16 输出。
- Toolkit INT8 的 GemmEx-int32 与 Lt 路线在该锁定镜像上失败；现行使用
  XBLAS `fc_fusion`、INT8 输入和 FP32 输出，不把失败路线改写为成功。
- Toolkit 的 512 MiB pinned async 整块提交在 register/alloc 两种 API 下均曾返回 `-999`；
  默认按 1 MiB 分块后统一等待。Base 的整块 `Tensor.copy_` 是另一协议，历史 4 GiB
  pinned nonblocking error 999 仍保留，不能把 Toolkit workaround 写成 Base 实现。
- pinned H2D/D2H 小载荷时延可能出现双峰，部分重复组 CV>5%。保留全部样本，
  不删除慢样本、拼接不同 run 或称其稳定。
- `xpu_memcpy` 通用复制吞吐与 Scopy 设备内存读写带宽不可合并中位数，计费分别为
  1×/2×payload。吞吐差距本身不能证明主机中转、PCIe 路径或内部 fallback。
- FP64/FP8/TF32 未形成当前栈支持的执行路线；跨机不在单宿主执行器范围内。
- XPULink 性能差异尚无已证明的唯一原因；环数或链路标称值不能代替实际通信协议测量。
- 历史 Base 索引有一份归档验证汇总与旧索引不匹配，原记录未重签覆盖。
  历史完整性异常与最终 run 的测量校验分别说明，不宣称全部历史索引零失配。

## 维护和重新验证

更改镜像、驱动、shape、载荷、dtype、rank 数、拓扑或计时方式时，重新确认对应范围。
先做静态计划，再在已授权、空闲且健康的设备上做功能检查；容量和通信分别安排资源。
保持测量、监控、正确性、清理、重复性及资格结论独立。

每次修改同时核对 [Base 指南](../../base/README.md)、[环境指南](README.md)、
[Toolkit 合同说明](../../base/toolkits/_common/kunlunxin/P800/README.md) 和受影响的 Case README。
原始数据冻结后，报告审计在临时副本执行并记录生成器版本；不能为通过审查修改原始失败或状态。

发布前检查待发布版本及历史中的站点信息、凭据和大型证据。当前版本取消跟踪不等于历史清除；
公开代码、镜像可获取性、离线兼容和硬件实测应分别给出结论。
