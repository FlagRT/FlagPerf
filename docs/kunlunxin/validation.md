# P800 验证与维护

本文区分可执行范围、历史实测和当前代码的离线检查。runtime 保持
`candidate / validated:false`；单项测量通过不自动晋级镜像，也不构成理论峰值或全卡健康认证。

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
- Toolkit INT8 的 GemmEx-int32 与 Lt 路线在该 candidate 失败；现行使用
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
