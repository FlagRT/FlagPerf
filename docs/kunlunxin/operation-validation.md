# P800 Operation 验证与已知问题

[返回 P800 Operation](../../operation/vendors/kunlunxin/README.md) · [机制与维护](operation-design.md)

本页区分框架能力、底层算子结果与验证范围。Operation 保持 `validated:false`；
底层失败不因“有覆盖”而通过，未实现功能也不据“可能是厂商问题”豁免。

## 已接受的记录限制

`P800-TRACE-001`：个别 FP16 Gems div/tanh 首次调用有主机提交，但缺少对应设备 kernel
事件。原数值检查可通过，缓存窗口可完整；发生频率未量化，根因尚未确定。
2026-10-21 项目维护者接受该间歇记录限制，暂缓专项追查，**不再以此单项阻塞评估适配 PR**。
这是项目提交决定，不是上游已批准、故障已修复或设备性能已验收。

原始 trace、失败和严格 `partial` 判定保留；以后再次发生仍按证据不足记录，不重试刷绿，
缺失调用的设备时间/路由不作为已验证结果。PR 审查处置为 `accepted-known-limitation`，
与程序原始状态分列，不能把该决定传播到共享性能、数值失败或六组计数器状态，
也不自动晋级 runtime。完整能力验收仍以本页未完成项为准。

## 当前未完成项与责任边界

| 项目 | 已确认事实 | 处理与责任边界 |
|---|---|---|
| 六组硬件计数器 | 当前适配器没有可信事件采集器/语义映射，均 not-supported；timeline 可单独成功 | 是工具接入缺口，不是硬件不支持的证明。需要工具配套配置、信号含义/单位、采样/缩放和归因规则，再完成解析与正负例；不能冒填同名 Ascend 指标 |
| 个别首次路由证据缺失 | 主机 launch 存在，但原 trace 缺相应 kernel 事件；缓存调用可能完整 | 设备仍可用且数值可能正确；缺的是可核验的执行记录。已按 P800-TRACE-001 接受为评估 PR 的已知限制、暂缓专项追查；原始证据不足判定保留，不能断言厂商责任或刷绿 |
| 原生 FP32 矩阵超差 | mm/addmm/bmm 超出原公共容差；原输入复放仍超差，CPU FP32 对 FP64 辅助对照也有少量超差 | 原生 PyTorch 在 P800 调用厂商后端，浮点算法/累加次序会有差异。只确认不满足现有数值门禁，尚未证明是具体 kernel 错误或阈值不合适；保留输入/输出，交后端团队定位，不放宽容差 |
| Gems 编译/运行失败 | 存在 uni_sram 编译资源错误、LLVM bad signature 断言、invalid device function、autotune 除零等 | 有真实编译器/运行栈证据；FlagGems、Triton/FlagTree、厂商后端的最终责任仍需定位。框架继续负责数据、调用、身份和记录正确，不自行改算子或规避失败 |
| 独占性能 | 现有完整矩阵来自共享授权，尚无独占资格 | 真实取得空闲设备、句柄与物理租约后才可验收；SMI 零利用率不足以证明独占。先调查框架问题，再在独占窗口执行完整 Docker/local 验证 |
| local 全量 | 七个代表 case 各两条路径，共 14 个 case/路径组合，尚非全矩阵 | 代表覆盖包含反向与多类型，但不能代替 308 个适用组合；后续完整 local 验证与 Docker 分开记账 |

### 本轮不要求的环境范围

当前没有 Ascend 硬件，不安排 Ascend 实机回归；只能报告已有静态/CPU 回归范围。
“新环境分发未验证”指其他团队在另一台主机取得镜像、安装依赖并复现，不是说已经存在一台待测试机器。
当前只承诺已验证主机与锁定栈的范围；跨主机安装、公共 registry 获取和裸宿主安装未验证，不能据此宣称通用复现完成。
这些边界与当前主机 Docker/local 能否执行分开，不凭空增加机器或把它们称为现有执行 bug。

## 2026-10-20 提交后独立复测

运行代码为 `08039482df7aaf1adb435abc1dfce0fb5d58f6d7`，源码指纹为
`c95f13296401cad87623380619ed388514df057f855dd8b62b39c9b1f8692e03`。
后续文档整理不改写本轮或更早实测证据，以下分母不混入补充诊断。

| 范围 | 结果与边界 |
|---|---|
| Docker daily | 52 case、七种 dtype、双路径 728 项；420 not-applicable、275 partial、27 failed、6 blocked；308 适用项全部尝试 |
| 路由 | 286 passed、1 partial、21 缺失；缺失项不能算通过 |
| allocator memory | 275 measured、33 skipped，不是整卡 HBM 流量 |
| Docker full 代表 | abs/linear/relu × 三种浮点 × 双路径共 18 项；路由、memory、timeline 均成功；六组计数器各 not-supported，任务仍 partial |
| local 代表 | 七个 case 双路径 daily/timeline，共 14 项：11 partial、2 failed、1 blocked；路由 12 passed、2 缺失；在外层持宿主租约的自建锁定容器中运行，不是裸宿主安装或全矩阵 |
| 数据核验 | 991 份输入/模块/设备审计 passed，307 份已完成测量的实际输出存在且输入哈希一致；不表示输出数值全部通过 |
| 生命周期 | 已检查 run 的本轮 worker 消失、postflight、租约释放正常；自己的 profile-target watchdog/SIGTERM 回收验证通过 |
| 离线报告/回归 | 聚合重建确定性与原始哈希核验通过；Operation 127、Base 256（1 skip）、Ascend Toolkit 80、公开分发 13 项通过；P800 81 为 Operation 子集，不重复累加 |

本轮 partial 路由为 **Gems div FP16 首次调用**：有主机 launch，trace 中无对应 kernel，缓存调用完整。
历史一轮的 Gems tanh FP16 也出现过相同类缺事件，但本轮 tanh 路由通过；两轮原记录分别保留。
先前说明将历史 tanh 写入本轮段落，现按批次更正，未修改原 trace 或任务状态，也没有据另一次成功宣称根因已解决。

原生 mm FP32 同输入 replay 仍为 82/262144 个元素超过原阈值；Gems mm FP16 replay 仍遇到 uni_sram 编译错误。
diagnose 总状态 passed 只表示检查覆盖/执行完成，必须同时读取 correctness 和原任务状态。
三个 container exited/no such object 的原始日志另观察到 LLVM `CallInst::init` 的 bad signature 断言，原自动分类与 failed 状态保留；
这说明编译阶段发生断言，不足以证明完整根因或排除全部框架因素。

## 2026-10-20 晚间定向诊断

在 `502848f6` 上，对 Gems div/tanh FP16 各执行三次独立 run，共六项，使用原 daily 工作量、seed 和容差。
六项首次/缓存设备关联均 passed，probe/measure 数值均 passed；清理、postflight 和物理租约释放核验正常。
这是三次预先限定次数的诊断复测，不是失败重试直至成功；原失败 trace 的主机 launch 有记录但 kernel 缺失，
新结果不覆盖旧 partial，也不足以证明间歇缺事件已修复。未修改 profiler、预热、算子或门禁来改变结果。

SMI 中 0/1/4/7 显示零显存/利用率，但句柄检查发现其他训练进程仍持有节点；这些候选卡未通过独占检查。
六项诊断继续使用已授权的共享模式，全部任务总体仍 partial，不产生独占性能结论；未开展 Docker/local 完整验收。
宿主进程可见性、句柄和 UUID 观察均单独记录，不将持卡训练进程归为只读观察者，也不干预其运行。
后续顺序为：先继续定位事件采集缺口及框架错误，再重新检查独占资源，最后分别完成 Docker 和 local 全矩阵。

## 同日其他验证与较早评估

以下均为各自批次事实，不并入上面的提交后复测分母，也不回写旧证据。

### 同日较早批次与补充检查

较早批次曾记录 Gems tanh FP16 首次 trace 缺 kernel、缓存完整；同输入 replay 通过。该条不属于上表最新提交后矩阵。

无 `.git` 的纯源码副本也实际执行锁定本地镜像的 abs FP32 双路径，正确性/路由通过、共享性能 partial；
这不代表公开 registry、跨主机镜像获取或裸宿主安装已验证。

relu 三种浮点双路径的 6 组观察器开/关对照使用同保存输入，两侧正确性通过、kernel 名称及数量相同。
该检查只覆盖所列 6 组，不是全算子观察中性证明。profile 子进程实际执行时分别触发 watchdog 与 SIGTERM，
本轮子进程和容器均消失、postflight 与租约释放正常；中断及采集失败状态原样保存。
最终原生 abs、数值失败的原生 mm FP32、编译阻塞的 Gems mm FP16 均实际同输入 replay；后两项原异常仍存在。
diagnose 的 passed 表示公共检查覆盖完整，仍须读取其中 correctness failed，不能解读为算子通过。
最终三批离线聚合及再次重建确定性已核对，report 退出 0 不改变失败任务。


### 2026-10-16 profiling 补充验证

2026-10-16 的新增验证均使用共享授权，输入与运行身份逐项核验，不计入历史全矩阵分母：

| 范围 | 任务结果 | 附加阶段 |
|---|---|---|
| Docker abs FP32 双路径 smoke/full，2 项 | 2 partial | 2 项 memory/timeline measured；六组硬件指标 not-supported，整体 profiling partial |
| Docker abs/add/sum/mm/native_dropout 三种浮点类型双路径 daily/full，30 项 | 23 partial、4 failed、3 blocked | 23 项 memory、20 项 timeline measured；3 项原生反向 timeline 当时缺少跨线程关联，原记录保留 |
| 修正跨线程关联后，Docker abs/sum/native_dropout 三种浮点类型双路径 daily/timeline，18 项 | 15 partial、3 failed | 15 项 memory/timeline measured，包含原生 native_dropout 的三种类型；Gems 失败项不补采 |
| local abs/sum/native_dropout FP16 双路径 daily/timeline，6 项 | 5 partial、1 failed | 5 项 memory/timeline measured；Gems native_dropout 测量失败 |

local 验证在调用者自建的锁定镜像容器中执行，由外层持有宿主 UUID 租约，不代表裸宿主安装或 local 全矩阵验收。
以上独立批次保留各自源码身份；新解析器不回写旧报告。没有取得独占性能资格，不晋级 Operation runtime。

### 2026-10-13 最小入口验证

2026-10-13 的共享卡最小验证范围：

| 组合 | 数值链路 | 目标路由 |
|---|---|---|
| Docker/local abs、add FP32，原生 | probe/measure 通过 | partial：缺少具体原生目标派发证据 |
| Docker/local abs、add FP32，FlagGems | probe/measure 通过 | 首次及缓存 XPU launcher 证据通过；共享性能仍 partial |
| Docker/local mm FP16，原生 | probe/measure 通过 | partial |
| Docker/local mm FP16，FlagGems | 编译受阻，未产生有效测量 | 锁定栈的 `OutOfResources: uni_sram PassManager::run failed` |

上述 mm 异常只有同时匹配异常类型、FlagGems mm 与 Triton XPU `make_ttxir` 的真实调用栈才分类 blocked；
仅错误关键词不能分类，未知异常仍 failed。没有修改库、缩小 smoke 规模或改变容差来绕过失败。
以上 2026-10-13 最小验证不代表全量验收；后续覆盖范围见下节，仍不据此晋级 Operation runtime。

### 历史评估（2026-10-14）

以下为锁定栈上 Docker / 单卡共享授权模式的实际评估；local 的代表验证和补充路由复测范围另列于下文。

| 范围 | 组合数 | 结果 |
|---|---:|---|
| 52 case 默认类型 smoke | 104 | 2 blocked、8 failed、94 partial |
| 低精度代表补测 | 6 | 1 blocked、2 failed、3 partial |
| 10 个异常 case 独立双路径复测 | 20 | 2 blocked、8 failed、10 partial |
| 29 case daily，七种 dtype 双路径 | 406 | 232 not-applicable、174 partial |
| 15 case daily，七种 dtype 双路径 | 210 | 124 not-applicable、71 partial、9 failed、6 blocked |
| 8 case daily，七种 dtype 双路径 | 112 | 64 not-applicable、30 partial、18 failed |

daily 的 406 项包含 174 个适用组合和 232 个 catalog 类型不适用项，涵盖算术、激活、比较及位运算 29 个 case。
使用公共默认 warmup=5、iters=20、rounds=10、seed=2026；新增第二批涵盖归约、矩阵和三角操作。
三批共完成 52 case / 308 个适用组合的尝试，另有 420 个类型不适用项，无未运行项。
原始 daily 覆盖账目为 275 partial、27 failed、6 blocked、420 not-applicable；独立复测不重复计数。
三批保留各自源码修订，不是同一源码版本的全量通过认证；后续交付审查仍需按身份核对。
这 174 个 daily 适用项的 probe 与 measure 正确性均通过；执行清理也已确认。
partial 需结合逐项正确性和路由读取；共享计时不用于独占性能资格，运行栈仍为 validated:false。

10 个 smoke 异常 case 经独立双路径 run 复现，原状态和失败阶段一致。除 mm/mv 的 uni_sram 编译阻塞外，
FlagGems addmm/bmm 出现 LLVM bad signature 断言，cross_entropy_loss 出现 kernel -714 异常，
layer_norm 出现 Triton autotune 计时除零，log_softmax/native_dropout/native_group_norm/softmax 出现
invalid device function。log_softmax 还存在 probe 数值门禁失败。低精度补测保留 mm FP16、layer_norm BF16、
native_dropout FP16 的对应异常。以上记录错误发生位置，不把它们当作已查明的最终根因。
复现可使用 [P800 最小命令](../../operation/vendors/kunlunxin/README.md#环境与最小运行)，明确指定对应 --case/--dtype/--oplib，并使用新的结果目录；不降低尺寸或容差来改写结论。

第二批 daily 的 86 个适用组合中，71 个 probe/measure 数值通过，3 个原生 FP32 组合（addmm、bmm、mm）
在两个进程均超过公共容差，另 12 个 Gems 组合在 probe 执行或编译失败。CPU FP32 对 FP64 的辅助对照也存在少量超差，
因此只确认未通过门禁，不据此认定最终根因。Gems addmm FP32、bmm FP32/FP16 为 LLVM bad signature 断言；
addmm FP16/BF16、bmm BF16 为 uni_sram 编译失败，因不匹配已审阅的 mm 调用栈分类规则，仍保留 failed。
mm/mv 三种浮点类型符合既有 mm 编译栈规则，保留 blocked。没有通过关键词扩展 blocked 范围。
上述 12 个异常 case/dtype 组合均进行了独立双路径 daily 复测，共 24 项；状态和失败阶段全部复现，
输入哈希、配置与运行身份一致。原生 FP32 addmm/bmm/mm 的超差元素数分别为 89/166/82；容差未放宽。

路由门禁补充了实际 kernel 名称和最近目标实现校验，并用嵌套辅助 kernel、未知/缺失/错误目标名称验证负例。
新规则只读核对既有 290 条已保存路由记录，状态没有变化；原始报告未改写。
Docker abs/add 的 FP32/FP16/BF16 双路径 smoke 与 daily 各 12 个适用组合，local smoke 同样 12 个组合，均完成数值检查；
Gems 目标路由通过，原生路由和共享性能仍 partial。另复测 Docker mm FP32/FP16 与 local mm FP16，保留 Gems 编译阻塞。
local 七个代表 case 的双路径 smoke 已实际尝试：abs FP32、mm FP16、sum BF16、bitwise_and INT32、all INT64、
layer_norm BF16、native_dropout FP16，共 14 项：11 partial、2 failed、1 blocked。
失败为 Gems layer_norm / native_dropout，阻塞为 Gems mm；其余 probe/measure 数值检查通过。
native_dropout 含反向。验证在调用者自建的锁定镜像容器内执行 local CLI，由外层 harness 持有宿主 UUID 租约；
不是裸宿主运行栈安装证明，也不改变 local 只管理当前命名空间的边界。七份报告分别保留，不自动聚合。
补测不重复计入 daily 覆盖分母。
修订前后的数据保留各自源码身份；没有把新判定写回旧证据，后续聚合应保留对应源码分组。

第三批 daily 覆盖 group_norm、native_group_norm、layer_norm、softmax、log_softmax、cross_entropy_loss、
dropout、native_dropout 的 FP32/FP16/BF16 双路径。24 个原生项及 Gems group_norm/dropout 的 6 项完成
probe/measure 数值检查，路由与共享性能仍 partial。其余 18 个 Gems 项 failed：

| Case（各含三种浮点类型） | 原始失败阶段与现象 |
|---|---|
| native_group_norm、native_dropout | measure：invalid device function |
| layer_norm | probe：Triton autotune ZeroDivisionError |
| softmax | probe：invalid device function |
| log_softmax | probe 数值失败，随后 measure：invalid device function；两项记录均保留 |
| cross_entropy_loss | probe：xpuLaunchKernel("empty_kernel") 返回 -714 |

上述 18 个 case/dtype 均以新 run 独立双路径 daily 复测，共 36 项：18 partial、18 failed；
状态与失败阶段全部复现，输入哈希和配置一致。复测中另有一次 softmax FP16 run 因 Docker 清理查询超时被停止，
该异常 run 原样保留、排除于有效复测账目；随后确认本轮容器消失、设备健康和租约释放，用新 run 补齐 FP16 及尚未执行的 FP32。
另一次 native_group_norm BF16 复测在 probe 的 docker inspect 查询超时，与原始 kernel 错误不同，
也单独保留并用新 run 补测，不算作原异常的复现。未放宽清理门禁。这些是观察到的异常位置，尚不能认定最终根因。
沿用公共 build_case：native_group_norm、softmax、log_softmax、cross_entropy_loss、native_dropout 含反向；
group_norm、layer_norm、dropout 为前向。模块参数从输入包恢复；正确性反向采用非零上游梯度。
dropout 检查缩放、保留率及掩码对应梯度，不要求 CPU 与设备随机序列相同，也不把原计时的零梯度当作完整反向正确性证明。
第三批及 local 的清理、postflight、租约释放均确认；不改变公共 workload、容差、FlagGems 或 Triton 来规避失败。

### 2026-10-16 同输入复放和故障诊断

2026-10-16 使用保存输入实际复放了原生 abs FP32、原生 mm FP32 和 Gems mm FP16。
abs 数值检查通过而路由仍 partial；原生 mm 再次出现 82/262144 个元素超出原容差；Gems mm 再次在 probe 编译报 uni_sram 错误。
三项输入哈希未变化，postflight 与租约释放记录完整。诊断总状态描述检查覆盖/执行状态，须同时读取原任务状态和各项检查；
例如原生 mm 的诊断执行完成不等于数值通过。复放只重做 reference-check/probe 及适用诊断，不重跑 measure，不能用它证明测量阶段异常已复现。
另以相同 FP16 输入比较显式传入 `CUDART_DUMMY_REGISTER=1` 与现行环境，两侧均为 2 partial、1 failed、1 blocked。
bootstrap 记录分别为未设置和 1，但两组成功测量的运行身份中该变量均为 1；失败测量未产出完结身份。
这只比较了是否在启动时显式设置，不能视为整个运行期间恒定的 0/1 对照，也不能排除初始化时机或底层作用。
没有据此修改正式 profile 或宣称修复。

native_dropout FP16 的测量栈进一步定位到反向 `grad_output.contiguous()` → 昆仑芯 FlagGems `copy_` → ATen redispatch。
独立诊断把 FP16 标量 0/1 展开为 `[1,1024,1024]` 后转为连续张量：原生两项完成且值正确，Gems 两项均在同一 copy_ 转派发位置报 invalid device function。
这将问题缩小到展开布局的复制路径；异步异常的最终底层原因尚未证明，也不能外推所有失败均由此引起。
未通过改为连续上游梯度、改计时协议、禁用注册或替换实现让原 benchmark 通过；诊断结果不计入正式覆盖分母。

### 2026-10-20 路由与计数器接口诊断

针对历史上首次调用偶发缺少设备 kernel 事件的问题，使用锁定镜像、原 daily 输入和共享授权，
对 Gems `div`/`tanh` FP16 各执行两次独立 run；四个任务的首次和缓存窗口均通过 host launch、
correlation、`ac2g` 流和设备 kernel 的严格关联。本批启用 CUPTI TRACE 日志，日志可能影响采集时序，
不用于性能资格。该结果只说明本批次未复现，不能覆盖历史 partial，
也没有重试至成功、放宽门禁或改变 profiler 生命周期。严格解析器现在在 partial 结果中写出
`missing_calls`、`missing_correlations` 和 `launches_without_device_kernel`，以便区分“主机提交存在但设备记录缺失”的具体关联。

六组硬件计数器仍未实现。full profiling 额外对运行时已加载的 CUPTI 兼容库做只读输出契约探针；
若 `cuptiDeviceGetNumMetrics` 和 `cuptiDeviceGetNumEventDomains` 均在两种不同哨兵值下返回成功却不写调用方输出，
报告记录 `counter_interface.status=not-supported`，六组仍逐项 `not-supported`。这不是硬件不支持或零丢样的结论，
也不把 CUDA 兼容 trace 字段当作 P800 计数器；完成六组仍需要厂商事件配置、信号单位、采样缩放和归因定义。

增加诊断字段后的独立 daily/full 复测两项中，div 首次/缓存均通过，而 tanh 首次再次缺少
反向 launch correlation 30 的设备 kernel；前向 correlation 26 和缓存窗口完整，probe/measure 数值均通过。
原始 partial 与缺事件 trace 完整保留。这表明间歇缺事件尚未解决，不能由前四项未复现得出修复结论。
收紧探针为只查询设备 metric/domain 接口后，最终代码的两项 daily/full TRACE 诊断关联及数值均通过，
timeline 均 measured、六组仍 not-supported；这同样不能抵消前述 partial。未使用空 context 的
dropped-record 查询证明零丢样，也不把接口枚举探针当作六组实际采集。


### 2026-10-20 原生事件追查：本轮未复现，未解决

保持原 daily 的 Gems div/tanh FP16，在共享授权下完成固定次数对照：原生过滤函数后保存记录、
预分配 mmap 保存过滤前后记录、CUPTI 事件读取阶段保存返回记录，各六次独立 run；
另有不加载诊断库且不开启 TRACE 的两次独立 run。共 20 run、40 任务、80 个首次/缓存窗口，
本轮路由及 probe/measure 数值检查均通过。所有 run 整体仍 partial，内层 CLI 退出 2，
共享条件没有独占性能资格；外层退出 0 只表示调度与封存完成。

两组原生观察各捕获 24 个 buffer、60 条 kernel 记录，本轮无记录被原生过滤器剔除；
mmap 组过滤前后原始字节一致。读取阶段的 60 条记录均按 PID/correlation 对应最终 trace。
诊断介入可能扰动时序，以上只描述成功窗口，不能排除原失败曾发生采样遗漏或过滤。
锁定 Kineto 存在 outOfRange 时间范围过滤，但没有证据证明它导致历史缺失。
目标调用后已有设备同步，本轮没有添加睡眠、预热、重试或绕过过滤。

原生、转换和最终 trace 尚未在同一个失败窗口同时捕获，因此根因仍未确定，原 tanh FP16
首次反向 correlation 30 缺设备 kernel 的 partial 保留，不据本轮成功声明修复。
本轮执行代码为 3ca12671，生产代码未改，Operation 仍 validated:false；六组计数器仍未完成。
四批归档名称依次为
p800-native-buffer-20261020T145815Z、p800-native-mmap-20261020T150603Z、
p800-converted-20261020T151702Z、p800-uninstrumented-20261020T152232Z；
完整 artifacts、postflight、容器消失和租约释放均核验通过。

### 2026-10-21 联合采集：链路已核验，原缺失未复现

在同一进程同时保存原生过滤器入口/出口的记录字节、保留偏移、CUPTI 返回的 kernel
及 enable/disable/flush 生命周期，再按进程、采集会话和 correlation 对应最终 trace。
helper 只在本轮诊断容器加载，不安装进 runtime，不改变原 API 参数、返回值和记录。
这是原生过滤器入口之后的观察，尚未直接观察设备端采样；原生入口缺失时仍须调查
设备采集、缓冲和停止采样，不能直接判定硬件或某个库负责。

更正比较范围：10 月 20 日分批原生/转换/无诊断库对照未带 `--profiling full`，
与原失败调用条件并不完全一致。本轮恢复 `--profiling full`，原 daily Gems div/tanh FP16、
diagnostics off、TRACE 关闭；预先限定最多八个独立 CLI run，发现不完整路由窗口即停止。
八次均未复现：16 任务、32 首次/缓存窗口通过，48 条路由关联与独立 timeline 的
280 条提交关联均有原生输入、保留记录、CUPTI 返回和最终 kernel，逐条一一对应。
24 个进程、48 个采集会话，无记录槽溢出或无效快照；376 条原生记录均保留且过滤前后字节一致。

完整外层 2512 文件 SHA-256、八个内层 artifacts、probe/measure 正确性、postflight、
容器消失及租约释放已核验。内层八次仍 partial/退出 2，外层退出 0 仅表示调度和封存成功；
共享卡无独占性能资格，timeline measured，六组计数器仍 not-supported。
归档名为 `p800-chain-full-20261020T163022Z`
（UTC 10 月 20 日，北京时间 10 月 21 日）。执行代码 e2f42fd3，运行源码指纹仍为
`b341b549aeb5c22e63c8bdfa799b312308734282de8b461770e688439e3bd988`。

诊断会影响时序；本轮未取得失败同次的逐层证据，不能给原缺失分配责任或声明修复。
原 tanh FP16 首次反向 correlation 30 缺 kernel 的 partial 与原始文件不变，
Operation 保持 validated:false。诊断 helper 和分析脚本随项目本地证据单独归档，
不作为已发布生产入口；不据成功复测晋级，也不扩大本轮到全量或独占验收。
