# flagprof/vendors/ascend — 昇腾（Ascend / CANN）厂商适配器

## 1. 定位与职责

本目录是 flagprof 的**昇腾厂商适配层**，负责把 CANN / `torch_npu` 官方 profiler 产出的 Chrome trace
接入 flagprof 统一的"采集 → 导入 → 关联 → 台账"流水线。

设计边界：**厂商特定决策只留在这里**，通用流程全部放在共享模块
（`vendors/session.py`、`vendors/format.py`（`SourceFormat` 注入契约）、`vendors/import_common.py`、
`vendors/prepare.py`、`vendors/rules_common.py`、`vendors/native_common.py`、`vendors/disposition.py`
与 `flagprof.correlate` / `flagprof.store`）。
适配器由 `vendors.format.load_adapter(vendor)` 动态发现（校验 vendor 名并加载
`flagprof.vendors.<vendor>.importer`）；`prepare.py` 再按需加载 `<vendor>.disposition` 与
`<vendor>.native_rules`。适配器需在 `importer.py` 上暴露约定的入口面：
`device_taxonomy / default_registry / field_context / device_fields / build_rules`。

## 2. 文件总览

| 文件 | 角色 | 一句话职责 |
|---|---|---|
| `__init__.py` | 包标记 | 声明封存格式族 `TRACE_FAMILY = 'cann'`（本包不依赖 torch_npu 即可导入） |
| `session.py` | 在线采集 | `AscendProfiler` 后端：封装 `torch_npu.profiler` 的采集/导出/校验 |
| `capture.py` | 官方 raw 分析 | 将封存的 `sealed-raw/data` 复制到私有工作目录，调用 CANN 导出并检查日志和唯一的 `trace_view.json` |
| `execution.py` | 运行环境 | 声明 NPU 设备、运行时扩展及支持的通信后端，为各 rank 选择设备 |
| `telemetry*.py` | 设备监控 | 通过 DCMI 或 npu-smi 采集运行中设备的指标 |
| `contracts/` | 版本化格式契约 | 保存不同版本的角色、类别和已核验的来源约束 |
| `importer.py` | 离线导入 | 录制 trace 导入入口 + 适配器约定入口面（`device_fields` / `device_taxonomy` / `default_registry` / `field_context`） |
| `format.py` | 源格式分类 | 版本化 CANN 记录分类法 `Taxonomy`（v1 至 v4，默认 v4）：设备判定、类目、可归一化性、host_pid、launch 判定 |
| `rules.py` | 关联（fixture 规则族） | CANN 显式键关联规则：`connection_id`、Enqueue/Dequeue 桥、flow、同线程嵌套 |
| `native_rules.py` | 关联（原生归属） | 作用域级归属规则：cpu_op 嵌套 + 显式 Enqueue/Dequeue 桥（`cann-native-t8-v1`） |
| `disposition.py` | 台账分类 | 源账本分类器：完全由 legacy inventory 驱动，逐字保留原始类目 |
| `registry.json` | 证据登记 | profiler 控制类任务（`PROFILING_ENABLE/DISABLE`）的白名单证据与调查元数据 |

## 3. 各文件详解

### 3.1 `session.py` — 在线采集后端 `AscendProfiler`

实现 `vendors/session.py` 中 `ProfilerSession` 要求的 `ProfilerBackend` 协议：

- **`activity_source = 'npu'`**：会话编排层以此做同一进程内的活动源互斥（两个会话不能同时占用 `npu`）。
- **`__init__(hardware_identity)`**：可选注入硬件身份事实，但**必须带 `source` 字段**（否则 `ValueError`）；
  缺省记为 `status='unknown'` 并写明原因——体现"身份必须有出处"的原则。
- **`conflict()` → `'not_detectable'`**：当前安装的 torch_npu 没有可靠的活动会话查询接口；
  明确**不**通过启动一个重叠 profiler 去探测，如实上报不可探测。
- **`identity(config)`**：惰性 `import torch, torch_npu`；返回
  - `hardware`：`torch.npu.get_device_name()` / `current_device()`，`source='torch.npu public device API'`，附加注入的硬件身份；
  - `collector`：名称、torch_npu/torch/python 版本、activities=`['CPU','NPU']`；
  - `collector_prerequisites=[]`、`private_detection_symbols=[]`——会话层要求 prerequisites 为空，否则拒绝启动。
- **`set_destination(path)`**：用公共工厂 `tensorboard_trace_handler(path, analyse_flag=True, async_mode=False)`
  设定下一个官方捕获目录；仅在上一窗口 manifest 已 fsync 之后才被调用（顺序持久化）。
- **`create(config, on_ready)`**：构建 `torch_npu.profiler.profile`：
  `activities=[CPU, NPU]`，`schedule(wait=0, warmup=0, active=config.active, repeat=config.windows)`，
  透传 `record_shapes / with_stack / profile_memory`，`on_trace_ready=on_ready`。
- **`export(profiler, path)`**：调用上面保存的 handler 落盘。
- **`validate(path)`**：
  1. `validate_chrome(path, '**/ASCEND_PROFILER_OUTPUT/trace_view.json', …)`，
     设备记录判定为 `ph=='X'` 且所属进程名为 `Ascend Hardware` 或 `Communication`；
  2. 额外解析 `**/profiler_info*.json`，汇总 `config / cann_version / torch_npu_version` 到
     `actual_configuration`；解析失败记 `invalid_profiler_info`，一个都没有记 `profiler_info_missing`，
     二者都会把窗口判为 invalid。

### 3.2 `importer.py` — 离线导入入口与适配器入口面

模块自述："Thin recorded CANN import; **physical mappings are never inferred from pid**."

- **`device_fields(device, events)`**：为归一化后的设备事件提取 `dev` / `stream`：
  - `dev`：由 `process_labels` 事件建立 pid → label 映射，设备 pid 的 label 必须严格匹配 `NPU (\d+)`；
  - `stream`：取 args 的 `Physic Stream Id`（缺省回落 `tid`，存在但为 `None` 视为缺失）；
  - 任一缺失 → 抛 `UnrepresentableError(missing)`，该事件在台账中记为 `unrepresentable`（见 `vendors/disposition.py` 的 `reconcile`）；
  - 注释强调：返回的是**逻辑 profiler 标签，不是物理板卡/插槽号**。
- **`import_fixture(..., taxonomy_version=None)`**：默认读取同目录 `registry.json`，并把 `taxonomy_version`
  透传给共享导入事务；委托 `vendors.import_common.import_fixture(vendor='ascend', build_rules=…,
  device_fields=…, …)` 完成 prepare → O_EXCL 预约 → 派生文件 → writer 落盘 → barrier → completion。
- **`device_taxonomy(version=None) -> SourceFormat`**：惰性导入并返回 `format.Taxonomy`
  （缺省用 `format.VERSION`，当前为 `'4'`）——`prepare.py` 由此拿到分类法实例。
- **`default_registry() -> dict`**：加载 `registry.json`——registry 缺省值的统一入口
  （`prepare.py` 与 `import_common.import_capture` 均改走此处）。
- **`field_context(events)`**：只返回 `process_labels` 事件——告诉 `prepare.py` 提取
  `device_fields` 时无需扫描全部模型事件（`prepare.py` 中 `field_records = adapter.field_context(events)`）。

### 3.3 `format.py` — 版本化 CANN 记录分类法（`Taxonomy`）

实现共享的 `SourceFormat` 协议（`vendors/format.py`：注入式源格式契约，"no SDK or vendor-specific
classification"）。分类决策从冻结的通用流水线（`inventory.enumerate_raw` / `chrome.adapt`）**注入**到本模块：

- 默认版本为 v4，也可按记录读取 v1、v2、v3。v1 使用早期 CANN 的 compute 任务名，v2 起加入 `KERNEL_AICORE` 等新名称。
- `identity()` 对所选契约及其实现求摘要，并记录契约修订号。导入结果可据此核对当时使用的分类依据。
- `context()` 和 `is_device()` 使用源进程名和版本化的轨道角色识别设备事件。`category()` 依据通信名、`Task Type` 和已登记特例生成类别。
- `normalizable()` 检查设备轨道、排除的重复事件，以及契约是否接收具有来源时间的未知执行任务。v3/v4 的 `host_pid()` 使用已解码的宿主身份，早期版本沿用当时的 PID 映射规则。
- **`is_launch(event)`**：`name=='Node@launch'`，或 `ph=='s'` 且 `cat=='HostToDevice'` 的 flow 起点
  ——与 `rules.py` 的两条 launch 来源一一对应。

### 3.4 `rules.py` — 显式键关联规则（fixture 链接器）

模块自述："CANN explicit-key rules. **Multiplicity always reaches the union core**."

- **`correlation_keys(record)`**：只读取 `chrome.args.correlation_id`，经 `format.queue_key_valid` 校验为非负 `int`（不接受 bool、浮点数、字符串或其他别名）后，返回 `torch_npu.queue` 命名空间中的键。判定与账本、格式审计同源。
- **`build_rules(trace)`** 返回四条规则：
  1. `cann.flow`（`flow(trace,'cann','HostToDevice')`，device 侧）：利用 Chrome flow 事件
     （`cat='HostToDevice'` 的 `s/f` 相位 + `chrome.id`）把设备事件连回 host launch；
  2. `cann.connection_id`（device 侧）：设备事件的 `args.connection_id` ↔ 名为 `Node@launch` 的 host 记录，
     命名空间 `(trace_uid, connection_id)`；
  3. `cann.same_thread`（host 侧）：launch 被同线程 `op` 区间完整包住即命中；
  4. `cann.enqueue_corr`（host 侧，核心桥接）：对每个 launch 依次求——
     包住该 launch 的 `Dequeue*`（`ph=='X'`）区间 → 每个 dequeue 的每个 correlation key →
     同 `(trace_uid, host_pid, key)` 下的 `Enqueue*` 记录 → 包住该 enqueue 的所有 `op`（owner），
     产出 `Hit(owner, evidence=(dequeue, enqueue))`。
     **不要求 Enqueue 唯一**：多个 owner 全部保留，多义性交给上游 union core 裁决。

### 3.5 `native_rules.py` — 原生作用域归属规则

`VERSION = 'cann-native-t8-v1'`。与 `rules.py` 的桥接逻辑同源，但服务于**原生归属通道**
（`flagprof.correlate.native.relate` + 覆盖率统计），而非 fixture 链接器。

- **`build_native_rules(trace, config: ScopeConfig)`**：
  1. `native_common.classify` 把 host 记录分为 `calls`（`cpu_op` 的 X 相位）、`annotations`
     （`user_annotation` 类目或 `reuse/ cross_vendor/ spike/ ProfilerStep` 前缀）、`modules`（模块级）；
  2. `dequeues`：`ScopeIndex`（`Dequeue*` X 相位）；`enqueues`：按 `(trace_uid, host_pid, correlation_key)` 分组；
  3. 对每个 launch：包住 launch 的 Dequeue（`launch_interval` 处理 flow start 无时长的情况）→
     correlation key → Enqueue → 包住 Enqueue 的 call → `Hit(owner, (dequeue, enqueue))`；
  4. 返回两条 host 规则 + 分类结果：
     - `cann.native_nesting`（`nesting_rule`）：launch 被哪些 call 直接包住；
     - `cann.native_enqueue`：显式 Enqueue/Dequeue 桥的命中表；
     - `(calls, annotations, modules)` 供 `relate` 做原生归属与覆盖统计。

### 3.6 `disposition.py` — 源账本分类器

- **`classifier(events, inventory)`**：**完全由 inventory 驱动**（`prepare` 阶段 `enumerate_raw` 产出的
  loc → category 原始台账）：
  - 命中 → `(category, 'unrecognized_device_activity')`：**逐字保留** legacy 原始类目；
  - 未命中 → `(None, 'host_or_metadata')`：只作为源事件存在，不进入设备类目。
- 对比 `nvidia/disposition.py`：nvidia 版本还带 gpu track 启发式与 `gpu_user_annotation` 处理；
  ascend 版本不做任何 vendor 层再分类，台账保守。

### 3.7 `registry.json` — 证据登记表

- `version: 1`；`entries` 两条 profiler 控制类任务的白名单证据：
  - `profiler_enable`：`match: {task_type: 'PROFILING_ENABLE'}`；
  - `profiler_disable`：`match: {task_type: 'PROFILING_DISABLE'}`；
  - 每条都带 `basis`（"Explicit profiler-control task, not user compute or graph replay."）
    与 `evidence`（具体 `trace_uid` + `loc` 溯源）。
- `investigation`：调查元数据——候选 task_type、源 trace UID、
  `not_whitelisted`（`SDMA_SQE / allreduceAicpuKernel / graph_replay / unknown`）、原始 fixture 数 28。
- 用途：`correlate.linker.link(trace, rules, registry)` 的白名单/证据依据。
  注意：registry 缺省值统一由 `importer.default_registry()` 提供——ascend 返回本文件，
  nvidia 的同名入口返回 `{"entries": []}`（无证据登记）。

## 4. 端到端数据流

默认 `perf/modules` 使用 final 采集。目标正常结束后封存 raw。`analyse` 在采集目录所在的盘创建私有工作副本，调用厂商 SDK，检查退出状态、分析日志和导出文件，成功后封存 trace。分析日志写入采集目录，SDK 写入工作副本，`sealed-raw/data` 保留原始字节。已生成的封存 trace 可以在标准库环境中用 `flagprof import "$CAPTURE" --store "$STORE"` 导入，校验通过的检查点保存在目标 Store 盘。详见[离线存储说明](../../offline/README.md)。

下列流程对应显式窗口及录制样本的关联导入。

1. **在线采集**：`ProfilerSession` + `AscendProfiler` → `window-XXXX/ASCEND_PROFILER_OUTPUT/trace_view.json`
   + `profiler_info*.json`（session.json manifest 由会话层维护）。
2. **导入**：`importer.import_fixture`（或 `import_common.import_capture` → `import_fixture`）
   → `prepare_import`：
   `load_adapter('ascend')` → `device_taxonomy(taxonomy_version)`（分类法实例）
   → `stamp_events` → `enumerate_raw(taxonomy=…)`（inventory）→ `disposition.classifier` + `enumerate_source`（ledger）
   → `chrome.adapt(taxonomy=…)`（NormalizedTrace）→ `preserve_unbound_communication`（legacy 无 corr 通信标记兼容）
   → `rules.build_rules` + `link`（registry 缺省走 `adapter.default_registry()`）
   → `adapter.field_context` + `device_fields`（dev/stream，缺失即 `UnrepresentableError`）
   → `reconcile`（台账守恒校验）→ `native_rules.build_native_rules` + `relate`
   → `taxonomy.identity()` 记入源身份 `parser.taxonomy` → Writer 落盘
   （`ledger.jsonl / native.jsonl / summary.json / evidence-index.json / vendor_raw / traces / imports/completion.json`）。
3. **关联消费**：规则产出的 Hit 表（含全部多义命中）由 union core / `relate` 归并，
   得到每条设备活动的 host 归属与证据链（dequeue、enqueue、owner call）。

## 5. 关键设计决策

1. **物理映射绝不从 pid 推断**：`dev`/`stream` 必须有显式来源（`process_labels` 的 `NPU n`、
   `Physic Stream Id`），缺失即判 `UnrepresentableError`，不允许猜测；且明确区分逻辑标签与物理位号。
2. **多义性完整上交**：Enqueue 不要求唯一，所有 `(owner, evidence)` 命中全部输出，
   由 union core 统一裁决——vendor 层不做取舍。
3. **台账保守**：设备类目逐字沿用 legacy inventory 原始类别，vendor 层不引入新的启发式分类。
4. **分类法版本化、注入式、可追溯**：设备类目/可归一化性/launch 判定等分类决策集中在 `format.py`
   的版本化 `Taxonomy` 中，通过 `SourceFormat` 契约注入通用流水线（`enumerate_raw` / `adapt`），
   而不是散落在共享代码里；`identity()` 的 sha256 随 trace 持久化到 `parser.taxonomy`，
   事后可精确知道每条 trace 用哪套分类依据导入；v1 使用旧任务名，v2 起同时识别新旧两代 CANN Task Type。
5. **只走官方公共接口**：`torch_npu.profiler` 公共工厂 + `torch.npu` 公共 device API；
   不探测私有符号（`private_detection_symbols=[]`），无前置依赖（`collector_prerequisites=[]`）。
6. **SDK 惰性导入**：torch/torch_npu 只在方法内部 import，本包本身可在未装 SDK 的环境导入
   （离线导入 fixture 不依赖 torch_npu）。
7. **不可探测即如实上报**：`conflict()` 返回 `'not_detectable'`，不通过副作用探测。

## 6. 与 nvidia 适配器的差异速览

| 维度 | ascend（本目录） | nvidia |
|---|---|---|
| 源格式分类（`format.py`） | CANN v1 至 v4：按进程名 + `Task Type` 分类 | Kineto 契约：按 Chrome `cat` 分类 |
| host→device 关联键 | 显式 `correlation` 参数（key 名含 correlation）+ `connection_id` | `cuda_runtime/driver` 的 `correlation` + `External id` |
| 桥接机制 | `Dequeue*` → `Enqueue*` 显式桥 + `HostToDevice` flow | `ac2g` flow |
| dev/stream 来源 | `process_labels` `NPU n` + `Physic Stream Id` | 设备事件 args 自带 |
| host_pid 还原 | 按版本解码 CANN 来源 PID | pid 原样 |
| `field_context` | 仅 `process_labels` 事件（避免全量扫描） | 全部事件 |
| disposition 策略 | 纯 inventory 驱动 | 额外 gpu track 启发式 + `gpu_user_annotation` |
| registry.json | 有（profiler 控制任务证据） | 无（`default_registry()` 返回空 entries） |
| 采集后端 | `torch_npu.profiler` | `torch.cuda` profiler |
