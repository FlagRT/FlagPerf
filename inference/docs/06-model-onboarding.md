> 本文从 v2 设计工作区迁入，保留设计、阶段裁决与历史状态。当前代码入口及两人协作方式见 [文档导航](README.md)，可执行命令见 [使用说明](usage.md)。未发布的本地实验与任务书出处见删除前固定 Git 提交中的 [历史引用索引](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/history-references.md)，其出处保留。

# 06 语义层与模型接入

> 上游：[02 架构](02-architecture.md) L3/§7、[03 schema](03-event-schema.md) §4.7/§4.9 ｜ 下游：[07 分析器](07-analyzers-and-reports.md)、[09 CLI](09-cli-and-campaign.md)

要解决的问题：**500 个模型，接入成本必须接近零；但"逐层"又需要知道层在哪里。**

答案是：逐层语义有三个来源，自动的那一个覆盖绝大多数模型，手写的那一个只在需要精度差分时才用。

## 1. scope 的三个来源

| 来源 | 机制 | 覆盖面 | 需要用户做什么 |
|---|---|---|---|
| **自动** | module 探针遍历 `nn.Module` 树，按规则选层打 hook | 任何 PyTorch 模型 | 无 |
| **声明** | Tier-1 profile YAML 指定边界名与选择器 | 需要精度差分的模型 | 约 20 行 YAML |
| **手工** | `flagprof.scope("name")` 上下文管理器 | 引擎内部阶段、自定义循环 | 几行代码（通常写在插件或引擎适配里） |

三者产出的都是 `span(cat=module/user/engine)` 事件，分析器不区分来源，只看 `cat` 与 `name`。

### 1.1 自动选层规则

全量打 hook（每个 `nn.Module`）会产生巨量 span 且开销不可控。默认规则：

1. 深度不超过 `max_depth`（默认 4）；
2. 跳过"透明容器"（`nn.Sequential`、`nn.ModuleList` 本身，但进入其子项）；
3. 跳过叶子算子型 module（`nn.Linear`、`nn.LayerNorm` 等单算子包装）—— 它们的信息已由 `op` 事件覆盖；
4. **重复结构折叠**：`model.layers.0..31` 这类同构层，默认只对前 `n`（默认 2）层与最后 1 层打 hook，其余层归入一个聚合 span。可用 `--layers all` 关闭折叠。

> 规则 4 是 500 模型场景的关键：一个 70 层的模型若全打 hook，span 数量与开销都乘 70，而前两层的数据通常已足以定位问题。折叠状态写入 `coverage`。

规则参数可被 profile 覆盖。

## 2. Tier 0：零配置能得到什么

对任意一个 HuggingFace 风格的模型目录，`flagprof record --target hf:<dir> --task <type>` 即可得到：

- 算子清单与完整签名（`op` + `sig`）→ R1 的模型—算子映射、动态 shape 分布
- FlagGems 命中/回退清单与原因（`route`）→ R47、R18/R22
- 编译事件与缓存命中（`compile`）→ R45
- 设备 kernel 耗时与归因（按 [04 §6](04-correlation.md) 的可达级别）
- 通信与显存、主机/设备 counter
- 按自动 module 树的逐层耗时归因
- health 与失败现场

**得不到**：跨芯片数值差分（需要封存输入，见 §5）、语义层边界对齐的张量比对（需要 Tier 1）。

## 3. 任务类型（task）

`--task` 决定内置运行器如何构造输入与执行，是 Tier 0 能工作的前提。覆盖 R92–R97 的五批模型：

| task | 输入构造 | 执行 | 对应批次 |
|---|---|---|---|
| `embedding` | tokenizer 编码文本（内置语料或用户给定 jsonl） | 单次 forward，取 pooled 输出 | R92 语义向量 |
| `rerank` | query-doc 对 | 单次 forward | R93 重排序 |
| `multimodal` | 图/文对，按 processor 编码 | 单次 forward | R94 多模态检索 |
| `generate` | prompt + 生成参数 | `generate()` 或逐步 decode，每步打 `span(cat=step)` | R95 生成式与 RAG |
| `ctr` | 稀疏特征（变长 id 列表 + 稠密特征），按 schema 生成或读样本 | 单次 forward | R96/R97 广告 CTR/CVR |
| `custom` | 由 Tier-2 插件提供 | 同上 | 其他 |

每个 task 实现为 `runner/tasks/<task>.py`，是**通用**的，不绑定任何具体模型。接入新模型优先靠"选对 task + 调参数"，而非写代码。

## 4. Tier 2 插件：从 35 个方法瘦身到 5 个

v1 的 `ModelAdapter` 有 35 个方法（[01 §2.1](https://github.com/FlagRT/FlagPerf/blob/941e06f022521e48c89fecac2abe078960a6715a/inference/docs/history/01-diagnosis.md)）。其中资产校验、输入编码、模型加载、参考包契约、报告文案五类职责全部上移到框架。插件只保留**框架真的猜不到**的部分：

```python
class ModelPlugin(Protocol):
    def load(self, ctx) -> Any:
        """返回可调用对象。默认实现用 transformers.AutoModel，多数模型不需要覆写。"""

    def make_inputs(self, ctx) -> Iterable[dict]:
        """构造输入批次。默认实现由 task 提供。"""

    def forward(self, model, batch) -> Any:
        """执行一次。默认实现为 model(**batch)。自定义生成循环在这里。"""

    def boundaries(self) -> list[Boundary]:
        """声明层边界（精度差分用）。也可以在 Tier-1 YAML 中声明，不写插件。"""

    def describe(self) -> dict:
        """模型元数据：名称、版本、权重哈希、任务类型。"""
```

五个方法**全部可选**，各有默认实现。预期 90% 以上的模型停留在 Tier 0/1，不写插件。

## 5. Tier 1：profile YAML 与 call_key

### 5.1 profile 示例

```yaml
# profiles/qwen3-embedding-0.6b.yaml
model:
  name: qwen3-embedding-0.6b
  task: embedding
  path: ${MODELS}/Qwen3-Embedding-0.6B

inputs:                      # input recipe —— seal 模式的输入封存依据
  recipe:
    kind: text
    source: corpus://zh-mixed-256     # 内置语料标识，或 file://...
    n: 32
    max_len: 512
    seed: 20261009
  # 生成结果会被写入 run 目录并记 sha256；对照 run 直接复用同一份文件

layers:
  max_depth: 3
  fold_repeats: 2
  boundaries:                # 精度差分的对比点
    - name: embed_out
      at: model.embed_tokens
      capture: out
    - name: block0_out
      at: model.layers.0
      capture: out
    - name: block_last_out
      at: model.layers.-1
      capture: out
    - name: pooled
      at: __output__
      capture: out
  mask:                      # 变长输入时排除 padding
    from: attention_mask

capture:
  stats: true                # 默认
  blob: false                # 需要时由 CLI --blob <boundary> 覆盖
```

约 20 行即可获得逐层精度差分能力。`at` 支持 module 路径、负索引、`__output__`、`op:<name>` 四种选择器。

### 5.2 call_key：跨 run 对齐张量的确定性键

> **2026-10-28 按 [D50](DECISIONS.md#d50) 第 6、7 条（P4 首批口径）**：
> - `seq_in_scope` 为该 module 在本 run 内第几次被调用；`sample_idx` 取固定输入的样本 id。
> - 算子边界用 `op:<name.overload>`，不用 `op:<sid>`，因为 sid 跨 torch 版本是否稳定没有验证；sid 仍写在事件上，供与 `perf.op` 连接。
> - 配对后另用 `src` 检查两侧来源是否互相配对，误配记 `misaligned_producer`。

Oracle–Device 差分的本质是 **run_A 的某个张量 ⨝ run_B 的对应张量**。join 键必须在两台不同厂商的机器上确定性一致，因此**不能**包含时间、地址、设备名、corr。

```text
call_key = <scope_path> # <seq_in_scope> # <boundary> # <sample_idx>
例：  model.layers.0 # 3 # block0_out # 7
```

| 分量 | 来源 | 为什么确定 |
|---|---|---|
| `scope_path` | span 树从根到当前 span 的 `name` 拼接 | module 路径由模型结构决定 |
| `seq_in_scope` | 该 scope 内第几次经过此边界 | 生成式模型同一层被调用多次（每 token 一次），靠它区分 |
| `boundary` | profile 中声明的名字，或 `op:<sid>` | 显式声明 |
| `sample_idx` | 批内样本下标 | 输入封存后固定 |

**失配即报告，不猜测**：若 run_A 有而 run_B 没有某个 `call_key`（例如国产芯片上该层走了不同分支），分析器输出"仅 A 存在"的清单，不做任何近似匹配。

### 5.3 seal 模式的输入封存

> **2026-10-28 按 [D50](DECISIONS.md#d50) 第 2 条**：CLI 的 `seal` 已用于封存外部导出（[D46](DECISIONS.md#d46) 第 7 条），本节的 `record --seal` 改称“固定输入模式”，参数为 runner 的 `--inputs-from`。P4 首批复用 v1 A100 参考包 `inference/oracles/qwen3-bf16-v6-20260929/` 的固定输入；按 `inputs.recipe` 生成固定输入留到后续批次。

```text
flagprof record --profile profiles/qwen3-embedding-0.6b.yaml --seal --out runs/
  1. 按 inputs.recipe 生成输入 → 写入 <run>/inputs/batch-*.pt，记 sha256
  2. 采集 identity（包版本、环境变量、设备、代码闭包哈希）
  3. 正常 attach + 跑
  4. manifest.mode = "seal"，identity.inputs.sha256 填入
```

对照 run 用 `--inputs-from <run_A>` 直接复用 A 的输入文件（不重新生成），这样连随机数实现差异都被排除。

分析时：`flagprof report oracle --ref <run_A> --cmp <run_B>` 先做 identity diff，差异必须被两次 run 的 `experiment` 声明解释，否则拒绝出结论（N4）。

## 6. 引擎接入（vLLM 等）

服务化引擎没有"一次 forward"的概念，scope 来自引擎内部阶段：

- `vendors`/`engines` 下提供可选的引擎适配器，在已知版本的 vLLM 上打 `span(cat=engine, name=prefill/decode/schedule)`；
- 适配器失效（版本不匹配）时**自动跳过**，退化为只有 module 树与 op，不中断 run；
- 引擎适配器与 ModelPlugin 是两件事：前者描述**服务框架**，后者描述**模型**。
