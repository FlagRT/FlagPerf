# OneRec 执行策略由谁定义

性能/精度分析必须保持被测执行协议，观察到差异时如实记录和定位，不能通过改变数值策略取得一致。P800-INF-007 仍开放；先前的 FP32 词表头策略已撤回，离线重投影仅作诊断。

## 当前实现的职责划分

| 内容 | 当前来源与控制者 |
|---|---|
| 权重、层数、维度、EOS/pad 等模型配置 | 本地模型资产中的权重与配置；框架校验文件哈希和已登记架构 |
| backbone、attention、Linear 等实际计算 | 已安装 Transformers 模型实现调用 PyTorch/厂商后端；kernel 和部分累加/归约行为由后端决定 |
| dtype、attention、numerics | 操作者的 runner 参数；本次协议为 BF16/eager/default，runner 显式传给加载器并记录身份 |
| prefill/decode、KV、位置、取最后有效 token、argmax | 项目 `runner/onerec.py` 的 `GreedyExecution` 手工编排，不是直接调用模型的 `generate()` |
| 生成步数和停止策略 | 项目 runner 的 accuracy/performance 协议：accuracy 遇 EOS 或上限停止，performance 固定步数 |
| TP2 分片与聚合 | 模型配置的 base TP plan 加项目显式词表计划；兼容层映射已安装 Transformers 的计划名称，后端执行 collective |
| profiler、探针、导入和报告 | 分析框架负责观测；不应为使结果一致而改 dtype、选词或算子行为 |

`runner/onerec.py` 的文件说明称其语义移植自 v1，但这不是已经逐项证明与模型官方推荐推理实现等价的证据。特别是当前路径直接执行 `model.model(...) → model.lm_head(...) → argmax`，不能自动假定它继承了 `model.forward()` 或 `generate()` 中的 logits dtype、logits processor 或采样处理。

## 原来的策略是什么

`--dtype bfloat16 --attention eager` 加载权重；`--numerics default` 读取并记录后端当前开关。`GreedyExecution.advance()` 直接调用词表头，未额外提升输出精度，随后按 `torch.argmax` 选择 token。`--numerics reference` 是另一个显式协议，会设确定性及精度开关，不能与 default 的结果混作同一协议。

因此策略由三方共同决定：模型/库提供实现，项目定义执行编排，操作者选择协议参数；实际 kernel 由后端执行。不能说“全部是模型自己控制”，也不能因为后端出现数值差异就认定项目 runner 已无责任。

## 后续核验边界

继续保持原协议。先在独立诊断中核对原始模型推荐调用及当前安装 Transformers 的 `forward/generate` 语义，再定位同输入单卡/TP2 的首次算子分歧。若发现 runner 偏离正式规范，必须提交具体规范、前后行为与各平台影响，由维护者审查统一修正；不得只给 P800 提高精度以取得相同 token。

历史 BF16 logits 并列与 FP32 离线重投影说明分数如何翻转，不构成原 BF16 模型精度通过，也不证明上游 hidden 差异是正常误差。原始失败、模型/输入身份和数值证据继续保留，见 [验证记录](inference-v2-validation.md)。
