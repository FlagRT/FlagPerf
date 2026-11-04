# P800 Inference v2：验证结果与开放缺陷

运行方法见 [适配指南](inference-v2.md) 和 [从空工作区到报告](inference-v2-quickstart.md)。本文区分运行完成、采集可读、数值一致和资格通过；记录更新时间为 2026-11-04，设备数据来自 2026-11-03，未因文档更新重新执行硬件矩阵。

## 当前结论

候选适配的主要已测链路能够完成。**P800-INF-007：OneRec 单卡/TP2 数值不完全一致，根因未确认，TP2 精度尚未通过。** 性能 disposition 仍为 `not_qualified`，完整性、时钟和通信关联证据尚有限制。不能声明“所有结果无错误”或“完整能力认证”。

设备实测基于 dev-1.0 起点 `41ab2dce` 上的候选分支：主要扩围使用 final6 冻结快照、`af12f57e` 与 `af1b8e9c`；`d753fb8d` 下完成最终 CPU 回归和 Qwen 双窗口复测。快照的 CRLF 与 Linux checkout 的源码字节闭包不同，不能将全部历史任务标为最终提交实测。后续文档提交不改变原始运行源码或证据。

| 范围 | 已测事实 | 不能据此推导 |
|---|---|---|
| Qwen3-Embedding-0.6B | BF16/eager、max-length 32；五预设、多批 2+1、内存、有限 SiLU 路由及离线消费；复核 24 份输出 SHA/有限值，和对应基线一致 | 任意输入/模型、全部 FlagGems 算子或跨硬件精度通过 |
| OneRec-8B 单卡 | batch 1、32 步 greedy；A/A oracle 的 1280 个 full 边界 digest 相同、max_abs=0，生成 token 无差异，KV 已释放 | 独立高精度参考验证，或与 TP2 等价 |
| OneRec TP2 | NCCL 兼容层、FlagCX 各完成 accuracy/perf-lite/perf；两 rank 内部一致、两后端已测序列相同；每 rank 32 步、2304 AllReduce、32 AllGather | 单卡/TP2 数值等价、完整设备通信调用关联或性能资格 |
| 大模块 trace | Qwen 4,126,414、OneRec 8,215,086 条源记录；封存恢复导入后可分析 | 默认预算足够、任意规模低延迟；原预算/超时失败仍保留 |
| final/windowed | Qwen 双窗口及 OneRec 32 步的 projection/gap 可读；按 source clock 隔离窗口 | scheduled window 覆盖整个请求或设备时钟误差已知 |
| 故障与包 | timeout/异常/SIGTERM/SIGKILL、损坏 trace、零 blob 预算、wheel 构建/独立安装与离线 contract 读取 | 故障注入的非零退出全部属于未修复运行错误 |

“五预设已测”包含首次失败后独立补测或封存恢复，不表示原单轮全部成功。已有失败和补测各保留身份。

## 回归与实测边界

`d753fb8d` 的最终回归按套分开统计，不重复累计子集：

| 套件 | 总数 | 通过 | 跳过 |
|---|---:|---:|---:|
| Inference 标准库（Python 3.10，各主机） | 19 | 18 | 1，可选 torch CPU |
| Inference 标准库（Python 3.12，各主机） | 19 | 18 | 1，可选 torch CPU |
| 真实 Gloo（单独执行） | 1 | 1 | 0 |
| Base CPU | 381 | 380 | 1，既有跳过 |
| Operation CPU | 167 | 167 | 0 |
| 公开分发 | 14 | 14 | 0 |

这些是 CPU/软件回归，不能替代 NVIDIA/Ascend 或 Base/Operation 硬件复测。无设备机器可从 `inference/` 运行 `python -B -S -m unittest discover -s tests -v`；可选真实 Gloo 使用具备匹配 torch 的解释器、去掉 `-S` 单独执行 `test_communication.py`。

2026-11-04 加入 FP32 logits 候选后，独立源码副本的标准库回归为总计 21、18 通过、3 可选跳过；新增两项 torch CPU 数值测试单独运行均通过，另有真实两进程 Gloo 分片/聚合检查通过。未重复执行 Base/Operation 硬件测试，也没有将这些新测试数回填历史提交。

本次开销观察为加载后四次同步前向、三轮固定顺序样本：perf-lite 1.1321×、裸 profiler 1.2634×、封装 profiler 1.2600×、coverage 18.7500×，均相对对应无采集基线。排除模型加载、导出和离线导入；仅作本栈诊断扰动观察，不作性能达标或跨设备排名。

## P800-INF-007：单卡与 TP2 分歧

同一 P800 主机中，单卡 performance 与 NCCL 兼容层 TP2 performance 的模型文件 SHA、输入 SHA、样本、BF16/eager、numerics、32 步协议一致；差异为 world size/分片计划、通信后端及 TP 的 accelerate overlay。结果：

| 检查 | 单卡 | TP2 |
|---|---|---|
| 第 14 个生成 token（0-based step 13） | 6741 | 5582 |
| 其余 31 个 token | 两者相同 | 两者相同 |
| 最终 hidden SHA-256 | `8539b8458b0c7f894b9d4a9502c51c3df56e590b1eae076fbc188cec788c7325` | `37ff6ebae5f7d817f5cc1b8349b778ca839cbdc447a2f46ee538bdb74a70f546` |
| KV 释放 | true | true |

FlagCX TP2 与 NCCL 兼容层的已测 token 序列、最终 hidden 摘要一致。这只证明 TP2 内部重复一致，不消除与单卡的分歧。后续只读张量诊断见下节；BF16 分片/归约顺序可能造成上游差异，但不能据此排除适配问题或直接归责厂商。

可从已保存的两份 runner.json 只读复查，不需要重新占用设备：

```bash
"$PY" -B -S - /path/to/single/runner.json /path/to/tp2/rank-0/runner.json <<'PY'
import json, sys
a, b = [json.load(open(p, encoding='utf-8')) for p in sys.argv[1:]]
for key in ('model_files', 'inputs_sha256', 'samples', 'dtype', 'attention', 'numerics', 'generation'):
    assert a['oracle_identity'][key] == b['oracle_identity'][key], key
x = [r for r in a['requests'] if not r['warmup']][0]
y = [r for r in b['requests'] if not r['warmup']][0]
print('lengths:', len(x['tokens']), len(y['tokens']))
print('differences:', [(i, u, v) for i, (u, v) in enumerate(zip(x['tokens'], y['tokens'])) if u != v])
print('hidden_equal:', x['final_hidden'] == y['final_hidden'])
PY
```

脚本打印事实，不自判精度通过。若长度不同也必须记录，不能只看 zip 后的配对项。修复验证须使用统一源码/依赖和固定输入，先复现，再采集 prefill、逐 decode 层及分歧步 logits；需要时使用相同已保存前缀定位，避免不同生成轨迹混淆归因。对照修复前后与两后端，保留原始不一致记录，不放宽容差使其通过。

### 2026-11-04：已定位的舍入触发与候选处理

从已保存的 accuracy 张量读取全量 blob 并逐个核对 SHA，发现 prefill 第一个 decoder layer 输出已经不同；第 14 步前，两侧生成 token 相同。该步单卡的 `6741/5582` logits 为 `24.625/24.5`，TP2 为 `24.5/24.5`。现行 `torch.argmax` 对并列最大值取较小索引，因而 TP2 选中 5582。

使用原始 BF16 hidden 和同一模型词表权重，在 CPU 对**完整词表**做 FP32 重投影：单卡候选分数约 `24.581793/24.493568`，TP2 约 `24.482998/24.439430`，两侧均选择 6741。保存的 32 步 hidden 重投影后，两侧的 argmax 也全部相同。候选行另外用 FP64 点积核对。这里验证的是既有 hidden 的输出投影；不是重新生成 32 步，也没有消除上游 hidden 差异或证明全部误差只来自输出头。

候选选项 `FLAGPROF_P800_ONEREC_LOGITS=fp32` 在 P800 专属适配模块中将未绑定的词表权重及其输入转换为 FP32，使输出在 argmax 前保留 FP32；不改变 backbone、greedy 规则、TP 分片或公共 v2 架构。默认 `model` 保留原行为。单卡与 TP2 都须设置同一选项；数值策略写入 runner `oracle_identity.numerics.p800_logits`，新策略结果不能冒充原 BF16 协议的通过记录。

```bash
export FLAGPROF_P800_ONEREC_LOGITS=fp32
# 在已准备好的单卡/TP2 容器内，重新执行相同固定输入的 runner/launch 命令。
# 对照运行另开新容器或去掉该变量，使用独立输出目录。
```

FP32 词表增加常驻权重内存（本模型单卡约 1.35 GiB，TP2 每 rank 约 0.67 GiB，另有转换临时内存），可能改变性能；须重测。CPU 低 margin 负例及真实 Gloo 双进程词表分片/聚合测试已通过；**尚无本策略的 P800 设备生成通过证据**。当前设备预检发现他人训练句柄或在用显存，未越过隔离门禁执行。缺陷仍开放、TP2 精度未通过，后续须同卡/同依赖单卡与 TP2 自由生成、固定前缀定位、两后端及重复输入复测。

## 固定输入包生成

以下为 OneRec 的**公开合成诊断输入包**示例，不是业务数据或独立 oracle 真值。先自行取得权重并核验来源、revision、使用条款与文件哈希；权重不随仓库分发。用固定镜像解释器执行，模型只读位于 `/models/OneRec-8B`，输出包目录须不存在：

```bash
"$PY" -B - <<'PY'
import hashlib, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
model = Path('/models/OneRec-8B')
root = Path('/fp/onerec-package')
root.mkdir(exist_ok=False)
prepared = root / 'prepared'
prepared.mkdir()
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True,
    trust_remote_code=False, padding_side='left')
values = dict(tokenizer(['Recommend an item for this user.'], padding=True,
    truncation=True, max_length=256, return_tensors='pt'))
torch.save([{'inputs': values}], prepared / 'inputs.pt')
(prepared / 'forward-inputs.json').write_text(json.dumps({
    'batches': [{'sample_ids': ['public-synthetic-0']}]}), encoding='utf-8')
(prepared / 'samples.json').write_text(json.dumps(['public-synthetic-0']), encoding='utf-8')
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
model_files = {p.name: sha(p) for p in model.iterdir()
    if p.suffix in ('.json', '.safetensors', '.txt')
    and p.name not in ('source-inventory.json', 'download-complete.json')}
manifest = {'oracle_id': 'public-onerec-p800-diagnostic-v1',
    'files': {str(p.relative_to(root)): {'sha256': sha(p)} for p in prepared.iterdir()},
    'contract': {'model_files': model_files,
        'inputs': {'batch_size': 1, 'max_length': 256, 'padding_side': 'left'},
        'model': {'dtype': 'bfloat16', 'attention': 'eager',
                  'options': {'max_new_tokens': 32}}}}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(root)
PY
```

所有对照重用生成后的同一份包，不能在两侧重新 torch.save 后期待文件 SHA 相同。修改输入或模型须生成新包并使用独立身份；不能冒充历史 NVIDIA/Ascend reference。Qwen 精度输入包也使用这四个文件及哈希，contract 按实际任务/输入声明；固定包支持 `--samples` 和重新组 batch，多批次身份须对齐。

accuracy 对照显式保存 `--tensor-blob all`，并检查预算/跳过计数。`report oracle` 先检查身份与 alignment，再检查 `digest_equal`、max_abs、非有限值和 first divergence。rsqrt 人工输入扰动负例的 max_abs 为 0.057190895、16 个元素不同，而各侧 replay 对自身预期 max_abs=0；这是检测与重放功能正常的证据，不能写成模型无误差。

## 缺陷与证据边界

| 编号 | 触发与处理 | 状态 |
|---|---|---|
| P800-INF-001 | tracing 变量未先于 torch 设置，或逻辑卡不匹配，无设备轨 | 已加门禁；空/截断轨负例保留 |
| P800-INF-002 | 固定镜像缺 pidfd/memfd Python 包装 | 受限 Linux syscall 兼容；超时、SIGTERM/SIGKILL 已测 |
| P800-INF-003 | windowed timeout 未传递，projection 未写入或来源时钟混合 | 已修复；Qwen 双窗口、OneRec 32 步及旧失败分别保留 |
| P800-INF-004 | wheel 漏打包 P800 contract | 已修复；独立构建安装后无 torch 读取通过 |
| P800-INF-005 | 默认 1 GiB、磁盘空间或导入时间预算不足 | 显式预算、新目录封存恢复；不是任意规模保证 |
| P800-INF-006 | exporter 缺 loss/时钟误差上界，BKCL 后台 launch 缺调用因果键 | 开放；完整性 unknown、clock_valid=false、设备调用关联 unavailable |
| P800-INF-007 | OneRec 单卡/TP2 第 14 token 与最终 hidden 不同 | 开放；根因未确认，TP2 精度未通过 |

compile listener 只观察成功编译/持久 cache hit/miss，失败、失效和进程内复用不可据此推断。硬件 counter、8 卡、跨机与服务热加载不在当前实现交付范围；未做 NVIDIA/Ascend 实机数值回归。原生通信类别识别不生成 logical call 关联，主机 enqueue 耗时不等于设备通信耗时。windowed 请求不完整时保留 partial/unavailable。

真实历史失败包括前置环境缺失、无设备轨、空间中止、modules 800 秒导入超时、TP 网络/依赖/接口错误和测试脚本错误；后续复测不覆盖原失败。模拟 OOM、人工异常、timeout、SIGTERM/SIGKILL 属于故障用例，需与 expected 状态逐项对照，不能只做日志关键词计数。弃用及 profiler 未预热警告原样保留，采集耗时不作为无扰动性能结论。

## 证据可得性与问题反馈

上述实机验收原始模型、完整大 trace 和全部运行目录不随此源码仓库发布；因此仅凭本页无法独立复算每一项历史统计。仓内 `inference/tests/fixtures/p800/` 只含小型公开片段、来源说明和哈希，用于解析/门禁回归，不代表完整模型性能证据。维护者在 PR/issue 中应补充经过检查、可分享的命令、被测 commit、输入/模型哈希、runner/report 摘要及所需原始文件，不能将未公开证据写成仓内可直接下载。

反馈时附固定栈/驱动版本、模型与输入身份、逻辑卡/物理 UUID 对应、完整命令、预期与实际状态、错误日志和产物哈希。业务输入、令牌及私钥不应进入公开 issue；设备标识和本机路径按分享策略处理。`--declare` 只登记可允许的身份变化，不能豁免数值差异或伪造输入一致。
