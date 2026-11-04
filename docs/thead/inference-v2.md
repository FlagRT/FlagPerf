# PPU Inference（flagprof v2）适配指南

平头哥 PPU 的 Inference v2（flagprof）厂商适配。Inference v2 在目标进程内接入官方
PyTorch profiler 采集设备事件，经封存导入进入公共 Store，再由分析器生成性能/显存/通信
报告；架构见 [Inference 框架文档](../../inference/docs/README.md)，通用用法（含内置
运行器 `flagprof.runner`、delivery publish→verify→restore、双 rank launch、检查表）见
[使用说明](../../inference/docs/usage.md)。本页只记录 PPU 特有的身份、镜像与边界，
不重复通用流程；所有命令均经真机实测（2026-11-03）。

## 当前状态与边界

- 适配状态：candidate（候选适配）。2026-11-03 在一台 16 卡 PPU 主机（驱动
  `1.5.1-1d747a`）完成：单卡端到端（record → import → inspect → report）、
  Qwen3-Embedding-0.6B 上 perf / perf-lite / accuracy A/A oracle / modules /
  perf+memory / coverage（gems 镜像）全部预设、TP2 双 rank（nccl 后端）与 delivery
  publish→verify→restore；同型第二台主机（驱动 `1.5.3-5239a4`，契约允许的另一版本）
  复制了核心链路（CPU 套件 / 单卡冒烟 / TP2），结果一致（见验证一节末条）。多机
  （>2）、TP>2 与长期运行不在此范围。
- `performance_disposition` 对 thead 采集固定为 `not_qualified`：PPU 时钟一致性未完成
  验收，报告中的性能数字只作观测，不作资格判定。
- 集合通信识别只采信 exporter 显式元数据（kernel args 的 `Collective name` 字段）。
  TP2 隔离实测证实该决定：PPU 集合通信 kernel（pccl 名族）全部携带显式元数据，且
  `pcclKernel_AllReduce_RING_LL_Sum_float` 名字含 AllReduce 而元数据是 `barrier`——
  kernel 名会误导，元数据才是权威；因此不钉 kernel 名字族。
- 采集完整性不做声明：`final_evidence` 只记录可证的显式导出 flag（如 `record_shapes=1`），
  其余字段保持 unknown；这由 torch 2.10.0 exporter 的 Kineto chrome trace 结构决定
  （`schemaVersion=1`、顶层 `record_shapes`/`profile_memory`/`baseTimeNanoseconds`）。
- PPU 栈没有环境变量级的 profiler 开关，身份不走运行时探测，而走显式 sourced JSON
  （见下），导入与运行时做 UUID 交叉核验。

## 环境准备

容器镜像来自平头哥运行时栈发布（`flagrt/thead-ppu-runtime` 系列与 FlagGems 变体），
下表为验证时锁定的本地镜像 ID；环境重建后以本地 `docker images` 输出为准：

| 镜像 | ID（sha256 前 12） | 用途 |
|---|---|---|
| `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64` | `1ba7ead15378` | 默认采集与运行（torch 2.10.0，含 transformers/accelerate/triton，无 flag_gems） |
| `flagperf/thead-ppu-runtime-gems:1.0.0` | `087a234b7abe` | `coverage` 预设（FlagGems 路由覆盖）专用；torch 同为 2.10.0，`layer_norm` 在该构建未注册 |

PPU 在 torch 内使用 CUDA 兼容命名，`--device` 传 `cuda:N`。物理卡与逻辑设备的对应以
ppu-smi 为准：`ppu-smi -i N -q` 输出 Product Name / PPU UUID / Bus Id / Minor Number /
Driver Version；torch `get_device_properties(N).uuid` 去掉 `GPU-`/`gpu-` 前缀后须与
ppu-smi 的 UUID 一致。

### sourced 身份

每张物理卡一条 device 记录，数组顺序与 torch 逻辑序一致。下列脚本在宿主执行
（保存为 `gen_identity.py`），从 `ppu-smi -i N -q` 读出全部字段并生成单卡身份：

```python
import json, re, subprocess, sys

physical = sys.argv[1]   # 物理卡号，如 0
out = sys.argv[2]
q = subprocess.run(['ppu-smi', '-i', physical, '-q'],
                   stdout=subprocess.PIPE, universal_newlines=True).stdout

def field(name):
    m = re.search(r'^\s*' + re.escape(name) + r'\s*:\s*(.+)$', q, re.M)
    return m.group(1).strip() if m else None

model = field('Product Name')
uuid = field('PPU UUID')
bdf = field('Bus Id')
minor = field('Minor Number')
driver = field('Driver Version')
doc = dict(source='ppu-smi', driver=driver, devices=[dict(
    model=model, uuid=uuid, pci_bdf=bdf, driver=driver, minor=minor,
    physical=physical, host_device_node='/dev/alixpu_ppu' + physical)])
with open(out, 'w') as f:
    f.write(json.dumps(doc))
print(json.dumps(doc))
```

多卡身份把各单卡 JSON 的 `devices` 按逻辑序合并（顶层 `source`/`driver` 任取一张卡）：

```python
import json
a = json.load(open('identity-card0.json'))
b = json.load(open('identity-card1.json'))
a['devices'] = [a['devices'][0], b['devices'][0]]
with open('identity-2cards.json', 'w') as f:
    json.dump(a, f)
```

运行与导入时经 `FLAGPROF_THEAD_IDENTITY` 注入路径。校验规则：`source` 必须是
`ppu-smi`，`model` 必须是 `PPU-ZW810E`，所选逻辑卡的 UUID 与 torch 观测完全一致，
且 `pci_bdf`、`driver`、`host_device_node` 不得为空；任一不满足即拒绝采集。

格式契约钉在 `inference/flagprof/vendors/thead/contracts/v1.json`：
`hardware.model=PPU-ZW810E`，driver ∈ {`1.5.1-1d747a`, `1.5.3-5239a4`}，
collector = torch.profiler 2.10.0，`exporter.schema_version=1`。契约外照常导入，
但适用性标注 `outside`，不宣称已验证。

## 最小命令

宿主侧备好身份文件后，在容器内运行。flagprof 以源码方式随仓库工作树挂载，镜像内未
安装，因此必须设 `PYTHONPATH=/work/inference`（`$PWD` 为仓库工作树）：

```bash
IMG=flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64
DEV="--device=/dev/alixpu_ctl --device=/dev/alixpu --device=/dev/alixpu_ppu0"

# 采集（默认 final 预设：完整会话，目标正常退出后独立导入）
docker run --rm --shm-size=4g $DEV \
  -v "$PWD":/work -w /work \
  -e PYTHONPATH=/work/inference \
  -e FLAGPROF_THEAD_IDENTITY=/work/identity-card0.json \
  $IMG python3 -B -m flagprof record --vendor thead --preset perf \
  --store /work/stores/smoke \
  -- python3 /work/target.py

# 读取采集与导入状态（把 record 输出中的 run 路径写入 FP_RUN；
# 该路径是宿主机路径，容器内使用要换成容器挂载路径，如 /work/stores/...）
docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work/inference $IMG \
  python3 -B -m flagprof inspect "$FP_RUN" --verify --json

# 生成自包含报告
docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work/inference $IMG \
  python3 -B -m flagprof report "$FP_RUN" --out /work/reports/smoke
```

成功判定（JSON 字段，详见 usage.md 检查表）：record 输出 `"status": "complete"`、
`"target_exit".returncode` 为 0、final 预设 `"format_applicability": "within"`；
`inspect --verify --json` 每条 trace `import_status=complete`、`integrity=verified`；
report 的 `results` 中 `perf.op` / `perf.layer` / `perf.device` / `perf.gap` 为 `ok`。

采集真实模型可用内置运行器（输入 JSONL 每行一个字符串或含 `text`/`query`/`input`
的对象，详见 usage.md）。注意两个必守约定：模型参数须带 `hf:` 前缀，`--samples`
必传且为正。把上面 `--` 后的目标换成：

```bash
  -- python3 -B -m flagprof.runner hf:/work/models/Qwen3-Embedding-0.6B \
    --inputs /work/inputs.jsonl --outputs /work/out.json \
    --device cuda:0 --dtype bfloat16 --samples 8 --batch-size 1
```

embedding 任务已实测支持 bfloat16/float16/float32、任意 batch、eager/sdpa 注意力
（2026-11-04 PPU 实测，各产出有限且逐 forward 确定的 fp32 池化结果）。`--task
generate` 则是 OneRec 专用路径：只接受 sealed `--inputs-from` 包（schema 见
`flagprof/runner/fixed.load`，公共基线无通用打包工具）并在加载后硬校验架构为
`qwen3`/hidden 4096/36 层/vocab 176384——其他因果 LM 无法走该入口，生成评测请以
自定义目标 + `flagprof.attach()` 接入采集（usage.md）。

窗口模式加 `--capture-mode windowed`（按目标的 `flagprof` step 句柄推进窗口，
`--active`/`--windows` 控制窗数，`--timeout` 前传给受控目标进程）。多窗口时钟按窗口
独立投影（`source_clock=per_window`），不跨窗双计。外部自行导出的 trace 用
`flagprof seal <trace> --out <dir> --vendor thead --normal-close` 封存后经
`flagprof import` 进入 Store（capture 与 store 必须是两棵独立目录树）。

## TP2 双卡

`flagprof launch` 生成显式双 rank 工作进程并各自建 Store；`--store` 指向**尚不存在**
的组目录（由 launch 创建，预先建好会报 `File exists`）。目标经环境变量
`FLAGPROF_COMM_BACKEND` 与 `RANK` 获知后端与秩。PPU torch 2.10.0 注册的集合通信后端
只有 `nccl`（`flagcx`/`hccl`/`cmcc` 未注册，`gloo` 仅 CPU）。最小双 rank 目标
（保存为 `tp2_target.py`）：

```python
import os

import torch
import torch.distributed as dist

backend = os.environ['FLAGPROF_COMM_BACKEND']
rank = int(os.environ['RANK'])
dist.init_process_group(backend)
device = 'cuda:' + str(rank % torch.cuda.device_count())
x = torch.ones(1024, 1024, device=device) * (rank + 1)
for _ in range(4):
    dist.all_reduce(x, async_op=False)
torch.cuda.synchronize()
print('rank', rank, 'sum', float(x.sum()), flush=True)
dist.barrier()
dist.destroy_process_group()
```

双卡容器需同时暴露两卡设备节点，身份 JSON 含两条 device 记录（顺序即 torch 逻辑序）：

```bash
docker run --rm --shm-size=4g \
  --device=/dev/alixpu_ctl --device=/dev/alixpu \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 \
  -v "$PWD":/work -w /work \
  -e PYTHONPATH=/work/inference \
  -e FLAGPROF_THEAD_IDENTITY=/work/identity-2cards.json \
  $IMG python3 -B -m flagprof launch --vendor thead \
  --comm-backend nccl --nproc-per-node 2 \
  --store /work/stores/tp2-group \
  -- python3 /work/tp2_target.py
```

TP2 报告分两种情况。`launch` 组目录里的 `group.json` 组报告（每 rank 一份，带交叉秩
校验）面向**内置运行器驱动**的 TP2 run——校验读取运行器记录的 TP 身份。外部自备的
裸 `torch.distributed` 目标不含该身份，组报告会拒绝
（`actual TP backend or world size differs from group manifest`），此时对每个 rank
的 run 目录单独出报告：

```bash
docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work/inference $IMG \
  python3 -B -m flagprof report /work/stores/tp2-group/rank-0/runs/<run-id> \
  --out /work/reports/tp2-rank0
```

裸目标单 rank 报告的 `perf.layer` 为 `unavailable` 属实（无运行器 step 层标记）；
`perf.op`/`perf.device`/`perf.gap` 正常。

## 验证与测试

- CPU 回归：容器内 `PYTHONPATH=inference python3 -m pytest inference/tests -q`
  （2026-11-03 两台主机实测均 17 passed）。覆盖 PPU 身份 UUID 交叉核验与字段缺失
  拒绝、契约与 NVIDIA 互异且拒绝错误硬件栈、解释器显示名（python/python3）均在
  契约内、时钟变换仅在钉住 exporter 版本且 base 唯一时生效、集合通信显式元数据分类
  与未知名字归 compute、真实 exporter 片段导入回归与窗口投影防篡改、以及
  pidfd/memfd/owned 超时等进程原语。
- 真机片段 fixture：`inference/tests/fixtures/thead/gemm-fragment.json` 逐字抽取自
  2026-11-03 PPU 冒烟会话（gemm kernel 及其 launch 关联、用户注释与进程元事件），
  `provenance.json` 记录源 trace 与片段 sha256、镜像、采集器与硬件版本；
  `.gitattributes` 对该 fixture 关闭行尾转换。
- 真机预设矩阵（Qwen3-Embedding-0.6B，8 条中文输入，batch 1，bfloat16；权重来自
  ModelScope 2026-11-03 master 快照，逐文件 sha256 记录于 download-provenance.json，
  不随代码分发）：

| 预设 | 模式 | 实测结果（2026-11-03） |
|---|---|---|
| perf | final | `within`；perf.op/layer/device/gap = ok；61910 源记录 |
| perf-lite | windowed | 主机探针 112 事件、8 forwards；perf.layer = ok（设备采集不属该预设，`capture_status=not_requested` 是公共架构口径） |
| accuracy ×2 + oracle | windowed | `--tensor-blob all`；oracle identity_diff/layer/first_divergence = ok，`first_inequal=null`（同栈逐位一致）；op_sensitivity 需另配 `--tensor-ops-in` |
| modules | final | 5293952 源记录；op/layer/device/gap = ok；导入约 15 分钟 |
| perf + `--profile-memory` | final | memory = ok（13052 记录/11137 归一化/1915 排除） |
| coverage（gems 镜像） | windowed | route.coverage = ok；FlagGems 注册 6 函数（见运维注意） |

  `compile.health` 在所有预设为 partial——编译监听器只覆盖成功编译与持久缓存命中，
  失败、失效与进程内复用按面板口径不观测。
- TP2：`launch --vendor thead --comm-backend nccl --nproc-per-node 2` 在双卡实测，
  group complete、双 rank returncode 0、双 run 导入 complete；导入分类 comm=5
  （allreduce/barrier，basis=explicit_collective_metadata）。
- delivery：publish → verify → restore 跨容器实测通过（19 files/168472 bytes，
  restore 映射保持原路径结构）。
- 复制验证（同型第二台主机，驱动 `1.5.3-5239a4`，同一镜像 ID）：CPU 套件 17 passed；
  同模型 smoke perf `complete + within`（61914 源记录，4 面板 ok）；TP2 nccl
  group complete、双 rank returncode 0、分类 comm=5/memcpy=106/compute=4 与首台一致。
  gems 镜像未复制到该主机，coverage 面板按因 unavailable。单卡与 TP2 的源记录数在
  两台主机同量级（如 TP2 rank-0 均为 5957），个别记录数的个位差异属 profiler 采样
  波动，确定性只在 accuracy 的张量比较上声明。

### 运维注意（实测踩坑）

- 容器内必须设 `PYTHONPATH=/work/inference`（指向仓库 `inference/` 目录），否则
  `No module named flagprof`。
- `modules` 预设导入 907 s > record 默认 `--timeout 900`：显式给大 timeout；被杀的
  import 会留下 Plan-B checkpoint，复用同一 work 目录重跑会 exit 2——每轮重试用全新
  work/scratch。
- `--flaggems-include` 接受 JSON 文件路径（非内联数组）；include 要求全部注册成功，
  本构建 `layer_norm` 未注册，实测可用集为
  `["mm","add","mul","softmax","rms_norm","silu"]`。
- record 输出的 run 路径在宿主机上，传给容器内 inspect/report 时要换成容器内挂载路径。
- `launch --store` 的目录必须不存在（launch 自建）。
- 内置运行器的模型参数须 `hf:<directory>` 前缀且 `--samples` 为正（usage.md 有载）。

## 与 Base / Toolkit / Operation 的关系

本适配只覆盖 Inference v2 观测链路。Base Benchmark 与 Toolkit 见
[PPU 适配指南](README.md)，算子验收见
[PPU Operation](../../operation/vendors/thead/README.md)；各模块验收状态独立维护，
互不替代。
