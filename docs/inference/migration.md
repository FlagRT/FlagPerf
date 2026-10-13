# Inference 版本迁移

当前提供四模型、A100 参考比较、运行监控及离线恢复。源码统一采用 benchmarks、inference_engine、vendors、components、registry 与 runtime 分层。
此前的 models/engines/analysis/reporting 内部位置已收敛到该布局；公开使用以 CLI、配置和结果协议为入口。

## 现有用户

保留 Ascend 的 list、preview、accuracy、performance、export、report，以及 `--image` 配置入口。
将本机设置存入 `config/local*.yaml`。模型与镜像默认为空；原生路径不再要求准备编译器包，组件路径仍校验用户提供的包。
性能默认开启独立监控；成功无异常的普通任务默认精简原始证据，诊断时加 `--keep-tensors` / `--keep-evidence`。

NVIDIA 通用 preview/accuracy/performance/export 入口收敛为 A100 参考生产：

```bash
python3 run_inference.py oracle --config config/a100.yaml --model-path /absolute/path/to/model --output result/new-oracle
python3 run_inference.py performance-baseline --config config/a100-performance.yaml \
  --model-path /absolute/path/to/model --output result/new-baseline
```

run.py 不启动 NVIDIA 容器。参考消费见[A100 指南](references.md)。
历史结果按读协议保留；schema 1～3 参考包只读兼容，新参考使用 schema 4。可转换到新目录，不自动覆盖旧包。
策略仍绑定执行源码与环境，代码或目录变化后须新 preview，不修改历史身份摘要继续使用。

## 更早的执行框架

旧 host.yaml、SSH 集群启动、任务评分器、其他模型和编译引擎已移出当前工作树，不属于当前支持清单。
在包含完整历史的仓库根目录恢复迁移前提交：

```bash
git worktree add --detach ../FlagPerf-legacy 3e7c558b6f56e6ea5f9c8b318852d97d1517a14c
```

浅克隆需先获取历史。恢复源码仍需原流程对应的环境、模型和数据，不能由源码存在推断与当前镜像兼容。
