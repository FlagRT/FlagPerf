# 旧 Inference 迁移说明

当前 Inference 使用显式命令进行模型精度和性能分析。原 host.yaml、SSH 集群启动、旧模型案例、任务评分器及
TensorRT/TorchTRT/Inductor 等编译引擎流程已移出当前工作树，不属于当前支持清单。
`run.py` 的旧无参数启动方式已被 `list|preview|accuracy|performance|export` 命令取代。

需要原框架时，在单独工作树恢复迁移前提交，不覆盖当前代码：

```bash
# 从包含完整历史的 FlagPerf 仓库根目录执行。
git worktree add --detach ../FlagPerf-legacy 3e7c558b6f56e6ea5f9c8b318852d97d1517a14c
```

然后阅读该历史工作树中的根 README 与 `docs/inference/inference-case-doc.md`。
旧流程仍要求其对应的软件、权重、数据集及设备环境，不能因源码可恢复而认为兼容当前镜像。
浅克隆若没有该提交，需先获取仓库历史。

新工具从 [当前入门指南](../../inference/README.md) 开始。旧配置不能直接传入新入口；模型和环境应按新配置重新填写。
preview 策略绑定执行源码及环境，迁移后重新生成。历史 schema 1/2 报告可以读取，但其策略不能用于 schema 3 执行或续探。
