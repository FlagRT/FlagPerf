# 成本预检与离线恢复

已有采集可在 CPU 环境分析、比较或重新展示，输出使用新目录。源结果和参考包只读保留。
完整分析需要 CPU PyTorch、NumPy、safetensors、PyYAML；只有标准库与 PyYAML 的宿主可做 list、help 和可用的报告视图。

## 成本预检

```bash
python3 run.py performance --config config/local.yaml --level layer --dry-run --output result/planned
```

预检不启动设备，输出 estimated/verified=false，列出输入、阶段、worker/rank、记录及空间估计和待验证项。
没有历史耗时就不编造时长。tmpfs 也消耗内存，预估可用空间不是实际运行保证。

## 恢复分析

```bash
python3 run.py analyze --source /absolute/path/to/saved-run --output result/recovered
python3 run.py report --source result/recovered --level layer --output result/recovered-view
```

analyze 用当前分析源码处理已保存的采集，复用有效检查点，恢复比较和符合条件的生产封包。
它不恢复设备采集、不执行历史源码、不修改源目录，也不将原失败或需复查状态提升为通过。
旧 report 作为历史证据保留；需要新展示时另用 report。缺失 tensor/trace 无法恢复，失败现场留在 `.incomplete-*`。

## 转换历史参考

```bash
python3 run.py repack-reference --source /absolute/path/to/old-package --output result/repacked
```

输出 `core/` 和 `evidence/`；消费时 `--oracle` / `--baseline` 指向 `core/`。转换不覆盖原包，完整校验原始观测和包合同。

## 重绘与重新解析

```bash
python3 run.py report --source /absolute/path/to/saved-performance --level total --output result/total-view
python3 run.py report --source /absolute/path/to/saved-performance --level layer \
  --reanalyze --analysis-timeout-seconds 900 --output result/reanalyzed-view
```

report 用于性能视图；`--portable` 复制可搬移的实体证据。重新解析需要完整原始采集，精简任务已删除的数据无法重建。
CUDA 分析在独立子进程中执行，默认 900 秒限时，与设备 worker 超时分开；超时保留证据并显示 partial。
`analyze --analysis-timeout-seconds N` 也可覆盖恢复分析的限时，N 必须为正整数。analyze/repack-reference 不接受设备、模型、组件等采集选项，配置仍以来源封存内容为准。
已导出的 JSON trace 可在 CPU 分析；若只有 CANN 原始二进制采集，导出还需要来源镜像、兼容 exporter 及其管理接口权限，不会重新执行模型。
明细可能位于 `artifacts/` 分片，搬移时保留整个目录，不能只复制顶层 JSON。
所有结果报告可通过 `tools/replay_reports.py --source SOURCE --output NEW_DIRECTORY` 重新展示，来源目录名称须唯一。
