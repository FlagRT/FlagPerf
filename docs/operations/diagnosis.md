# Operation 诊断与复放

[返回 Operation 入门](../../operation/README.md)

诊断帮助整理“在哪个阶段失败、已有结果说明什么、还缺少哪些证据”。它不自动确认数学根因或修改底层算子。
有两种触发方式：正常运行失败时自动补采，以及对已有 Case 结果执行独立 `diagnose`。

| 入口 | 何时使用 | 额外执行 |
|---|---|---|
| `run --diagnostics failures`（默认） | 原始运行中出现数值失败或路由缺口 | 按问题选择有限的参考复核、数值统计或 CPU 调用链 |
| `diagnose --source-task ...` | 已有封存任务，需要整理证据 | 不启动 Docker/NPU，不加载张量 |
| `diagnose ... --replay` | 已确认空闲设备，希望用原输入复查 | CPU 参考检查、一次原路径 probe、必要补采 |

正常通过不会触发失败诊断。`--diagnostics off` 只关闭附加检查，原正确性和路由检查继续执行。
数值失败先核对保存输入/模块状态与 CPU 参考；参考本身异常时停止后续数值和 trace 补采。
仅路由 partial 时至多增加一次 CPU 调用链，不因此升级为设备 kernel 证据。
初始化、OOM、超时等执行异常优先整理已有日志，不反复启动设备。

## 从具体任务开始

```bash
# 从 FlagPerf 根目录执行，替换成 report.md 中链接的具体任务子目录。
python3 operation/run.py diagnose --source-task /absolute/case-result --dry-run
python3 operation/run.py diagnose --source-task /absolute/case-result

# 可选：使用同一镜像和保存输入，在一张空闲设备上检查一次。
python3 operation/run.py diagnose --source-task /absolute/case-result \
  --replay --device-ids 0 --allow-privileged-root
```

源任务必须具有 schema 2 封存清单；缺少张量的 blocked/failed 任务仍可以离线诊断。
复放则必须有原输入、必要参考及镜像身份。镜像解析后的 ID 必须与来源一致；更换镜像应重新做普通评测。
默认采用来源设备编号，指定 `--device-ids` 时只接受一个设备。其他厂商的复放还受各自适配器约束。

输入复制到新的诊断目录，源记录不改写。输入摘要、来源索引和本次源码单独保存。
清理或设备健康检查失败会中止后续执行；补充数值/路由信息不会覆盖原算子的正确性门禁。

## 阅读诊断结果

报告分别列出原任务状态、本次执行状态、正确性、路由和证据缺口。
例如“probe 执行完成，但数值未通过”表示获得了计算结果，不等于执行过程崩溃。
一次 replay 通过也不推翻原失败，不能代替重复性或性能验证。

| 退出码 | 诊断入口的含义 |
|---|---|
| 0 | 公共检查完成，未列出覆盖缺口；原算子仍可能数值失败 |
| 2 | 缺少证据或部分检查不可用 |
| 1 | 校验、工具执行、清理或设备健康检查失败 |

新增输出默认位于 `operation/result/diagnosis-<id>/`，`--result-root` 可指定父目录。
报告中的复放命令根据该任务的真实来源生成；缺少所需文件时只列缺口，不生成不可执行的建议。

需要重新呈现已有诊断报告时使用：

```bash
python3 operation/run.py report --run-dir /absolute/diagnosis-result
```

报告重建不执行算子，也不修改封存 JSON、哈希或原判定。
