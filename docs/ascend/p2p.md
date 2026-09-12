# 单机 P2P 协议

正式协议名为 `p2p-single-node-v1`。目录仍属于原来的
`base/benchmarks/interconnect-P2P_intraserver/ascend/`，原 Case 的 `main.py` 保持不变。
公共 distributed backend 为 Torch-FL `flagos`，FlagCX 是内部通信数据面。

## 工作负载与范围

| 项目 | 固定约定 |
|---|---|
| 节点与 rank | 单节点、恰好两 rank |
| workload | FP32、单向 blocking `send/recv`、100 次 warmup |
| 原结果公式 | `2 × bytes / time`；这是原 Case 指标，不是物理单向 payload 带宽 |
| 拓扑与方向 | SIO：Device 14→15；HCCS_SW：Device 12→14 |
| 消息大小 | 4、16、64、256 MiB |
| 规模曲线 | 每个拓扑×大小三次 monitor off，共 24 run |
| 监控 A/B | SIO 64 MiB，off/on/on/off/off/on，共 6 run |
| 单 run 测量窗 | 最短 45 秒、目标 60 秒；timeout 180 秒 |
| 判定 | 单元 CV ≤5%，监控中位数差绝对值 ≤5%，完整双 rank、零 fallback、拓扑与 postflight 通过 |

固定配置是既有测量选定的工作量，不根据新的运行速度自动扩展迭代。
首次运行可能受初始化、编译和宿主状态影响；全矩阵约 30 分钟纯测量，端到端更长。
反向通信、跨机和长稳属于这份资格范围之外，不用单机结果推断它们已通过。
新主机的编号和拓扑必须重新确认；更换设备范围、工作量或阈值需要新计划和验证记录。

## 入口与命名

| 用途 | 文件 |
|---|---|
| 正式运行配置 | [ascend910_cann9_p2p.yaml](../../base/configs/ascend910_cann9_p2p.yaml) |
| 单次功能检查 | [case_config.smoke.yaml](../../base/benchmarks/interconnect-P2P_intraserver/ascend/case_config.smoke.yaml) |
| 消息大小校准 | [run_p2p_calibration.py](../../base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_calibration.py) |
| 正式资格矩阵 | [run_p2p_qualification.py](../../base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_qualification.py) |
| 正式计划 | [p2p-qualification-plan.json](../../base/benchmarks/interconnect-P2P_intraserver/ascend/p2p-qualification-plan.json) |
| 构建后通信与生命周期检查 | [qualify_runtime.py](../../base/vendors/ascend/torch_fl_2.10_flagcx/qualify_runtime.py) |

工作量按用途和拓扑命名：`case_config.p2p-{sio,hccs-sw}-{4m,16m,64m,256m}.yaml`，
校准使用 `case_config.p2p-calibration-<size>.yaml`。不保留已废弃的超长协议或开发阶段入口别名。
`qualify_runtime.py --gate` 使用 `sentinel`、`nondefault-stream`、`timeout-cleanup`、
`acl-event-ab`、`peer-failure`、`hung-p2p`。故障注入需独立预约设备，不能混入日常性能测试。

```bash
# 只输出单次功能检查计划。
python3 base/run.py benchmark run \
  --config base/configs/ascend910_cann9_p2p.yaml \
  --case interconnect-P2P_intraserver --device-ids 14,15 --nproc-per-node 2 \
  --case-config base/benchmarks/interconnect-P2P_intraserver/ascend/case_config.smoke.yaml \
  --monitor off --dry-run

# 校准与正式矩阵默认只规划，不触碰硬件。
python3 base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_calibration.py
python3 base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_qualification.py

# 已完成设备预约并接受整套矩阵占用后执行。
python3 base/vendors/ascend/torch_fl_2.10_flagcx/run_p2p_qualification.py --execute
```

单次 Benchmark 真正执行时移除 `--dry-run` 并传 `--allow-privileged-root`。
正式通信镜像已在声明范围内资格化，不需要 `--allow-candidate-runtime`；新建候选镜像仍走候选门禁。
`--execute` 的矩阵编排器会给子进程传入权限参数，执行前需审阅其计划。

新会话使用 `flagcx-p2p-qualification-<UTC>`，可用 `--execute --resume-session <session>` 恢复
同一计划产生的失败会话。恢复时重新核对计划哈希、任务与既有证据，并保留失败尝试。
命名变化前的历史会话不改写、不跨协议恢复；使用当时版本读取其原始证据。

## 证据和公开分发

[资格记录](../../base/vendors/ascend/torch_fl_2.10_flagcx/p2p-qualification.json)保留了
原计划、原 summary 和来源资格文件的 SHA-256，以及原镜像 ID、30 个成功 run 和一次失败尝试。
既有记录中最大 CV 为约 2.063%，monitor 中位数差约 0.0033%；这些是历史记录中的观测值。

仓库另提供明确标注为维护者记录的
[校准来源摘要](../../base/vendors/ascend/torch_fl_2.10_flagcx/p2p-calibration-record.json)。
原始校准/正式日志当前不可从本仓库取得，本次没有独立重验其全部原始文件或重跑 NPU。
摘要不是数字签名，也不能替代原始日志。公开计划通过摘要、配置和 runtime 的哈希关联，
可在独立克隆中静态检查；执行仍需实时 preflight、拓扑、测量和资源回收检查。

正式计划的设备、方向、workload、阈值、runtime 配置都与资格记录绑定。重命名导致的文件哈希
更新不用于覆盖原实验哈希，也不自动赋予新主机或新镜像相同的性能资格。
