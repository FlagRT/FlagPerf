# 昆仑芯 P800 Base 最终验收与交付复核（2026-10-08）

## 1. 结论

**Base 功能收尾与限定范围交付成立。** 15 个 case 的 10 项可执行与 5 项 early skip、统一入口、正确性、设备绑定、监控、资源清理及 per-run 报告均已复核。用户于本轮明确接受 D2H pageable/nonblocking 三组不足 15 秒的样本，不将该时长作为交付阻塞。

交付结论保留两个边界：八卡五次测量和稳定性通过，但卡 2 的历史不可纠正 ECC 使健康资格仍受限；pinned-blocking 等已有不稳定组原样保留。runtime manifest 保持 candidate / validated:false。本报告不作“全部组严格资格通过”或“全卡健康认证”声明。

## 2. 复核对象、方法与源码

- 远程：klx `/home/kzhang519/Zhiyu/runtime-team/FlagPerf`；开始时工作树干净，HEAD 与 origin 分支均为 `be0f2ea575849a5fdadd8e3e5ab1db317b304bff`。
- 冻结标签 `p800-base-frozen-20261008` 解引用为 `7371195ea11426ee33511fac1363f04d74eabe09`；它早于聚合层移除，不是现行分支的别名。
- 最终测量源：`base/result/p800-base-final-20261008/base-runs/` 61 run，`base/result/p800-idle-accept-20261008/base-runs/` 53 run；合计 114。另有 idle 根 `failed-attempts/` 七项及 control 中的首次容量权限拒绝，不混入成功样本。
- 本轮逐 rank 调用现行指标验证器，复算五次组，逐文件哈希，再在临时副本对全部 114 run 重建两次报告。未运行新的硬件负载、修改 runtime、重置设备或覆盖原始 run。
- 单 run 报告是 Base 的规范结论视图。已移除的 `acceptance-report` 历史输出及其“18/28”统计口径不再作为现行严格资格数量。

## 3. 验证结果

| 项目 | 本轮结果 |
|---|---|
| Base / Ascend Toolkit / PR0 CPU 回归 | 252 / 80 / 32 项全部通过 |
| P800 15-case dry-run | 10 applicable、5 skipped，命令全部成功 |
| 原始测量与正确性 | 114/114 passed；逐 rank metric/correctness 验证零错误 |
| 原始 run 总状态 | 110 passed、4 partial；后者为容量与 3 个短 smoke |
| 原始健康事实 | 104 run 无记录的不可纠正 ECC；10 run 含卡 2，degraded |
| 报告重建 | 114/114 两次字节一致，且与已存报告一致；副本 summary 未改 |
| 原始文件保护 | 7,196 个 run 文件前后 SHA256 无变化 |
| 严格重复组复算 | 22 组：15 passed、2 unstable、5 rejected；保留具体原因 |
| 项目时长例外 | 三组 D2H pageable/nonblocking 的窗口不足被接受；其中两组 CV≤5%，初组 CV 7.158% 仍不稳定 |
| 历史 Base 索引 | 6 份索引、25,392 条；仅 Day 5 的 verification.json 一处旧索引不匹配，见第 8 节 |

上述 22 组按同根的 `*-qN` 系列划分：21 组各五次，七卡组只有三次。另有五个 smoke 与一次容量，共 114 run。17 组可记录为“严格通过或时长例外接受”，不能将这 17 组全部标成 `qualification.py passed`。

## 4. 计算与设备内存

| 项 | 五次中位数 | 样本 CV | 状态 |
|---|---:|---:|---|
| FP32 | 118.4729 TFLOPS | 0.3242% | 严格通过 |
| FP16 | 259.1654 TFLOPS | 0.0742% | 严格通过 |
| BF16 | 118.9696 TFLOPS | 0.0010% | 严格通过，吞吐限制保留 |
| INT8 | 504.5260 TOPS | 0.3756% | 严格通过，xtorch_ops、BF16 输出 |
| 设备内存带宽 | 2235.9549 GB/s | 0.0040% | 严格通过，预分配 copy 读+写 |
| 容量（单次） | 102.898860032 GB / 95.83203125 GiB | 不适用 | 持有 98,132 MiB，32 次成功分配；释放、postflight、cleanup、lease 通过 |

计算使用 8192³。FP32/FP16/BF16/INT8 的 iterations 分别是 2000/5000/5000/8000，warmup 为 100/100/100/10。不能将它们全部写成 2000 次。INT8 Base 不是 Toolkit 的 XBLAS fc_fusion；D2D Base 是预分配 `Tensor.copy_`，不是计时内 clone 分配。

## 5. 传输与用户接受的时长例外

全部为 512 MiB、单卡 5 的历史测量；八变体各保留独立合同。

| 变体 | 代表组中位数 GB/s | CV | 项目交付结论 |
|---|---:|---:|---|
| H2D pageable blocking（idle） | 10.2853 | 2.458% | 严格通过 |
| H2D pageable nonblocking（final） | 19.6053 | 2.202% | 严格通过 |
| H2D pinned blocking（final） | 19.8705 | 2.595% | 严格通过 |
| H2D pinned nonblocking（idle） | 19.7922 | 4.163% | 严格通过，接近 CV 门槛 |
| D2H pageable blocking（final） | 11.0360 | 1.564% | 严格通过 |
| D2H pageable nonblocking（idle） | 18.6422 | 1.511% | 测量及稳定性通过；时长例外接受 |
| D2H pinned blocking（idle） | 24.0510 | 4.862% | 严格通过，其他两轮 unstable 保留 |
| D2H pinned nonblocking（idle） | 25.1146 | 2.017% | 严格通过 |

三组 D2H pageable/nonblocking 的真实时长范围：final 初组 **13.846–16.859 s**（CV 7.158%）、final-r2 **14.527–15.606 s**（CV 2.835%）、idle **14.221–14.755 s**（CV 1.511%）。用户明确表示这些不足 15 秒的样本不影响本次交付；本轮据此记录限定例外，不修改原始 elapsed、duration 标志或严格脚本。

实现解释：`MODE=measured` 的单 run 没有与 `MODE=qualification` 完全相同的时长拒绝行为，`qualification.py` 仍对两种模式统一要求 ≥15 秒。不能仅凭 summary passed 或没有 qualification_error 推断严格五次资格通过。

final 根 D2H pinned blocking 初组 CV 9.941%、r2 CV 6.170%，均为 unstable；idle 通过不覆盖它们。所有慢样本保留，主机竞争只能作为与观察一致的解释，不写成已证明的唯一因果。

Base 传输使用整块 Tensor.copy_；1 MiB XRE 分块仅属于 Toolkit 路线。历史 4 GiB pinned async 错误与当前 512 MiB Base 通过范围分别记录。

## 6. 通信、健康与不能外推的结论

| 组 / 物理卡 | 次数 | 最慢 rank 中位数 GB/s | CV | 复核结论 |
|---|---:|---:|---:|---|
| MPI pair / 0,5 | 5 | 28.1996 | 0.121% | 严格资格通过 |
| P2P pair / 0,5 | 5 | 31.1537 | 0.049% | 严格资格通过，单向 |
| MPI six / 0,3,4,5,6,7 | 5 | 33.2643 | 0.206% | 严格资格通过 |
| MPI seven / 0,2,3,4,5,6,7 | 3 | 32.2995 | 0.162% | 有效测量；重复不足且含卡 2 |
| MPI eight / 0–7 | 5 | 64.0284 | 0.971% | 测量/监控/重复性通过，健康 degraded；严格资格拒绝 |

八卡组的 4200 次迭代通过测量窗，卡 1 当次未重现历史挂死。卡 2 的不可纠正 ECC=8 不因测量成功而清除；本次用户接受的例外仅涉 D2H 时长，不扩展为健康例外。七卡不是五次正式组，八卡不是健康资格通过。Base 同机 MPI 代码支持 2–8 rank，不意味着每种卡数/拓扑都有五次实测。

## 7. 指标、报告与发布边界

单向 copy/P2P 为 payload×iterations/time；D2D 为两倍 payload；AllReduce 同时报 algbw 与 busbw，后者为 algbw×2×(n−1)/n。每 run 取最慢 rank，组中位数、样本标准差 ddof=1 和 CV≤5%；不叠加多 rank 带宽，不把容量查询替代实际持有量。

规范输出为每 run 的 `report.md`、`report_monitor.md`、SVG 和原始 JSON。Base qualification 是重复性/证据/资源门禁，不是 Toolkit 厂商 diagnosis。Toolkit 于 2026-10-08 定义的合规校验语义单独维护，不移植到 Base 结论层。

当前代码接受 candidate 显式 opt-in；运行镜像没有自动获得 validated。跨机、FP64/FP8/TF32、FlagGems、Training/Inference/Operation 和厂商理论峰值排名不在本次已验证范围。

## 8. 历史索引异常与证据修订原则

Day 5 `evidence/day5/sha256-index.json` 对 `verification.json` 保存的摘要为 `1a1abab68e1da8824ad6ce709d07105746b9ef3a9cf05fb9b7b258c3c21922ba`；现存已提交文件摘要为 `56fd8d9301774b8606ef3e4a1e1b816ebacbadfecf4b11e5b6090ad90729f273`。其内容是一份归档验证汇总，不能据此宣称整个历史索引零失配。其他 25,391 条 Base 索引匹配。

本轮保留旧索引和文件原始字节，以独立的修订说明、实际摘要与本轮索引记录该例外，不重新签写历史哈希来掩盖差异。这不改变 114 个最终 run 的逐 rank 校验、报告一致性和 7,196 文件不变的结论。

Toolkit `run-records` 是元数据子集，随附 index 指向完整远程 run；其 README 明确禁止用子集核验整套 index。本轮排除这 16 份子集索引，并额外确认完整 integration snapshot 629 条匹配；不把“未镜像全部文件”误报为损坏。

## 9. 交接与后续

可以按本报告限定范围交付 Base。后续维护重点是卡 2 健康修复后的重测、卡 1 故障根因、XPULink 反差及 pinned-blocking 跨批稳定性；它们均保留历史证据。D2H 时长例外按用户决定不列为本次交付阻塞；未来需要严格资格时可固定更长迭代配置后重新五次测量，不能拼接旧样本。

本轮文档修改没有重新运行硬件，也未改变测试实现、门禁、manifest 或历史标签。文档与 klx 代码仓分别本地提交，远端 origin 仍保留本轮开始时的提交，未推送本轮更新。


复现入口：[使用与维护手册](p800-base-handoff.md)。审计明细：[本轮证据](../vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/base-doc-audit-20261008/README.md)。
