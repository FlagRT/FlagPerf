# P800 Base 文档复核证据 — 2026-10-08

本目录是对 klx `FlagPerf@be0f2ea5` 已有证据的只读复核附件。未生成新的硬件测量。运行原件保留在 `base/result/p800-base-final-20261008/` 与 `p800-idle-accept-20261008/`。

| 文件 | 用途 |
|---|---|
| audit-summary.json | 复核总览、测试计数与用户接受的 D2H 时长例外 |
| runs.json | 114 个 run 的计时、层状态、rank 数与健康事实 |
| groups.json | 22 个重复组的严格脚本结果及独立项目交付结论 |
| source-sha256.json / source-unchanged.json | 7,196 原始文件摘要和复核前后不变证明 |
| archive-index-audit.json | 6 个 Base index 与 1 个完整 Toolkit snapshot；Day 5 一处差异保留 |
| report-verification.json | 114 run 的报告确定性与已存报告对照 |
| dry-run-matrix.json | 15 case 的 10 applicable / 5 skipped 及原因 |
| base.log / ascend.log / pr0.log | 252 / 80 / 32 测试日志，负例中的 ERROR/WARNING 不等于测试失败 |
| audit.py / verify.py | 本轮远程只读复核脚本；结果写入独立审计目录，报告在临时副本生成 |
| sha256-index.json | 本目录交付文件索引，自身除外 |

用户接受的时长例外只适用于三组 D2H pageable/nonblocking，不改变严格 qualification.py、CV 失败或 ECC 健康结论。分组项目结论为人工交付政策记录，不冒充现行代码自动判定。

Day 5 verification.json 的旧索引摘要和实存摘要都记录在 archive-index-audit.json；旧文件和旧 index 不改写。Toolkit run-records 的 16 份复制索引描述完整远程 run，README 已说明它们不是元数据子集的完整性清单，因此排除该误用。

source-sha256 保存的是本轮观察时的摘要，不能单独证明从测量日至本轮间无人修改；本轮另与原始 metric/correctness 合同、已存报告和历史索引交叉核查。

本地修订前文档快照保存在独立项目文档仓，不复制到 FlagPerf。
