# 验证与维护

For the 2026-09-20 PR1 control-plane refactor, see the separate
[CPU-only validation record](../../base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/evidence/day2/review.md)
and [migration guide](../../base/docs/vendor-control-plane.md). Those tests use
an isolated CPU PyTorch environment on klx; they do not supersede the historical
locked-runtime or hardware evidence below. Toolkit test injection follows the
moved preflight implementation, and the legacy command remains covered.


## 本次验证口径

2026-09-12 的集成验证包括：源码/配置等价检查、公开入口规划、配置哈希与资格范围绑定、
报告契约、文档相对链接，以及锁定镜像中的 Base、Toolkit、Operation 离线回归。
容器禁网，目标源码只读挂载，未映射 NPU。验证结果不包含本次硬件性能结论。

| 检查 | 结果 | 环境 |
|---|---|---|
| Base 离线回归 | 75/75 通过 | 锁定 communication 镜像 |
| Ascend Toolkit 离线回归 | 76/76 通过 | 锁定 operator 镜像；宿主标准库检查也通过 |
| Operation 离线回归 | 43/43 通过 | 锁定 operator 镜像，含 CPU Tensor 检查 |
| 公开分发契约 | 7/7 通过 | 宿主 Python；无 Git、相邻仓库及结果目录的导出树也通过 |
| 同步等价性 | 52 个 Operation Case 的 AST 一致，20 份变更配置的有效字段一致 | 与选定来源快照比较，忽略许可头、注释和正式命名 |

Base Case 主程序与来源逐字节一致。上述检查合计 201 项测试通过，GitHub 远端 CI 尚未执行。

历史实机范围及残余失败见 [Ascend 指南](README.md)；
Operation 的数值失败、依赖阻塞和路由 partial 保留。Toolkit 和 Base 的结果 schema、
计时公式、监控与诊断状态继续独立。日志解析的通过不能替代设备测量通过。

## 任何克隆均可运行的检查

宿主 Python 3.11 或更新版本，不需要 Torch、Docker、NPU、相邻源码仓库或私有结果目录：

```bash
python3 -m unittest discover -s tests -p test_public_distribution.py -v
python3 -m unittest discover -s base/toolkits/_common/ascend/A3/tests -v
```

公开分发检查覆盖各厂商静态计划、52 项目录、P2P 计划、允许配置哈希、公开摘要计数和文档链接。
它只确认静态接口，不能据此宣布其他厂商硬件通过。对应
[CI workflow](../../.github/workflows/ascend-contracts.yml)使用只读权限，不获取设备或私有镜像；
提交到远端前，本地成功不能称为 GitHub CI 成功。

## 锁定运行环境中的完整离线回归

在已经准备且核对 ID 的 operator 镜像中执行以下命令：

```bash
python3 -m unittest discover -s base/tests -v
python3 -m unittest discover -s base/toolkits/_common/ascend/A3/tests -v
python3 -m unittest discover -s operation/tests -v
```

P2P 相关 Base 回归也应在锁定 communication 镜像中执行。宿主源码只读挂载并使用
`--network=none`，省略所有 NPU 映射；这样验证的是 CPU、配置和执行契约。
不要在同一个 Python 进程混合发现 Base 和 Operation 测试，它们沿用原框架的顶层模块名。

代码中包含 numpy、Torch 和 PyYAML 依赖的检查必须以这组运行环境为准；没有装这些依赖的宿主
即使某些测试跳过，也不能报为完整回归通过。新协议检查应验证真实用户可观察的失败行为，
如镜像/配置漂移拒绝、证据缺失、错误 rank、占用冲突、清理失败和报告损坏。

## 新硬件或新镜像的验收

先核对镜像 ID、CANN、ToolBox、驱动和源码 pin，再通过实时 preflight 选择已预约设备。
先运行单 Case 功能检查，再按需求执行固定协议；每次保存有效配置、rank/设备映射、
计时范围、fallback、monitor、pre/postflight 和原始证据哈希。
短时默认配置逐项复验；容量/OOM、P2P 资格矩阵及故障注入分别安排资源。
停止或失败时确认精确容器和设备已释放，不能删除其他人的 lease 或终止外部任务。

性能比较必须对齐物理设备范围、workload、软件栈、计时边界和公式。
特别是 P2P 原公式带 `2×`，Toolkit 的厂商原值不可未经对齐就直接比较。
原始校准摘要不可用时，公开记录可用于静态复核；独立性能认证需要重新实测或取得原始证据。

## 发布与文档管理

原 Case 和 vendor 边界是适配的主要落点；不要为了统一入口重写全部执行器。
目录名描述用途，协议版本表示测量契约，runtime 版本表示软件身份。改变数值语义、
shape、warmup、ITERS、方向或阈值时，明确记录影响并重新验证，不能只更新哈希。

维护顺序为：修改代码/配置，更新领域指南，更新验证范围和必要摘要，记录
[CHANGELOG](../CHANGELOG.md)，最后复核待发布文件及提交范围。保留原许可证与版权，
依赖源码/patch 的许可证及镜像底座访问、再分发权限应单独确认。

实验目录、模型、输入张量、缓存、凭据和个人配置不应进入发布文件集。
检查工作树之外，也要检查拟发布的 Git 历史；删除当前文件不会自动抹去旧提交中的内容。
若发现真实凭据，应先撤销或轮换，再由维护者决定是否协调历史清理，见
[GitHub 敏感数据处理说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。
需要分发大型证据时应使用单独的制品存储并提供摘要和校验值，避免直接塞入代码仓库，见
[GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

本次不自动改写已有提交历史，不发布镜像，也不改变 GitHub 仓库设置。
当前公开运行环境仍受基础镜像获取权限和未发布 registry digest 限制；这项限制在环境指南明确说明。
