# Kunlunxin P800 Operation

[返回 Operation 入门](../../README.md) · [机制说明](../../../docs/kunlunxin/operation-design.md) · [验证与已知问题](../../../docs/kunlunxin/operation-validation.md)

使用 `--vendor kunlunxin --runtime-profile p800-xpytorch29` 选择 P800 + XPyTorch 2.9 专用适配器及运行栈。
profile 用于与原有 R300p/旧昆仑芯入口并存；省略后仍走旧兼容入口，不会自动选择 P800。
Ascend 直接由 `--vendor ascend` 选择专用适配器，不使用这个 P800 profile。
支持 Docker/local、原生 PyTorch 与显式 FlagGems；Operation 独立锁保持 **`validated:false`**，不沿用 Base 发布状态。

## 环境与最小运行

所有命令从 FlagPerf 仓库根目录执行。`list` 和 `run --dry-run` 只需 Linux/Python 3.10+ 标准库，
不访问 Docker 或设备。实际 Docker 执行需要兼容驱动、Docker 权限、`xpu-smi`、`fuser`、
完整宿主 `/proc` 句柄读取权限，以及可写的结果目录和共享锁；执行器不自动提权或清理他人进程。
锁定镜像、Python 和依赖版本见 [独立运行栈锁](profiles/p800-xpytorch29.json)。
本仓库不分发镜像，也没有验证匿名 registry pull；镜像取得、离线导入和身份核对见 [机制说明](../../../docs/kunlunxin/operation-design.md#镜像取得与身份)。

```bash
# 查看公共案例与类型；不是硬件通过清单。
python3 operation/run.py list --case abs

# 先预览一个原生案例；0 是示例，替换为实际获准使用的物理设备。
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --workload smoke --dry-run

# 环境、设备和权限满足后执行同一案例。
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 0 \
  --case abs --dtype FP32 --workload smoke
```

`--oplib nativetorch` 为默认原生路径；增加 `--oplib both` 可在同次运行中比较原生与 FlagGems。
`--workload smoke|daily` 控制工作量，默认 daily；`--profiling off|timeline|full` 控制独立采集，默认 off。
旧 `--profile` 是 workload 的兼容别名，不能与 `--workload` 同时传入；它与 runtime profile 不同。
全类型单卡双路径计划为 728 项：308 适用、420 类型不适用；完整规划和 timeline 示例见机制说明。

## 设备与权限

Docker 物理编号须唯一且在 0..7，无固定排除卡；执行器按宿主 UUID/PCI/minor 与框架实际 UUID 重新绑定。
**不要增加 Ascend 示例中的 `--allow-privileged-root`**；P800 拒绝该参数及额外 `--container-device`、`--mount`。
宿主 root 与 Docker privileged 不是同一概念。root 启动、`P800_KNOWN_READONLY_MONITORS=barad_agent`
只是 klx 站点记录，不是通用要求；只读观察者必须通过进程身份核验。

默认要求独占并持有与 Base/Toolkit 共用的物理 UUID 租约；SMI 显存/利用率为零不代表没有进程持有设备节点。
只有获准共享时才增加 `--allow-shared-device`，记录占用且性能保持 partial；不能据共享计时给出独占性能资格。
不通过修改白名单、删除锁或终止他人进程获得“空闲”。

## local 运行

在调用者的宿主或容器中准备锁定版本的 Python/依赖/动态库，并在外部预约设备。
`OP_LOCAL_PYTHON` 必须指向该环境的 Python 绝对路径，CPU 测试 venv 不适用。
local 的编号为该环境当前可见的逻辑编号；仅提供当前命名空间的逻辑锁，不替代宿主预约或镜像核验。

```bash
: "${OP_LOCAL_PYTHON:?Set the verified Python absolute path}"
"$OP_LOCAL_PYTHON" -B -S operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --execution local --device-ids 0 \
  --case abs --dtype FP32 --oplib both --workload smoke \
  --result-root operation/result/p800-local --dry-run
```

确认计划和资源后去掉 `--dry-run`；共享需另加授权参数。`-S` 用于在 site hooks 执行前进行运行栈检查。
多个 local run 分别读报告，不通过公共 Docker 聚合自动合并。

## 正确性、路由与诊断边界

两条实现路径独立对 CPU 参考检查；数值超差、编译失败和运行异常保留输入、实际输出（如已生成）与错误栈，
不替换算子、放宽容差或修改输入来取得通过。原生 PyTorch 也会调用厂商后端，不能据“原生”推断数值必然通过。
路由检查在独立 probe 的首次/缓存调用取证，缺 kernel 事件保持 partial；这表示证据不足，不表示设备丢失。
allocator memory 与设备 timeline 已接入；full 的六组硬件计数器尚无可信采集器和语义映射，报告 not-supported，
**这不是已证明硬件不支持**。状态、责任边界和复测结果见 [验证与已知问题](../../../docs/kunlunxin/operation-validation.md)。

实际运行生成 `operation/result/operation-<id>/report.md`，终端打印 Report/Summary/Results；dry-run 仅输出计划。
正确性、路由、性能和诊断分别读取；诊断完成或报告生成成功不等于原任务通过。
离线报告和显式同输入复放见 [公共诊断指南](../../../docs/operations/diagnosis.md)。

## 代码职责

- `p800.py`、`profiles/p800-xpytorch29.json`：适配入口、运行身份与能力边界。
- `bootstrap.py`、`execution.py`：启动检查、CPU 隔离与实际输入/模块/设备核验。
- `host.py`：宿主身份、物理租约、占用、健康与本轮 worker 清理。
- `dispatch.py`、`routing.py`：独立首次/缓存调用的 ATen、launch 与设备 kernel 关联。
- `profiling.py`：独立 allocator 外的设备时间线及计数器支持状态；公共编排/报告位于 `operation/runtime/`。
