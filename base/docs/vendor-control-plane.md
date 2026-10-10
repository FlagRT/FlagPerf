# 厂商控制面与 Case 合同

本文面向适配维护者。日常运行见 [Base 指南](../README.md)，环境及支持范围见
[Ascend](../../docs/ascend/README.md) 和 [P800](../../docs/kunlunxin/README.md)。

## 职责划分

`vendors/registry.py` 显式注册 `ascend` 和 `kunlunxin`；不通过设备名称自动猜测厂商。
`vendors/protocol.py` 定义 `VendorProvider` 和不可变的 `DeviceBinding`。

| 组件 | 职责 |
|---|---|
| provider | runtime/profile、镜像校验、设备身份与绑定、容器权限、bootstrap、监控策略 |
| executor | 参数和静态计划、资源租约、容器生命周期、证据收集、清理和状态合成 |
| Case resolver | generic/vendor/chip/override 配置、入口、环境脚本、requirements 与资产哈希 |
| worker/driver | 核对资产和设备绑定，执行具体 workload、正确性及计时 |
| report | 从已有证据生成阅读视图，不访问设备或重新判定历史硬件结果 |

静态规划只读取源码配置。它不得调用 Docker/SMI、占用设备、获取租约、导入设备 runtime、
启动监控或创建结果目录。宿主解析 Benchmark YAML 需要 PyYAML；容器内框架依赖由 runtime 锁管理。

## 设备选择与租约

Benchmark 的 `--physical-device-ids`、Ascend 兼容的 `--npu-ids` 和逻辑
`--device-ids` 互斥。P800 只使用显式物理选择；Ascend Toolkit 使用 NPU 或逻辑 Device 选择。
物理选择保留请求顺序，绑定分别记录物理 ID、framework logical ID、rank、节点、PCI/UUID、
请求顺序和稳定 resource key。不能把 local rank 直接当作 P800 的 framework ordinal。

新租约按 vendor/resource identity 的哈希命名，以稳定顺序获取。
Ascend 同时获取旧逻辑设备锁，以继续与原 Toolkit/兼容入口互锁；P800 保持 UUID 锁协议兼容。
部分获取失败或异常退出仅释放本轮已获取的锁。`flock` 随进程退出释放，不代表设备自动空闲。
锁是协作机制，不能替代预约；禁止删除他人的锁文件或修改共享锁权限绕过互斥。

P800 使用有界生命周期：镜像与宿主身份检查、锁内复查、容器规格核对、容器 UUID 绑定、
执行与监控、精确 CID 清理、锁内 postflight、租约释放。清理不确定需保留失败和 recovery-required，
不能自动换卡或 reset。Ascend 保留自己的既有执行路径；两家厂商不共用硬件门禁结论。

## 配置合并与入口

`benchmarks/case_assets.py` 由宿主、worker 和 Case 共同使用。
配置顺序为 `generic < vendor < chip < override`，入口选择存在的最具体 `main.py`，
环境脚本按 generic/vendor/chip 顺序执行。芯片层通过 `--case CASE:CHIP` 显式选择。

`--case-config` 是最终覆盖层，省略的键继承前面层次，不替换整份厂商配置。
完整旧配置仍可使用；旧用户若依赖“省略键即删除默认值”，需要显式检查合并后的计划。
override 在执行时保存为 `case-config/override.yaml`，`case-assets.json` 记录相对路径、
内容哈希与有效值。worker 重新解析并核对，资产变化即拒绝。

Case 层次中的路径和符号链接不得逃逸 Case 目录；显式 override 可位于仓库外，执行前需快照。
只有 legacy NVIDIA 路径保留 A100 默认芯片回退。

每个可执行入口需有合法 runtime requirements。子层可以收紧支持状态或最低 rank，
不能放宽父层限制；缺失/格式错误是配置错误。显式 unsupported Case 在设备/Docker/租约访问前
输出 skipped，并保留静态计划和原因。工作量 override 不能绕过 runtime、rank 和支持门禁。

## 结果与兼容

| 证据 | 当前 schema | 兼容方式 |
|---|---:|---|
| Benchmark summary | 3 | 共享 renderer 继续读取旧 Ascend 1/2 |
| Benchmark monitor summary | 2 | 保留 monitor 1 读取 |
| Benchmark result | 1 | 各 Case 数值语义保持独立 |
| 报告元数据 | 3 | 重新渲染不改写实验状态 |
| Case assets | 1 | host/worker/Case 使用一致的资产合同 |
| P800 preflight summary | 1 | `kind=benchmark-preflight`，身份观察与性能测量分开 |

监控事件关联显式设备绑定和精确窗口；缺少/重复 rank、采样不足或字段无效时不得判完整。
历史证据缺少身份时标为缺失，不从当前配置补猜。
退出码为 passed/skipped 0、failed 1、partial 2；CLI 配置/授权错误也可为 2，须结合错误信息读取。
监控关闭为 not-run；报告失败与实验状态分别记录。

Toolkit 的统一报告入口按 suite 分派。P800 的原生测量、监控、合规诊断合同与 Ascend DMI
合同独立，不把字段名称相似当作测量等价。离线重建会更新派生报告和元数据；冻结结果的审计使用副本。

## 回归要求

新增厂商或修改共享逻辑时，同时验证：无设备规划、原厂商参数与 Case 入口、配置覆盖、
unsupported 提前跳过、租约互斥及异常释放、设备绑定、监控窗口、旧 schema 报告读取。
验证命令见 [Ascend 验证](../../docs/ascend/validation.md) 与 [P800 验证](../../docs/kunlunxin/validation.md)。
CPU/mock 通过说明执行合同，不能表述为硬件性能或故障恢复实机认证。
