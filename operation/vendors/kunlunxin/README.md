# Kunlunxin P800 Operation

P800 使用显式 `--runtime-profile p800-xpytorch29` 接入公共 Operation catalog。
当前支持静态规划、运行栈身份约束及宿主身份/租约/清理契约；设备 worker 尚未接通，真实执行和设备复放会明确拒绝。
该状态不代表 P800 算子已实机验证，也不继承 Base 的 validated 结论。
省略 runtime profile 时仍使用原有昆仑芯兼容入口，不改变其他设备的运行栈。

## 静态规划

从 FlagPerf 根目录执行，只需 Linux/Python 3.10+ 标准库，不访问 Docker 或设备：

```bash
python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --device-ids 4 \
  --oplib both --dtype FP32 FP16 BF16 INT32 INT16 BOOL INT64 --dry-run

python3 -B operation/run.py run --vendor kunlunxin \
  --runtime-profile p800-xpytorch29 --execution local --device-ids 1 \
  --case abs --dtype FP32 --oplib both --workload smoke --dry-run
```

第一条生成 728 个组合，其中 308 适用、420 类型不适用；省略 dtype、选择 both 则为 104 项。
Docker 编号是宿主物理号，唯一且在 0..7，没有固定排除卡；local 是当前命名空间可见编号，非负且唯一。
静态计划不会猜测 UUID、minor 或框架编号，Docker 的 worker_device 保持 null，待实际观察。

`--runtime-profile` 选择运行栈；`--workload smoke|daily` 选择工作量，默认 daily。
旧 `--profile` 仍是 workload 的兼容别名，不可与 `--workload` 同用。
case、dtype、size、迭代与正确性协议均沿用[公共入口](../../README.md)。

## 运行身份与边界

[运行栈锁](profiles/p800-xpytorch29.json)记录期望镜像 ID、Python 和包版本，来源为已存在的 Base 锁记录。
镜像可访问性与实际包版本须在使用环境另行核验；镜像 tag 不能代替 ID，显式 `--image` 也须匹配锁定 ID。
没有自动下载镜像或安装依赖。锁中的 Conda 路径属于镜像内部，不是通用宿主路径。

真实 local 运行将使用调用 CLI 的 Python；CPU 测试 venv 不能作为 P800 运行栈。
本阶段仅提供规划示例，不能通过删除 `--dry-run` 获得可用的设备 benchmark。
独立 Operation 锁保持 validated:false；源码快照包含该锁，任务记录 profile 的 SHA-256，未知厂商/profile 组合拒绝。
P800 不接受 `--allow-privileged-root`、附加 `--container-device` 或 `--mount`，也暂不支持 timeline/full 设备 profiling。

## 宿主资源与失败处理

Docker 接入复用 Base 的 P800 provider：锁前读取物理编号、PCI、UUID、minor/节点及占用，按 UUID 获取
`/tmp/flagperf-base-device-leases` 中的资源锁和 `/tmp/flagperf-p800-<UUID>.lock` 兼容锁，持锁重新检查身份与占用。
这些路径是现有 Base/Toolkit 的共享协议；不能给 Operation 单独换目录来规避占锁。
只映射选中设备的实际节点及 `/dev/xpuctrl`，不以物理号推断节点或框架编号。
任务保存宿主绑定，真实框架编号仍须由 worker 观察 UUID 后匹配；静态测试不构成设备映射实测。

宿主检查需要运行账号能够完整读取设备占用、进程句柄和必要 `/proc` 信息；权限不足时应由管理员准备运行账号。
执行器不自动调用 sudo，不把“看不到其他用户进程”视为空闲。宿主运行权限与容器 `--privileged` 是两回事。
只读观察者沿用 Base 的严格识别及显式授权机制，授权名称随证据记录；不内置某台机器的云代理例外。
`P800_ALLOW_FOREIGN_HANDLES=1` 在 Operation 中明确拒绝，不能继承它绕过独占检查。

仅清理本轮创建的容器：通过 run ID 标签核对所有权，随后确认容器已不存在。
启动失败、worker 死亡、watchdog、SIGTERM/中断均进入清理；未确认回收或健康复查失败时停止后续任务及设备诊断。
postflight 在锁释放前执行。清理、健康或锁释放失败进入总状态；释放文件锁不等于宣称设备已恢复健康。
local 只使用当前命名空间的逻辑锁和原有 visibility，报告标记 externally-managed，不声称完成宿主预约或镜像核验。

开发者可在已准备的 CPU 测试环境运行 `python -B -m unittest discover -s operation/tests -p 'test_p800*.py' -v`。
测试使用模拟设备/容器和临时锁目录，覆盖与 Base/Toolkit 的锁冲突、身份变化、失败停止和清理；不会占用实际设备。
