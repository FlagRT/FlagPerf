# Kunlunxin P800 Operation

P800 使用显式 `--runtime-profile p800-xpytorch29` 接入公共 Operation catalog。
当前支持静态规划与运行栈身份约束；设备 worker 尚未接通，真实执行和设备复放会明确拒绝。
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
