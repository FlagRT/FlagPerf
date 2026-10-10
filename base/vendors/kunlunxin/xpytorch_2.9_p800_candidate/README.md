# P800 XPYTORCH 2.9 candidate runtime

本目录保存 P800 Base/Toolkit 使用的运行时身份、依赖锁、设备绑定及检查工具。
运行测试从 [Base 使用指南](../../../README.md) 开始；环境准备与支持范围见
[P800 适配指南](../../../../docs/kunlunxin/README.md)。

## 运行时身份

| 项目 | 锁定值 |
|---|---|
| profile | `xpytorch_2.9_p800_candidate` |
| 镜像 | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` |
| image ID | `sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975` |
| 平台 | `linux/amd64` |
| Python / PyTorch | `3.10.18` / `2.9.0+cu129` |
| Conda prefix | `/root/miniconda/envs/python310_torch29_cuda`，镜像内部路径 |
| 宿主观察基线 | Driver `5.0.21.47`、SMI Runtime `10.2`、头文件 `5.0.21` |
| 镜像厂商元数据 | XPYTORCH `XMLIR--bc1b1dc6f-dev+2026032411`、XRE `5.13.0.0` |
| 发布状态 | `candidate / validated:false`，聚合 validation scope 为空 |

[stack.lock.yaml](stack.lock.yaml) 是 JSON 兼容的锁文件，保存所有包版本、厂商元数据和
bootstrap 约定；[image-manifest.json](image-manifest.json) 保存镜像身份。
锁中的 XHPC 依赖元数据声明了不同版本的 XRE/XTDK，不能据既有实测称其已获厂商一致性认证。
新镜像或驱动应独立验证，不通过改 tag 或跳过检查复用旧结果。

镜像需由使用者自行取得并核对授权，本仓库没有验证匿名 pull，也不分发其底座制品。
镜像名含 CUDA 兼容标识不表示实际执行硬件为 NVIDIA。

## 工具与职责

| 文件 | 用途 |
|---|---|
| `container_bootstrap.sh` | 进入锁定 Conda 环境，隔离 site hooks 并设置受控启动路径 |
| `verify_runtime.py` | 校验镜像、包版本、可见设备及宿主上下文，驱动隔离子进程检查 |
| `device_mapping.py` | 从 SMI PCI/UUID/minor 到实际节点和 framework UUID 的映射 |
| `hardware_probe.py` | 有界的微小 API/身份检查，不作为正式吞吐 benchmark |
| `launch_qualification.py` | 底层 runtime API 探针，保留已有接口；日常身份检查优先用统一 `benchmark preflight` |
| `qualification-record.json` | 带日期和范围的历史探针记录，不是当前全部 Base/Toolkit 的资格声明 |

历史记录中的原始 evidence/result 不随源码分发；相关路径只是来源标识，不是运行前置依赖。
现行功能与历史结果范围以 [公开验证说明](../../../../docs/kunlunxin/validation.md) 为准。

## 设备与环境约定

物理 SMI index、设备节点 minor、framework ordinal 和 rank 分开记录。
每轮先查询所选卡的 PCI、UUID、minor 和字符设备，再获取稳定 UUID 租约，锁内重查。
容器仅映射所选计算节点及 `/dev/xpuctrl`，分配前校验全部可见设备与宿主选择一致。
单卡容器通常枚举为逻辑零，多卡顺序必须以 UUID join 为准，禁止写死对应关系。

bootstrap 根据验证后的可见集合设置 `CUDA_VISIBLE_DEVICES`，真正 unset
`XPU_VISIBLE_DEVICES` 与 `XPU_EVENT_KL3_ENABLE`，并固定 `USE_FLAGGEMS=0`。
通用设备名 `GPU` 或 CUDA 兼容 API 不能替代厂商和 UUID 身份检查。

正式计时使用 `perf_counter`/`perf_counter_ns` 并在边界同步实际目标设备。
早期 CUDA Event elapsed 返回零，因此不用于性能计算。微小探针与 native launch 计数只能
支持对应测试的执行路径，不证明所有内部操作都无 CPU fallback，也不是完整 kernel trace。

Base 容器使用 internal bridge；Toolkit 和底层 API 探针使用各自的无网络策略。
不同入口的权限、预约窗口、超时和证据语义不能互换，统一入口说明见 P800 指南。

## 离线检查

从仓库根目录执行：

```bash
python3 -S -B -m unittest discover \
  -s base/vendors/kunlunxin/xpytorch_2.9_p800_candidate -p 'test_*.py'
bash -n base/vendors/kunlunxin/xpytorch_2.9_p800_candidate/container_bootstrap.sh
```

这些检查覆盖版本锁、设备映射、UUID/minor、可见集、窗口和子进程失败等合同，
不启动真实 benchmark，不替代 Docker 生命周期和硬件验收。
