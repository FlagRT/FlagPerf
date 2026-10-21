# acblas 已知问题与算子团队移交记录

本页记录 PPU 适配验证中发现的厂商库（`libacblas`，闭源预编译）缺陷与低危观察，按项目约定**原样记录并移交厂商算子团队**；本项目不修改算子实现、不放宽容差、不绕过执行。各条目状态以厂商工具/库原始输出为权威，完整原始输出存内部证据仓快照（`probes/day9` T8/T8b）。

## 复现环境

| 项 | 值 |
|---|---|
| SDK | v2.1.0-a5f865（acblas version 1400） |
| HGGC / Driver / VBIOS | 13.0 / 1.5.1-1d747a / 1.6.6-a35155 |
| 容器镜像 | `flagtree-ppu-3.6-dev:202608`（候选镜像，未验证态） |
| 调用方 | ctypes 直连 `libacblas.so`；PyTorch 2.10.0 GEMM 同样汇入该库 |
| 芯片 | PPU-ZW810E（96GB），单卡 |

## ISS-1 INT8 GEMM 输出错误（acblasGemmEx，computeType=32I，GEMM_DEFAULT）

- **现象**：`acblasGemmEx` 以 `R_8I` 输入、`R_32I` 输出、`computeType=32I(72)`、`algo=GEMM_DEFAULT` 调用，**返回码 0（成功）但输出数值错误**：256×256 正确性核对中 16/16 抽样条目全错，最大相对误差 1.07e9（整数参考为逐元素点积，alpha=1.0/beta=0.0，`OP_N/OP_N`，lda=ldb=ldc=256）。错误不是溢出样式——同一输入在参考实现下精确可复算。
- **API 面边界（顺带记录）**：同 API 以 float 计算type（32F/77/78/75）调用 int8 输入时返回 rc=15 并报 `[acblas_internal.cpp:384] For int8 gemm, Dtype = float is not supported yet!`——该拒绝路径干净，仅 32I 计算type 走到错误输出。
- **对照**：同探针下 FP32/FP64/FP16/TF32/BF16 五个 dtype 的 typed 与 Lt 两族全部通过（err 8.3e-16–1.0e-2，各自容差内），排除调用方式与缓冲区问题。
- **影响**：toolkit `computation-INT8` case 无法注册（按约定不绕过），本框架 INT8 走 `torch._int_mm`（K-major 前置，269.25 TOPS，另行五次资格通过）。

## ISS-2 INT8 acblasLtMatmul 进程崩溃（computeType=32I）

- **现象**：`acblasLtMatmulDescCreate(computeType=72, scaleType=R_32I)`、layout 创建（`R_8I/R_8I/R_32I`，全部 `OP_N`，ld=n）、`acblasLtMatmulAlgoGetHeuristic` 全部 rc=0（返回 8 个候选、workspaceSize=0）；随后 `acblasLtMatmul` 调用**进程 SIGSEGV（rc=-11），stderr 为空**，无任何诊断输出。同探针同批次的 13 个其他变体（含同结构 bf16/fp32 Lt）全部正常返回，崩溃定位在 int8 Lt 执行路径内部（疑似 JIT 编译/内核选择阶段）。
- **影响**：与 ISS-1 共同导致 INT8 计算在 acblas 双路径（typed/Lt）均不可用。

## OBS-1（低危观察，未定级缺陷）多变体单进程退出阶段挂起

- **现象**：单进程内顺序创建/销毁**多个** Lt 管线（多组 desc/layout/heuristic/workspace）完成全部计算并打印收尾标记后，解释器**退出阶段**无限挂起，需 timeout 击杀（EXIT=124）。无数据错误、无越界迹象；单管线进程与 PyTorch 密集调用（2000 次/进程）均不复现。
- **处置**：保留观察，不入缺陷移交主表；若算子团队排查 ISS-2 时顺带复现，可提供探针原始输出（内部证据仓快照 `probes/day10`）。

## 已撤回项（免于重复排查）

- **BF16 `acblasLtMatmul` 曾被怀疑内核腐蚀**：2026-10-20 经对照实验定案为**测试框架自身缺陷**（correctness 以测量规模 layout 跑小规模缓冲，越界毒化上下文；框架侧已修复并全量复测通过），**无需算子团队处理**。BF16 在本 SDK 上 typed/Lt 全部正常（err ≤1.0e-2，134.6–135.9 TFLOPS @8192）。

## 复测协议

SDK 升级后按本页参数重跑 T8/T8b 探针（ctypes 直连，n=256 正确性 + n=8192 计时，13 变体含全部对照）；ISS 状态翻转须以原始输出为准并同步更新本页与 `validation.md`。
