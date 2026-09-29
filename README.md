
<img width="4750" height="958" alt="github+banner__2025-11-11+13_27_10" src="https://github.com/user-attachments/assets/418d4df4-fccf-45ad-997e-1659fd19b6b8" />


----------
## FlagPerf是什么
[![Lint Code Base](https://github.com/FlagOpen/FlagPerf/actions/workflows/super-linter.yml/badge.svg)](https://github.com/FlagOpen/FlagPerf/actions/workflows/super-linter.yml)

**FlagPerf是智源研究院联合AI硬件厂商共建的一体化AI硬件评测引擎，旨在建立以产业实践为导向的指标体系，评测AI硬件在软件栈组合（模型+框架+编译器）下的实际能力。**

## 第一次使用，从这里开始

先按要测量的对象选择模块，各模块的环境和启动方式不同：

| 你想了解什么 | 阅读入口 |
|---|---|
| 基础算力、内存、互连及厂商诊断 | [Base 入门](base/README.md) |
| 一个算子的正确性、性能与失败原因 | [Operation 入门](operation/README.md) |
| 模型输出差异、整体/逐层性能及组件开关影响 | [Inference 入门](inference/README.md) |
| 模型训练 | [Training 文档](training/README.md) |
| 生成式服务评测 | [Generate 文档](docs/generate/generate-case-doc.md) |

当前 Inference 面向 Qwen3-Embedding-0.6B；支持 Ascend 单卡/单机 TP，NVIDIA 单卡接口尚未实机验证。
Base、Operation、Inference 的当前单机入口不要求 SSH。先阅读所选模块的环境准备，再执行其最小示例。
完整导航见 [docs](docs/README.md)。

## 📣 FlagPerf评测亮点

![cooperation](assets/imgs/overview.png)

1. **构建多维度评测指标体系，不止关注“耗时”:**

   FlagPerf 指标体系除了衡量“芯片能否支持特定模型训练”的功能正确性指标之外，还包含更多维度的性能指标、资源使用指标以及生态适配能力指标等。

   > 指标详细介绍见 [这篇文章](https://mp.weixin.qq.com/s/rwTFsthioBty5W2P-Lg9iw)

2. **支持多样例场景及任务，覆盖大模型训练推理场景**

   FlagPerf 已经涵盖计算机视觉、自然语言处理、语音、多模态等领域的**30余个经典模型，80余个训练样例，**支持评测AI硬件的训练和推理能力，以及大模型场景的推理任务评测。

3. **支持多训练框架及推理引擎，灵活连接AI硬件与软件生态**

   **在训练任务场景中**，除了支持 PyTorch、TensorFlow，FlagPerf 还在积极与 PaddlePaddle、MindSpore 研发团队密切配合。作为国产训练框架的领军者，百度 Paddle团队、华为昇思MindSpore 团队正在将 Llama、GPT3 等明星模型集成至 FlagPerf 测试样例集。

   **在推理任务场景中**，当前 Inference 使用 PyTorch/Transformers eager，提供模型输出与模块输出比较、整体性能及 FlagGems/FlagTree/FlagCX 组件比较。原编译引擎案例见 [旧版恢复说明](docs/inference/migration.md)。

4. **支持多测试环境，综合考察单卡、单机、多机性能**

   为全面评估国产AI芯片多样性、可扩展性、实际应用模拟情况，FlagPerf 设定了单卡、单机（通常是8卡）、多机三个测试环境，为不同的测试环境匹配了不同测试样例场景和任务。

   > 注：当前FlagPerf在保证测试环境除芯片外其他条件一致的情况下，进行芯片本身的离线批处理评测，暂不支持集群和客户端的性能评估。

5. **严格审核参评代码，关注“结果公平”，更关注“过程公正”**

   测试由智源研究院与众多芯片厂商联合展开。总体原则是确保客观、公平地评估芯片的通用性能，限制厂商开展有针对性的定制优化。在确定测试模型之后，首先由芯片厂商进行模型适配，这个过程中**只允许厂商进行分布式通信、批数据量（batch size）等和硬件执行强相关的方面的代码修改**，以确保模型能够在芯片上高效运行。其次由智源研究院依托基准测试平台FlagPerf对芯片能力开展测试，并确保测试过程顺利，芯片性能和稳定性得到最佳发挥。同时，**所有测试代码均已开源，测试过程、数据可复现。**

🎯 未来智源及众多AI硬件、框架团队还将共同拓展FlagPerf的评测场景，如开展集群性能的整体评估，以更全面的评估国产软硬件的性能。

## News 

- [4 Jun 2024]支持算子评测板块. [#562](https://github.com/FlagOpen/FlagPerf/pull/562)
- [20 May 2024]支持FlagPerf在容器内启动评测. [#542](https://github.com/FlagOpen/FlagPerf/pull/542)
- [6 May 2024]支持LLaMA3-8B megatron-core预训练. [#526](https://github.com/FlagOpen/FlagPerf/pull/526)
- [1 Apr 2024]支持基础规格评测板块. [#496](https://github.com/FlagOpen/FlagPerf/pull/496)
- [15 Jan 2024]支持megatron-Llama 70B预训练. [#389](https://github.com/FlagOpen/FlagPerf/pull/389)
- [27 Oct 2023]支持Torch-llama2 7B预训练，[#289](https://github.com/FlagOpen/FlagPerf/pull/289)
- [7 Oct 2023]支持Paddle-GPT3 预训练，[#233](https://github.com/FlagOpen/FlagPerf/pull/233)
- [27 Sep 2023]发布v1.0版本，支持20余个经典模型，50余个训练样例，支持多家芯片厂商的训练或推理评测 [#v1.0](https://github.com/FlagOpen/FlagPerf/releases/tag/1.0)
- [3 Aug 2023]支持推理框架, 支持常见基础模型的离线批推理评测 [#136](https://github.com/FlagOpen/FlagPerf/pull/136)

<details><summary>Full News</summary>

- [31 Oct 2023]支持Torch-Aquila 7B预训练，[#299](https://github.com/FlagOpen/FlagPerf/pull/299)
- [8 Feb 2023]支持Tensorflow框架[#7](https://github.com/FlagOpen/FlagPerf/pull/7)
- [6 Feb 2023]昆仑芯作为合作厂商进入共建生态 [#6](https://github.com/FlagOpen/FlagPerf/pull/6)
- [Dec 2022]天数智芯、百度PaddlePaddle作为最早一批厂商参与初版共建开发

</details>

## 支持列表

基础规格列表：

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>编号</th>
        <th>规格名称</th>
      <th>规格类型</th>
      <th>英伟达</th>
      <th>沐曦</th>
      <th>昇腾</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>1</td>
        <td>FP64算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP64/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-FP64/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>2</td>
        <td>FP32算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP32/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-FP32/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP32/metax">算子或原语</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP32/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>3</td>
        <td>TF32算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-TF32/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-TF32/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-TF32/metax">算子或原语</a></td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>4</td>
        <td>FP16算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP16/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-FP16/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP16/metax">算子或原语</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-FP16/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>5</td>
        <td>BF16算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-BF16/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-BF16/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-BF16/metax">算子或原语</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-BF16/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>6</td>
        <td>INT8算力</td>
      <td>算力</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-INT8/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/computation-INT8/metax">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/computation-INT8/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>7</td>
        <td>主存储带宽</td>
      <td>存储</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/main_memory-bandwidth/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/main_memory-bandwidth/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/main_memory-bandwidth/metax">算子或原语</a></td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>8</td>
        <td>主存储容量</td>
      <td>存储</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/main_memory-capacity/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/main_memory-capacity/nvidia/A100">厂商专用工具</a></td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/main_memory-capacity/metax">算子或原语</a></td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>9</td>
        <td>CPU-芯片互连</td>
      <td>互联</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/interconnect-h2d/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-h2d/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-h2d/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>10</td>
        <td>服务器内P2P直连</td>
      <td>互联</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/interconnect-P2P_intraserver/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-P2P_intraserver/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-P2P_intraserver/ascend">厂商专用工具</a></td>
    </tr>
    <tr>
      <td>11</td>
        <td>服务器内MPI直连</td>
      <td>互联</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/interconnect-MPI_intraserver/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-MPI_intraserver/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>12</td>
        <td>跨服务器P2P直连</td>
      <td>互联</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/interconnect-P2P_interserver/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-P2P_interserver/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>13</td>
        <td>跨服务器MPI直连</td>
      <td>互联</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/benchmarks/interconnect-MPI_interserver/nvidia/A100">算子或原语</a>,<br><a href="https://github.com/FlagOpen/FlagPerf/tree/main/base/toolkits/interconnect-MPI_interserver/nvidia/A100">厂商专用工具</a></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
  </tbody>
  </table>  


算子列表：

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>编号</th>
        <th>规格名称</th>
      <th>算子库</th>
      <th>英伟达</th>
    </tr>
  </thead>
  <tbody>
  <tr>
      <td>1</td>
        <td>mm-FP16</td>
      <td>nativetorch<br>flaggems</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/operation/benchmarks/mm/nvidia">A100_40_SXM</a></td>
    </tr>
    <tr>
      <td>2</td>
        <td>sum-FP32</td>
      <td>nativetorch<br>flaggems</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/operation/benchmarks/sum/nvidia">A100_40_SXM</a></td>
    </tr>
    <tr>
      <td>3</td>
        <td>linear-FP16</td>
      <td>nativetorch<br>flaggems</td>
      <td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/operation/benchmarks/linear/nvidia">A100_40_SXM</a></td>
    </tr>
    <tr>
      <td>...</td>
        <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
</tbody>
</table>

训练列表：
> [!TIP]
> **请在表格下方向右滑动查看更多厂商**

<table>
<tr><th>编号</th><th>模型名称</th><th>模型类型</th><th>英伟达</th><th>沐曦</th><th>昆仑芯</th><th>天数智芯</th><th>摩尔线程</th><th>昇腾</th><th>海光</th></tr>
<tbody><tr><td>1</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama3_70B">LLaMA3-70B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama3_70B-megatron">megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>2</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama3_8B">LLaMA3-8B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama3_8B-megatron">megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>3</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llava1.5_7b">llava1.5-7B</a></td><td>LMM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llava1.5_7b-deepspeed-torch">deep<br/>speed</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>4</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama2_70B">llama2_70b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama2_70B-megatron">megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/llama2_70B-megatron">megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/llama2_70B-megatron">megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>5</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mixtral_8x7B/megatron">Mixtral-8x7B</a></td><td>moeLLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/mixtral_8x7B-megatron">megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>6</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/qwen1.5_MoE">Qwen1.5-moe</a></td><td>moeLLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/qwen1.5_MoE-megatron">PAI-<br/>megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>7</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llava1.5_13b">llava1.5-13B</a></td><td>LMM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llava1.5_13b-deepspeed-torch">deep<br/>speed</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>8</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama2_7b">llama2_7b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama2_7b-deepspeed">deep<br/>speed</a>,<br/><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama2_7b-megatron-deepspeed">megatron-<br/>deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/llama2_7b-megatron-deepspeed">megatron-deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/pull/348">deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/pull/343">deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/pull/354">deep<br/>speed</a></td><td>N/A</td><td>N/A</td></tr><tr><td>9</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/aquila2_70B_container">aquila2_70b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/aquila2_70B_container-in_container">flagscale<br/>megatron</a></td><td>N/A</td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/aquila2_70B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/aquila2_70B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/ascend/aquila2_70B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/dcu/aquila2_70B_container-in_container">flagscale<br/>megatron</a></td></tr><tr><td>10</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/aquila2_7b_finetune">aquila2_7b<br/>(finetune)</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/aquila2_7b_finetune-flagscale">flagscale<br/>megatron</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>11</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/chatglm3_6b">chatglm3_6b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/chatglm3_6b-deepspeed">deep<br/>speed</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>12</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/aquila2_7B_container">aquila2_7b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/aquila2_7b-flagscale">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/ascend/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/dcu/aquila2_7B_container-in_container">flagscale<br/>megatron</a></td></tr><tr><td>13</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/aquila2_34B_container">aquila2_34b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/aquila2_34B_container-in_container">flagscale<br/>megatron</a></td><td>N/A</td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/aquila2_34B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/aquila2_34B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/ascend/aquila2_34B_container-in_container">flagscale<br/>megatron</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/dcu/aquila2_34B_container-in_container">flagscale<br/>megatron</a></td></tr><tr><td>14</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/baichuan2_13b">baichuan2<br/>13b</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/baichuan2_13b-deepspeed">deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/baichuan2_13b-deepspeed">deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/baichuan2_13b-deepspeed">deep<br/>speed</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/baichuan2_13b-deepspeed">deep<br/>speed</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>15</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/glm">glm</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/glm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/glm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/glm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/glm-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/dcu/glm-pytorch">pytorch</a></td></tr><tr><td>16</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/resnet50">resnet50</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/resnet50-pytorch">pytorch</a>,<br/><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/resnet50-tensorflow2">tensorflow2</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/resnet50-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/resnet50-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/resnet50-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/resnet50-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td></tr><tr><td>17</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/retinanet">retinanet</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/retinanet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/retinanet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/retinanet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/retinanet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/retinanet-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td></tr><tr><td>18</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bert_hf">bert_hf</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bert_hf-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/bert_hf-pytorch">pytorch</a></td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bert_hf-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/mthreads/bert_hf-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td></tr><tr><td>19</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bigtransfer">bigtransfer</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bigtransfer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/bigtransfer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bigtransfer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bigtransfer-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>20</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/cpm">cpm</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/cpm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/cpm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/cpm-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/cpm-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>21</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/efficientnet">efficientnet</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/efficientnet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/efficientnet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/efficientnet-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/efficientnet-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>22</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/faster_rcnn">faster_rcnn</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/faster_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/faster_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/faster_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/faster_rcnn-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>23</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mask_rcnn">mask_rcnn</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/mask_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/mask_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/mask_rcnn-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/mask_rcnn-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>24</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/mobilenetv2">mobilenetv2</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/mobilenetv2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/mobilenetv2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/mobilenetv2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/mobilenetv2-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>25</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/swin_transformer">swin<br/>transformer</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/swin_transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/swin_transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/swin_transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/swin_transformer-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>26</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/t5_small">t5_small</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/t5_small-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/t5_small-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/t5_small-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/t5_small-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>27</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/transformer">transformer</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/transformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/transformer-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>28</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/bert">bert</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bert-paddle">paddle</a>,<br/><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/bert-pytorch">pytorch</a></td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bert-paddle">paddle</a>, <a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/bert-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bert-paddle">paddle</a>, <a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/bert-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>29</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/detr">detr</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/detr-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/detr-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/detr-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>30</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/distilbert">distilbert</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/distilbert-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/distilbert-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/distilbert-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>31</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/gpt2">gpt2</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/gpt2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/gpt2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/gpt2-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>32</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/longformer">longformer</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/longformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/longformer-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/longformer-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>33</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/tacotron2">tacotron2</a></td><td>Audio</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/tacotron2-pytorch">pytorch</a></td><td>N/A</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/tacotron2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/tacotron2-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>34</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/transformer_xl">transformer<br/>xl</a></td><td>NLP</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/transformer_xl-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/transformer_xl-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/transformer_xl-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>35</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/vit">vit</a></td><td>CV</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/vit-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/vit-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/vit-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/iluvatar/vit-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>36</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/wav2vec2">wav2vec2</a></td><td>Audio</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/wav2vec2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/wav2vec2-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/kunlunxin/wav2vec2-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>37</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/WaveGlow">WaveGlow</a></td><td>Audio</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/WaveGlow-pytorch">pytorch</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/WaveGlow-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>38</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/dlrmt">DLRM</a></td><td>RS</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/dlrm-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>39</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/gpt3_13B">gpt3_13B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/gpt3_13B-paddle">paddle</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/gpt3_13B-paddle">paddle</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>40</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/gpt3_6.7B">gpt3_6.7B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/gpt3_6.7B-paddle">paddle</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/gpt3_6.7B-paddle">paddle</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>41</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama1_13B">llama1_13B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama1_13B-paddle">paddle</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>42</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama1_7B">llama1_7B</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama1_7B-paddle">paddle</a></td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/metax/llama1_7B-paddle">paddle</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>43</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/llama2_7b_finetune">llama2_7b<br/>finetune</a></td><td>LLM</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/llama2_7b_finetune-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>44</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/benchmarks/moflow">MOFlow</a></td><td>AI4sci</td><td><a href="https://github.com/FlagOpen/FlagPerf/tree/main/training/nvidia/moflow-pytorch">pytorch</a></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
</tbody></table>


当前推理能力：

| 模型 | 执行方式 | 主要能力 | 入口 |
|---|---|---|---|
| Qwen3-Embedding-0.6B | PyTorch/Transformers eager | 模型/模块精度、整体与逐层性能、TP 通信采样、组件比较、报告筛选与图导出 | [Inference](inference/README.md) |

支持与限制以 [能力清单](inference/support.json) 为准；旧模型和编译引擎不属于当前支持范围。

## 如何使用FlagPerf进行AI硬件评测

### 当前单机入口

Base、Operation 和 Inference 的依赖分别在各自指南中准备。宿主负责启动，设备依赖在对应镜像中运行；
无需套用下方历史训练流程的 SSH、host.yaml 或容器启动方式。

### 基础规格评测启动说明

见 [Base 入门](base/README.md)，选择 Benchmark 或厂商 Toolkit，并先查看 `--dry-run` 计划。

### 算子评测启动说明

见 [Operation 入门](operation/README.md)。可先运行 `python3 operation/run.py list` 查看算子，无需设备依赖。

### 训练评测启动说明

1. **下载FlagPerf并部署**

```Bash
# 先各服务器间root帐号的ssh信任关系和sudo免密配置
git clone https://github.com/FlagOpen/FlagPerf.git
cd FlagPerf/training/
pip3 install -r requirements.txt
```

2. **修改机器配置文件**

```Bash
cd Flagperf/training/
vim run_benchmarks/config/cluster_conf.py
```

集群配置文件主要包括集群主机列表和SSH端口，修改`HOSTS`和`SSH_PORT`为机器实际地址

```Bash
'''Cluster configs'''
#Hosts to run the benchmark. Each item is an IP address or a hostname.
HOSTS = ["10.1.2.3", "10.1.2.4", "10.1.2.5", "10.1.2.6"]
#ssh connection port
SSH_PORT = "22"
```

3. **修改模型配置文件**

```Bash
cd Flagperf/training/
vim run_benchmarks/config/test_conf.py
```

必改项：

```Bash
VENDOR = "nvidia" #选择本次运行的硬件
FLAGPERF_PATH="" # FlagPerf项目路径，如"/home/FlagPerf/training"
CASES={} # 本次运行的测例，按照对应模型readme准备好数据，修改模型对应的地址
#如运行"bert:pytorch_1.8:A100:1:8:1": "/raid/home_datasets_ckpt/bert/train/"，需要把:后面的路径替换为本地路径
```

4. **启动测试**

```Bash
python3 ./run_benchmarks/run.py
sudo python3 ./run_benchmarks/run.py
```

5. **查看日志**

```Bash
cd result/run2023XXXX/运行模型/
# ls
round1
# ls round1/
10.1.2.2_noderank0
# cd 10.1.2.2_noderank0/
# ls
cpu_monitor.log     pwr_monitor.log  rank2.out.log  rank5.out.log  start_pytorch_task.log
mem_monitor.log     rank0.out.log    rank3.out.log  rank6.out.log
nvidia_monitor.log  rank1.out.log    rank4.out.log  rank7.out.log


# tail -n 6 rank0.out.log
[PerfLog] {"event": "STEP_END", "value": {"loss": 2.679504871368408, "embedding_average": 0.916015625, "epoch": 1, "end_training": true, "global_steps": 3397, "num_trained_samples": 869632, "learning_rate": 0.000175375, "seq/s": 822.455385237589}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171032, "rank": 0}}
[PerfLog] {"event": "EVALUATE", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171032, "rank": 0}}
[PerfLog] {"event": "EPOCH_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 127, "time_ms": 1669034171159, "rank": 0}}
[PerfLog] {"event": "TRAIN_END", "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 136, "time_ms": 1669034171159, "rank": 0}}
[PerfLog] {"event": "FINISHED", "value": {"e2e_time": 1661.6114165782928, "training_sequences_per_second": 579.0933420700227, "converged": true, "final_loss": 3.066718101501465, "final_mlm_accuracy": 0.920166015625, "raw_train_time": 1501.713, "init_time": 148.937}, "metadata": {"file": "/workspace/flagperf/training/benchmarks/cpm/pytorch/run_pretraining.py", "lineno": 158, "time_ms": 1669034171646, "rank": 0}}
```

### 推理评测启动说明

按 [Inference 入门](inference/README.md) 准备兼容镜像和本地权重，先运行原生路径，再进行组件比较。
配置、TP、preview 续探和结果含义均从该入口下钻。旧版启动方式见 [迁移说明](docs/inference/migration.md)。

### 生成式推理评测启动说明

Generate 使用独立的环境与配置。请先阅读 [Generate 部署说明](generate/README.md) 和 [案例配置](docs/generate/generate-case-doc.md)，准备对应引擎后，在 `generate` 目录运行 `python3 main.py`。

## 参与共建FlagPerf

如需参与共建FlagPerf基础规格、训练、推理评测，请参考详细文档，依次位于[基础规格文档目录](https://github.com/shh2000/FlagPerf/tree/ud0401/docs/base)、[训练文档目录](https://github.com/shh2000/FlagPerf/tree/ud0401/docs/training)、[推理文档目录](docs/inference)。

为了更直观的展示厂商参与共建的实际工作量，下面给出6个已经合并进FlagPerf，面向不同特征厂商的Pull Request。

1. 模型训练适配适配

    - **第一次参与训练**适配工作的内容较多。除了适配模型case外，还需要适配厂商的dockerfile、monitor等，如[#246](https://github.com/FlagOpen/FlagPerf/pull/246)
    - **后续参与训练**适配工作量较小：
        - 如厂商以**cuda兼容**路线设计软硬件，典型适配case [#170](https://github.com/FlagOpen/FlagPerf/pull/170)
        - 如厂商**不兼容cuda**，则需要额外修改后端通信方案等等，典型适配case [#288](https://github.com/FlagOpen/FlagPerf/pull/288)。当case较复杂时，可能需要重写部分计算方式、半精度接口等，如[#158](https://github.com/FlagOpen/FlagPerf/pull/158)。

2. 模型推理适配

   当前模型适配、设备边界及目录职责见 [Inference 扩展说明](docs/inference/inference-case-doc.md)。
   新模型需明确输入、输出、padding 与 TP 分片语义；当前工具不会自动发现原框架的旧案例。

## FlagPerf合作伙伴


![cooperation](assets/imgs/coop.png)

## 许可证

本项目基于Apache 2.0 license。 
<br>本项目的代码来源于不同的代码仓库，关于各模型测试Case的情况，请参考各模型测试Case目录的文档。

## 联系我们

如有疑问，可以发送邮件至flagperf@baai.ac.cn，或在[issue](https://github.com/FlagOpen/FlagPerf/issues)中说明情况




