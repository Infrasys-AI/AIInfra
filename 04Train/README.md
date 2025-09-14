<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra/)版权许可-->

# 大模型训练概述

大模型训练的核心特点在于大规模分布式训练和高效 AI 框架的协同。分布式训练通过数据并行、模型并行等技术，将计算任务分散到多个 GPU 或节点，显著提升训练速度与规模。AI 框架（如 PyTorch）提供分布式支持、混合精度计算和梯度优化，确保高效资源利用与稳定收敛。两者结合，使训练千亿级参数的模型成为可能，同时降低硬件成本与能耗。

## 课程位置

![AI Infra](./images/arch01.png)

## 课程简介

- [**《1. 分布式并行基础》**](./01ParallelBegin/)：大模型分布式并行通过数据并行、模型并行和流水线并行等策略，将计算任务分布到多个设备上，以解决单设备内存和算力不足的问题。数据并行复制模型，分发数据；模型并行分割参数；流水线并行分阶段处理。混合并行结合多种方法优化效率，同时需解决通信开销和负载均衡等挑战，提升训练速度与扩展性。

| 大纲 | 小节 | 链接| 状态 |
|:-- |:-- |:-- |:--: |
| 分布式并行 | 01 分布式并行框架介绍  | [PPT](./01ParallelBegin/01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1op421C7wp) | |
| 分布式并行 | 02 DeepSpeed 介绍  | [PPT](./01ParallelBegin/02DeepSpeed.pdf), [视频](https://www.bilibili.com/video/BV1tH4y1J7bm) | |
| 并行 实践 :computer: | CODE 01: CODE 01: 从零构建 PyTorch DDP | [Markdown](./01ParallelBegin/Code01DDP.md), [Jupyter](./01ParallelBegin/Code01DDP.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train01ParallelBegin/Code01DDP.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 02: CODE 01: PyTorch 实现模型并行 | [Markdown](./01ParallelBegin/Code02MP.md), [Jupyter](./01ParallelBegin/Code02MP.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train01ParallelBegin/Code02MP.html) | :white_check_mark: |

- [**《2. 分布式并行进阶》**](./02ParallelAdv/)：大模型分布式并行通过数据并行、模型并行和流水线并行等策略，将计算任务分布到多个设备上，以解决单设备内存和算力不足的问题。数据并行复制模型，分发数据；模型并行分割参数；流水线并行分阶段处理。混合并行结合多种方法优化效率，同时需解决通信开销和负载均衡等挑战，提升训练速度与扩展性。

| 大纲 | 小节 | 链接 | 状态 |
|:-- |:-- |:-- |:--:|
| 分布式并行 | 01 优化器并行 ZeRO1/2/3 原理  | [PPT](./02ParallelAdv/01DSZero.pdf), [视频](https://www.bilibili.com/video/BV1fb421t7KN) | |
| 分布式并行 | 02 Megatron-LM 代码概览  | [PPT](./02ParallelAdv/02Megatron.pdf), [视频](https://www.bilibili.com/video/BV12J4m1K78y) | |
| 分布式并行 | 03 大模型并行与 GPU 集群配置  | [PPT](./02ParallelAdv/03MGConfig.pdf), [视频](https://www.bilibili.com/video/BV1NH4y1g7w4) | |
| 分布式并行 | 04 Megatron-LM TP 原理  | [PPT](./02ParallelAdv/04MGTPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1yw4m1S71Y) | |
| 分布式并行 | 05 Megatron-LM TP 代码解析  | [PPT](./02ParallelAdv/05MGTPCode.pdf), [视频](https://www.bilibili.com/video/BV1cy411Y7B9) | |
| 分布式并行 | 06 Megatron-LM SP 代码解析  | [PPT](./02ParallelAdv/06MGSPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1EM4m1r7tm) | |
| 分布式并行 | 07 Megatron-LM PP 基本原理  | [PPT](./02ParallelAdv/07MGPPPrinc.pdf), [视频](https://www.bilibili.com/video/BV18f42197Sx) | |
| 分布式并行 | 08 流水并行 1F1B/1F1B Interleaved 原理  | [PPT](./02ParallelAdv/08MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV1aD421g7yZ) | |
| 分布式并行 | 09 Megatron-LM 流水并行 PP 代码解析  | [PPT](./02ParallelAdv/08MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV1hs421g7vN) | |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 并行 实践 :computer: | CODE 01: ZeRO 显存优化实践 | [Markdown](./02ParallelAdv/Code01ZeRO.md), [Jupyter](./02ParallelAdv/Code01ZeRO.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code01ZeRO.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 02: Megatron 张量并行复现 | [Markdown](./02ParallelAdv/Code02Megatron.md), [Jupyter](./02ParallelAdv/Code02Megatron.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code02Megatron.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 03: Pipeline 并行实践 | [Markdown](./02ParallelAdv/Code03Pipeline.md), [Jupyter](./02ParallelAdv/Code03Pipeline.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code03Pipeline.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 04: 专家并行大规模训练 | [Markdown](./02ParallelAdv/Code04Expert.md), [Jupyter](./02ParallelAdv/Code04Expert.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code04Expert.html) | :white_check_mark: |

- [**《PyTorch 框架》**](./02PyTorch/)：PyTorch 在大模型时代以动态计算图为核心，提供灵活性和易用性，支持自动微分与 GPU 加速。其模块化设计便于扩展，兼容分布式训练（如 torch.distributed），助力数据、模型和流水线并行。通过 TorchScript 支持静态图部署，结合生态系统（如 Hugging Face、DeepSpeed），优化大规模模型的训练与推理效率，满足高性能需求。


- [**《模型微调与后训练》**](./03Finetune/)：大模型微调与后训练旨在适应特定任务或领域，通过调整预训练模型参数或部分参数实现高效迁移。微调通常使用小规模标注数据，更新全量或部分参数；后训练则在大规模未标注数据上继续训练，增强泛化能力。两者均需权衡计算成本与性能，常结合技术如 LoRA、量化等优化效率，同时避免过拟合和灾难性遗忘问题。

希望这个系列能够给朋友们带来一些帮助，也希望 ZOMI 能够继续坚持完成所有内容哈！欢迎您也参与到这个开源课程的贡献！

## 课程知识

![AI Infra](./images/arch02.png)

## 备注

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra/)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
> 
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
>
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交 PR 到开源社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！
