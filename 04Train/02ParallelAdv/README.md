<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型并行进阶

分布式训练可以将模型训练任务分配到多个计算节点上,从而加速训练过程并处理更大的数据集。模型是一个有机的整体，简单增加机器数量并不能提升算力，需要有并行策略和通信设计，才能实现高效的并行训练。本节将会重点打开业界主流的分布式并行框架 DeepSpeed、Megatron-LM 的核心多维并行的特性来进行原理介绍。

## 内容大纲

| 大纲 | 小节 | 链接 | 状态 |
|:-- |:-- |:-- |:--:|
| 分布式并行 | 01 优化器并行 ZeRO1/2/3 原理  | [PPT](./01DSZero.pdf), [视频](https://www.bilibili.com/video/BV1fb421t7KN) | |
| 分布式并行 | 02 Megatron-LM 代码概览  | [PPT](./02Megatron.pdf), [视频](https://www.bilibili.com/video/BV12J4m1K78y) | |
| 分布式并行 | 03 大模型并行与 GPU 集群配置  | [PPT](./03MGConfig.pdf), [视频](https://www.bilibili.com/video/BV1NH4y1g7w4) | |
| 分布式并行 | 04 Megatron-LM TP 原理  | [PPT](./04MGTPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1yw4m1S71Y) | |
| 分布式并行 | 05 Megatron-LM TP 代码解析  | [PPT](./05MGTPCode.pdf), [视频](https://www.bilibili.com/video/BV1cy411Y7B9) | |
| 分布式并行 | 06 Megatron-LM SP 代码解析  | [PPT](./06MGSPPrinc.pdf), [视频](https://www.bilibili.com/video/BV1EM4m1r7tm) | |
| 分布式并行 | 07 Megatron-LM PP 基本原理  | [PPT](./07MGPPPrinc.pdf), [视频](https://www.bilibili.com/video/BV18f42197Sx) | |
| 分布式并行 | 08 流水并行 1F1B/1F1B Interleaved 原理  | [PPT](./08MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV1aD421g7yZ) | |
| 分布式并行 | 09 Megatron-LM 流水并行 PP 代码解析  | [PPT](./08MGPPCode.pdf), [视频](https://www.bilibili.com/video/BV1hs421g7vN) | |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 并行 实践 :computer: | CODE 01: ZeRO 显存优化实践 | [Markdown](./Code01ZeRO.md), [Jupyter](Code01ZeRO.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code01ZeRO.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 02: Megatron 张量并行复现 | [Markdown](./Code02Megatron.md), [Jupyter](Code02Megatron.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code02Megatron.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 03: Pipeline 并行实践 | [Markdown](./Code03Pipeline.md), [Jupyter](Code03Pipeline.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code03Pipeline.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 04: 专家并行大规模训练 | [Markdown](./Code04Expert.md), [Jupyter](Code04Expert.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/Code04Expert.html) | :white_check_mark: |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
