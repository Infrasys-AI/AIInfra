<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 分布式并行基础

分布式训练可以将模型训练任务分配到多个计算节点上,从而加速训练过程并处理更大的数据集。模型是一个有机的整体，简单增加机器数量并不能提升算力，需要有并行策略和通信设计，才能实现高效的并行训练。本节将会重点打开业界主流的分布式并行框架 DeepSpeed、Megatron-LM 的核心多维并行的特性来进行原理介绍。

## 内容大纲

| 大纲 | 小节 | 链接| 状态 |
|:-- |:-- |:-- |:---- |
| 分布式并行 | 01 分布式并行框架介绍  | [PPT](./01Introduction.pdf), [视频](https://www.bilibili.com/video/BV1op421C7wp) | |
| 分布式并行 | 02 DeepSpeed 介绍  | [PPT](./02DeepSpeed.pdf), [视频](https://www.bilibili.com/video/BV1tH4y1J7bm) | |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 并行 实践 :computer: | CODE 01: 从零构建 PyTorch DDP | [Markdown](./Code01DDP.md), [Jupyter](./Code01DDP.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train01ParallelBegin/Code01DDP.html) | :white_check_mark: |
| 并行 实践 :computer: | CODE 02: PyTorch 实现模型并行 | [Markdown](./Code02MP.md), [Jupyter](./Code02MP.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/04Train01ParallelBegin/Code02MP.html) | :white_check_mark: |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
