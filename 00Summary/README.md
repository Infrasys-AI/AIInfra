<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 大模型系统概述

本章内容聚焦大模型领域，全面深入地探讨了 Scaling Law 在不同场景如标准、推理时间、并行及扩散等方面的解读，详尽阐述了大模型从训练到推理的全流程，涵盖训练业务流与软硬件栈的各个环节，以及推理业务流的优化策略。同时，剖析了 AI 系统与大模型系统在通用性、资源需求和软件栈变化上的显著区别，并展望了大模型系统的未来发展趋势，包括技术演进、应用场景爆发以及算力底座和生态的持续升级等多方面内容。

## 课程简介

[01](./01ScalingLaw.md)、[02](./02StandardScaling.md) 是对 Scaling Law 的解读，涉及经典 Scaling Law 在不同场景下的应用，如 Standard Scaling、Inference Time Scaling、Parallel Scaling 和 Diffusion Scaling 等相关研究。

[03](./03TTScaling.md)、[04](./05InferStack.md) 是大模型训练推理全流程，涵盖了训练业务流与训练软硬件栈，包括并行、加速，数据、模型结构、预训练、后训练，微调，评估等环节，以及推理业务流与推理软硬件栈，像加速、优化、长序列、输出采样、压缩、量化等方面。包括 AI 系统与大模型系统的区别，从通用性、资源需求（计算、网络、存储）以及软件栈变化（芯片使能、框架、分布式并行、三方套件）三个维度进行了阐述。

[05](./05InferStack.md) 是大模型系统未来发展趋势，探讨了大模型时代的现状与意义，如 Scaling Law 突破、全球竞争格局、产业价值等，分析了技术演进趋势，包含智能体崛起、多模态深度融合、轻量化与高效架构等，展望了应用场景爆发，如生成式重构、工业、医疗、教育等领域，还提到了算力底座持续升级（分布式智算中心发展）以及生态持续构建升级（国产大模型、国产软硬件栈）等情况。

## 备注

系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！

```{toctree}
:maxdepth: 1

01ScalingLaw
02StandardScaling
03TTScaling
04TrainingStack
05InferStack
06Future
```
