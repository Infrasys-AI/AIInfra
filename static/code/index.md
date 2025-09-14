---
title: AI Infra 
---

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 4

```{image} ./_static/logo-square.svg
:width: 150px
```
:::
::::

本开源项目主要是跟大家一起探讨和学习人工智能、深度学习的系统设计，而整个系统是围绕着在 NVIDIA、ASCEND 等芯片厂商构建算力层面，所用到的、积累、梳理得到大模型系统全栈的内容。希望跟所有关注 AI 开源项目的好朋友一起探讨研究，共同促进学习讨论。

![大模型系统全栈架构图](images/01Introduction/03Architecture03.png)

# 课程内容大纲

课程主要包括以下八大模块：

第一部分，对大模型系统和本课程内容进行系统概述[<u>**大模型系统概述**</u>](./00Summary/README.md)，

第二部分，AI 计算集群的介绍[<u>**AI 计算集群**</u>](./01AICluster/README.md)，主要是整体了解 AI 计算集群内容。

第三部分，通信与存储的介绍[<u>**通信与存储**</u>](./02StorComm/README.md)，大模型训练和推理的过程中都严重依赖于网络通信，因此会重点介绍通信原理、网络拓扑、组网方案、高速互联通信的内容。存储则是会从节点内的存储到存储 POD 进行介绍。

第四部分，集群中容器和云原生技术的介绍[<u>**集群容器与云原生**</u>](./03DockCloud/README.md)，从容器、云原生时代到 Docker 和 K8S 技术的应用，这其中包含实践内容： K8S 集群搭建与实践。

第五部分，涉及到大模型，就不得不提大模型训练[<u>**大模型训练**</u>](./04Train/README.md)，训练的基础是并行，加速是核心，后训练、强化学习和微调是关键，验证评估是目的。

第六部分，当下大模型的热点之一：推理[<u>**大模型推理**</u>](./05Infer/README.md)，首先介绍推理的基本概念，其次介绍如何对推理进行加速，之后从架构层次进行调度加速，输出采样，针对大模型进行压缩，这其中包含以下三个实践：1.长序列推理；2.输出采样；3.大模型压缩。

第七部分，介绍大模型所使用的算法和数据结构[<u>**大模型算法与数据**</u>](./06AlgoData/README.md)，首先介绍 Transformer 与 MOE 架构，之后针对图文生成与理解、视频语音大模型和数据工程进行介绍。

第八部分，介绍大模型在各行各界应用的介绍[<u>**大模型应用**</u>](./07Application/README.md)，首先介绍大模型的典型应用场景，之后深入进阶应用，接着梳理大模型应用面临的挑战和伦理问题，最后进行未来展望。



# 课程设立目的

本课程主要为本科生高年级、硕博研究生、大模型系统从业者设计，帮助大家：

1. 完整了解大模型，并通过实际问题和案例，来了解大模型的系统设计。

2. 介绍前沿系统架构和 AI 相结合的研究工作，了解主流框架、平台和工具来了解大模型系统。

**先修课程:** C++/Python，计算机体系结构，人工智能基础

# 课程目录内容

<!-- ## 一. 大模型系统概述 -->

```{toctree}
:maxdepth: 1
:caption: ===  大模型系统概述 ===

00Summary/README
```

<!-- ## 一. AI 计算集群 -->

```{toctree}
:maxdepth: 1
:caption: === 一. AI 计算集群 ===

01AICluster/README
01AICluster01Roadmap/README
01AICluster02L0L1Base/README
01AICluster03SuperPod/README
01AICluster04Performance/README
```

<!-- ## 二. 通信与存储 -->

```{toctree}
:maxdepth: 1
:caption: === 二. 通信与存储 ===

02StorComm/README
02StorComm01Roadmap/README
02StorComm02NetworkComm/README
02StorComm03CollectComm/README
02StorComm04CommLibrary/README
02StorComm05StorforAI/README
```

<!-- ## 三. 集群容器与云原生 -->

```{toctree}
:maxdepth: 1
:caption: === 三. 集群容器与云原生 ===

03DockCloud/README
03DockCloud01Roadmap/README
03DockCloud02DockerK8s/README
03DockCloud03DiveintoK8s/README
03DockCloud04CloudforAI/README
```

<!-- ## 四. 大模型训练 -->

```{toctree}
:maxdepth: 1
:caption: === 四. 大模型训练 ===

04Train/README
04Train01ParallelBegin/README
04Train02ParallelAdv/README
04Train03TrainAcceler/README
04Train04PostTrainRL/README
04Train05FineTune/README
04Train06VerifValid/README
```

<!-- ## 五. 大模型推理 -->

```{toctree}
:maxdepth: 1
:caption: === 五. 大模型推理 ===

05Infer/README
05Infer01Foundation/README
05Infer02InferSpeedUp/README
05Infer03Dispatch/README
05Infer04LongInfer/README
05Infer05Sampling/README
05Infer06CompDistill/README
05Infer07Framework/README
05Infer08DeepSeek/README
```

<!-- ## 六. 大模型算法与数据 -->

```{toctree}
:maxdepth: 1
:caption: === 六. 大模型算法与数据 ===

06AlgoData/README
06AlgoData01Basic/README
06AlgoData02MoE/README
06AlgoData03NewArch/README
06AlgoData04ImageTextGenerat/README
06AlgoData05VideoGenerat/README
06AlgoData06AudioGenerat/README
06AlgoData07DataEngineer/README
06AlgoData08Special/README
06AlgoData09VectorDB/README
```

<!-- ## 七. 大模型应用 -->

```{toctree}
:maxdepth: 1
:caption: === 七. 大模型应用 ===

07Application/README
07Application00Others/README
07Application01Sample/README
07Application02AIAgent/README
07Application03RAG/README
07Application04AutoDrive/README
07Application05Embodied/README
07Application06Remmcon/README
07Application07Safe/README
07Application08History/README
07Application10NewModel/README
```

<!-- ## 附录内容 -->

```{toctree}
:caption: === 附录内容 ===
:maxdepth: 1

00Others/README
```

## 备注

文字课程开源在 [AI Infra](https://chenzomi12.github.io/)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/chenzomi12/AI Infra)，欢迎引用！

> 非常希望您也参与到这个开源项目中，B 站给 ZOMI 留言哦！
> 
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
>
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！
