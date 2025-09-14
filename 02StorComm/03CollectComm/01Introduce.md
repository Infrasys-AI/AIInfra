<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 01.大模型集合通信

Author by: SingularityKChen

在 AI 系统中，**计算算子**和**通信算子**是两类核心操作。计算算子映射到 AI 芯片上实现。而通信算子的实现是**硬件互连技术**、**软件通信库**和**系统优化策略**共同作用的结果。当单个 AI 芯片无法满足大模型的时候，通信算子的性能直接决定了分布式 AI 训练的扩展效率，是构建大规模 AI 系统的核心环节。

本系列课程将从**通信算法**、**通信链路**、**通信原语**和**通信域管理**四个维度展开讲解，从而帮助读者理解集合通信如何影响 AI 基础设施的软硬件系统、AI 训练算法甚至 AI 模型的架构。

通信是指信息从一个地方传递到另一个地方的过程。它包括信息的发送、传输和接收等环节。当单一节点的算力或存储无法满足应用需求时，往往采用分布式内存架构的节点协作解决给定问题。此时原本在单个节点上执行的计算现在自然地分布在各个节点之间。

当数据需要共享和（或）不同节点的计算结果需要合并时，就会进行通信。同时涉及一组节点的通信操作称为集合通信操作。

!!!!!!!!!!!
这里是综述，所以可以引用一些大模型集合通信的相关的综述性的介绍，特别是比较基础的原理知识就可以了

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?aid=1355442092&bvid=BV1jz421h7CA&cid=1568779156&page=1&as_wide=1&high_quality=1&danmaku=0&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>

## 引用

- [NVIDIA DGX-2](https://www.nvidia.cn/data-center/dgx-2/)
