<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# Transformer 架构

本系列视频将系统性地讲解 Transformer 架构的核心原理及其在 LLM 中的关键技术实现。从 Transformer 的基础结构入手，逐步深入 Tokenizer 分词算法、Embedding 向量化技术、Attention 注意力机制及其多种变种算法，并专门探讨 Transformer 在长序列处理上的架构优化方案，最后详解大模型训练与推理中的关键参数设置。为深入理解与应用前沿 AI 技术奠定坚实基础。

## 内容大纲

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---- |
| Transformer 架构 | 01 Transformer 基础结构 | [PPT](./01Transformer.pdf), [视频](https://www.bilibili.com/video/BV1rt421476q/), [文章](./01Transformer.md) | :white_check_mark: |
| Transformer 架构 | 02 大模型 Tokenizer 算法 | [PPT](./02Tokenizer.pdf), [视频](https://www.bilibili.com/video/BV16pTJz9EV4), [文章](./02Tokenizer.md) | :white_check_mark: |
| Transformer 架构 | 03 大模型 Embedding 算法 | [PPT](./03Embeding.pdf), [视频](https://www.bilibili.com/video/BV1SSTgzLEzf), [文章](./03Embeding.md) | :white_check_mark: |
| Transformer 架构 | 04 Attention 注意力机制 | [PPT](./04Attention.pdf), [视频](https://www.bilibili.com/video/BV11AMHzuEet), [文章](./04Attention.md) | :white_check_mark: |
| Transformer 架构 | 05 Attention 变种算法 | [PPT](./05GQAMLA.pdf), [视频](https://www.bilibili.com/video/BV1GzMUz8Eav), [文章](./05GQAMLA.md) | :white_check_mark: |
| Transformer 架构 | 06 Transformer 长序列架构 | [PPT](./06LongSeq.pdf), [视频](https://www.bilibili.com/video/BV16PN6z6ELg), [文章](./06LongSeq.md) | :white_check_mark: |
| Transformer 架构 | 07 大模型参数设置 | [PPT](./07Parameter.pdf), [视频](https://www.bilibili.com/video/BV1nTNkzjE3J), [文章](./07Parameter.md) | :white_check_mark: |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 代码实践 | 01 搭建迷你 Transformer | [Markdown](./Practice01MiniTranformer.md), [Jupyter](./Practice01MiniTranformer.ipynb) | :white_check_mark: |
| 代码实践 | 02 从零实现 Transformer 训练 | [Markdown](./Practice02TransformerTrain.md), [Jupyter](./Practice02TransformerTrain.ipynb) | :white_check_mark: |
| 代码实践 | 03 实战 Transformer 机器翻译 | [Markdown](./Practice03MachineTrans.md), [Jupyter](./Practice03MachineTrans.ipynb) | :white_check_mark: |
| 代码实践 | 04 手把手实现核心机制 Sinusoidal 编码 | [Markdown](./Practice04Sinusoidal.md), [Jupyter](./Practice04Sinusoidal.ipynb) | :white_check_mark: |
| 代码实践 | 05 手把手实现核心机制 BPE 分词算法 | [Markdown](./Practice05BPE.md), [Jupyter](./Practice05BPE.ipynb) | :white_check_mark: |
| 代码实践 | 06 手把手实现核心机制 Embedding 词嵌入 | [Markdown](./Practice06Embedding.md), [Jupyter](./Practice06Embedding.ipynb) | :white_check_mark: |
| 代码实践 | 07 深入注意力机制 MHA、MQA、GQA、MLA | [Markdown](./Practice07Attention.md), [Jupyter](./Practice07Attention.ipynb) | :white_check_mark: |

## 内容大纲

![](./images/00outline.png)

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
