<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# MoE 混合专家

MoE（Mixture of Experts）架构，即专家混合架构，是一种通过多个专家模块并行处理不同子任务，由门控网络依据输入数据动态分配，决定各专家模块参与度，以实现更高效、灵活处理复杂任务，提升模型表现与泛化能力的技术。

## 内容大纲

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---- |
| MOE 基本介绍 | 01 MOE 架构剖析  | [PPT](./01MOEIntroducion.pdf), [视频](https://www.bilibili.com/video/BV17PNtekE3Y/) | :white_check_mark: |
| MOE 前世今生 | 02 MOE 前世今生  | [PPT](./02MOEHistory.pdf), [视频](https://www.bilibili.com/video/BV1y7wZeeE96/) | :white_check_mark: |
| MOE 核心论文 | 03 MOE 奠基论文  | [PPT](./03MOECreate.pdf), [视频](https://www.bilibili.com/video/BV1MiAYeuETj/) | :white_check_mark: |
| MOE 核心论文 | 04 MOE 初遇 RNN  | [PPT](./04MOERNN.pdf), [视频](https://www.bilibili.com/video/BV1RYAjeKE3o/) | :white_check_mark: |
| MOE 核心论文 | 05 GSard 解读  | [PPT](./05MOEGshard.pdf), [视频](https://www.bilibili.com/video/BV1r8ApeaEyW/) | :white_check_mark: |
| MOE 核心论文 | 06 Switch Trans 解读  | [PPT](./06MOESwitch.pdf), [视频](https://www.bilibili.com/video/BV1UsPceJEEQ/) | :white_check_mark: |
| MOE 核心论文 | 07 GLaM & ST-MOE 解读  | [PPT](./07MOEGLaM_STMOE.pdf), [视频](https://www.bilibili.com/video/BV1L59qYqEVw/) | :white_check_mark: |
| MOE 核心论文 | 08 DeepSeek MOE 解读  | [PPT](./08DeepSeekMoE.pdf), [视频](https://www.bilibili.com/video/BV1tE9HYUEdz/) | :white_check_mark: |
| MOE 架构原理 | 09 MOE 模型可视化  | [PPT](./09MoECore.pdf), [视频](https://www.bilibili.com/video/BV1Gj9ZYdE4N/) | :white_check_mark: |
| 大模型遇 MOE | 10 MoE 参数与专家  | [PPT](./10MOELLM.pdf), [视频](https://www.bilibili.com/video/BV1UERNYqEwU/) | :white_check_mark: |
| 手撕 MOE 代码 | 11 单机单卡 MoE  | [PPT](./11MOECode.pdf), [视频](https://www.bilibili.com/video/BV1UTRYYUE5o) | :white_check_mark: |
| 手撕 MOE 代码 | 12 单机多卡 MoE  | [PPT](./11MOECode.pdf), [视频](https://www.bilibili.com/video/BV1JaR5YSEMN) | :white_check_mark: |
| 视觉 MoE | 13 视觉 MoE 模型  | [PPT](./12MOEFuture.pdf), [视频](https://www.bilibili.com/video/BV1JNQVYBEq7) | :white_check_mark: |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| MOE 实践 :computer: | 01 基于 Huggingface 实现 MOE 推理任务 | [Markdown](./Practice01MOEInfer.md), [Jupyter](./Practice01MOEInfer.ipynb) | :white_check_mark: |
| MOE 实践 :computer: | 02 从零开始手撕 MoE | [Markdown](./Practice02SignalMOE.md), [Jupyter](./Practice02SignalMOE.ipynb) | :white_check_mark: |
| MOE 实践 :computer: | 03 MoE 从原理到分布式实现 | [Markdown](./Practice03IntrtaMOE.md), [Jupyter](./Practice03IntrtaMOE.ipynb) | :white_check_mark: |
| MOE 实践 :computer: | 04 MoE 分布式性能分析 | [Markdown](./Practice04MOEAnalysize.md), [Jupyter](./Practice04MOEAnalysize.ipynb) | :white_check_mark: |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
