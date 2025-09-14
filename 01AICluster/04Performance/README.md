<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 04.集群性能分析

本系列将系统讲解​​集群性能的全方位分析与优化实践​​，涵盖以下核心内容：从硬件资源、模型效率、利用率到业务指标的集群性能评估；基于 TOP500 和 MLPerf 的算力计算方法；网络通信带宽与线性度的量化分析；大模型训练/推理的内存需求测算；MFU（模型浮点运算利用率）的评估；以及能效与成本扩展计算，构建从基础指标到成本控制的完整性能优化体系。

## 内容大纲

> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 | 状态 |
|:--- |:---- |:-------------------- |:---- |
|:sparkling_heart:|:star2:|:sparkling_heart:| |
| 性能 实践 :computer: | CODE 01: 拆解 Transformer-Decoder | [Markdown](./CODE01Modeling.md), [Jupyter](./CODE01Modeling.md), [文章](https://infrasys-ai.github.io/aiinfra-docs/01AICluster04Performance/CODE01Modeling.html) | :white_check_mark: |
| 性能 实践 :computer: | CODE 02: MOE 参数量和计算量 | [Markdown](./CODE02MOE.md), [Jupyter](./CODE02MOE.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/01AICluster04Performance/CODE02MOE.html) | :white_check_mark: |
| 性能 实践 :computer: | CODE 03: MFU 模型利用率评估 | [Markdown](./CODE03MFU.md), [Jupyter](./CODE03MFU.ipynb), [文章](https://infrasys-ai.github.io/aiinfra-docs/01AICluster04Performance/CODE03MFU.html) | :white_check_mark: |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
