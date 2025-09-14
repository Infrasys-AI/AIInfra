<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 集合通信库

MPI 是集合通信库的鼻祖，英伟达 NVIDIA 大量的参考和借鉴 MPI 通信库相关的内容从而提出了业界集合通信库的标杆 NCCL。本将会从 MPI 开始，介绍业界的各种主流集合通信库的变种 XCCL。然后深入地剖析 NCCL 相关的实现算法、对外 API 等，最后还会介绍华为开源的 HCCL 集合通信库。

## 内容大纲

> `PPT`和`字幕`需要到 [Github](https://github.com/Infrasys-AI/AIInfra) 下载，网页课程版链接会失效哦~
>
> 建议优先下载 PDF 版本，PPT 版本会因为字体缺失等原因导致版本很丑哦~

| 大纲 | 小节 | 链接 |
|:--:|:--:|:--:|
| 集合通信库 | 01 通信库基础 MPI 介绍 | [PPT](./01MPIIntro.pdf), [视频]() |
| 集合通信库 | 02 02 业界 XCCL 大串烧(上) | [PPT](./02XCCL.pdf), [视频]() |
| 集合通信库 | 03 02 业界 XCCL 大串烧(下) | [PPT](./03XCCL.pdf), [视频]() |
| 集合通信库 | 04 英伟达 NCCL 原理剖析 | [PPT](./04NCCLIntro.pdf), [视频]() |
| 集合通信库 | 05 英伟达 NCCL API 介绍 | [PPT](./05NCCLAPI.pdf), [视频]() |
| 集合通信库 | 06 NCCL 通信算法与拓扑关系 | [PPT](./06NCCLPXN.pdf), [视频]() |
| 集合通信库 | 07 NCCL 双二叉树算法原理 | [PPT](./07DBTree.pdf), [视频]() |
| 集合通信库 | 08 华为 HCCL 架构介绍 | [PPT](./08HCCLIntro.pdf), [视频]() |
| 集合通信库 | 09 华为 HCCL 拓扑算法 | [PPT](./09HCCLOpt.pdf), [视频]() |
| 集合通信库 | 10 通信模型&通信影响 | [PPT](./10Summary.pdf), [视频]() |

## 备注

文字课程内容正在一节节补充更新，每晚会抽空继续更新正在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs) ，希望您多多鼓励和参与进来！！！

文字课程开源在 [AI Infra](https://infrasys-ai.github.io/aiinfra-docs)，系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
> 欢迎发现 bug 或者勘误直接提交代码 PR 到社区哦！
