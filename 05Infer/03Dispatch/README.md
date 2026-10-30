<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 架构调度加速

本章围绕请求批处理、调度顺序与 Prefill/Decode 分离，讨论如何在资源约束下改善大模型服务的吞吐与延迟。

## 课程位置

[大模型推理](../README.md) → 架构调度加速

## 课程简介

- [连续批处理原理](./01ContinuousBatching.md)：理解静态批处理与连续批处理的区别。
- [PD 分离推理：从原理到 vLLM 部署](./02PDDisaggregation.md)：以 vLLM v0.32.0 为例，分析 KV 传输成本、部署流程与延迟评测。
- [先到先服务调度](./05FCFSScheduling.md)：理解 FCFS 的执行顺序与调度取舍。

## 课程知识

| 类型 | 内容 | 链接 |
| :-- | :-- | :-- |
| 原理 | 连续批处理 | [Markdown](./01ContinuousBatching.md) |
| 实践 | 连续批处理教学实现 | [Markdown](./Code01ContinousBatching.md) |
| 原理与部署 | PD 分离与 vLLM 工程实践 | [Markdown](./02PDDisaggregation.md) |
| 教学模拟 | PD 分离简化模拟 | [Markdown](./Code02PDDispatch.md) |
| 原理 | FCFS 调度 | [Markdown](./05FCFSScheduling.md) |

## 备注

系列视频托管[B 站](https://space.bilibili.com/517221395)和[油管](https://www.youtube.com/@ZOMI666/playlists)，PPT 开源在[github](https://github.com/Infrasys-AI/AIInfra)，欢迎引用！

> 非常希望您参与到这个开源课程中，B 站给 ZOMI 留言哦！
>
