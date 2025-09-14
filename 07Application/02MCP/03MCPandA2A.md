<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# MCP 与 A2A 原理

## MCP 原理

2023 年 OpenAI 首次在大模型引入 [Function Call](https://platform.openai.com/docs/guides/function-calling) 的概念，通过结构化的输出，使得大模型不再局限于做一个聊天的工具，而是可以真实的调用外部的 API，让大模型具备了自主和外部系统交互、使用外部工具的能力，大大拓展了大模型能力的边界。

2024 年底，Anthropic 推出了 开源协议 Model Context Protocol（[MCP](https://modelcontextprotocol.io/introduction)）也即模型上下文协议，目标是统一应用程序和 AI 模型之间交换上下文信息的方式。使开发者能够以一致的方式将各种数据源、工具和功能连接到 AI，它是一个中间协议层，就像 USB-C 让不同设备能够通过相同的接口连接一样。

从提出以来，该协议得到了开源社区和商业组织的广泛关注和积极响应。并基于此协议提供了大量的 [MCP Server](https://github.com/modelcontextprotocol/servers)。某种程度上，MCP 已经成为了 AI 工具调用的行业标准。


### MCP 基本介绍

MCP 定义了应用程序和 AI 模型之间交换上下文信息的方式。统一了大模型与外部资源。比如数据源、工具之间的通信协议。为大模型提供连接万物的接口。目标是创建一个通用标准，使 AI 应用程序的开发和集成变得更加简单和统一。

### 从 Function Call 到 MCP，为什么需要 MCP

首先简单介绍一下 Function Call（函数调用），Function Call 在 2023 年被引入，提供了大模型与外部系统交互的能力，当大模型遇到自己无法直接回答的问题时，它会主动调用预设的函数（如查询天气、计算数据、访问数据库等），获取实时或精准信息后再生成回答。


大模型使用 Function Call 调用工具的架构如下：

![](images/03MCPandA2A_01.png)

Function Call 已经可以让大模型去调用工具，那为什么我们还需要 MCP 呢，因为对于不同的大模型 Function Call 对应的要发开一个 Function Call 工具，需要对不同的模型进行适配，比如参数格式、触发逻辑、返回结构等等，这个成本是非常高的。我们需要对每个模型都开发一套适配层。

以 vLLM 推理框架为例，从这个 [commit](https://github.com/vllm-project/vllm/pull/18874) 可以看到，为了适配 DeepSeek 的 Function Call 能力需要指定 tool call 的解析器，以及对应的 chat_template。这种垂直调用方式，大大提高了 AI Agent 的开发门槛。因此我们引入了 MCP 协议。

MCP 制定统一规范，不管是连接数据库、第三方 API，还是本地文件等各种外部资源，都可以通过这个 “通用接口” 来完成，让 AI 模型与外部工具或数据源之间的交互更加标准化、可复用。

大模型使用 MCP 调用工具的架构如下：

![](images/03MCPandA2A_02.png)

有了 MCP 之后

- 开发者按照 MCP 协议进行开发，无需为每个模型与不同资源的对接重复编写适配代码，可以大大节省开发工作量。

- 已经开发出的 MCP Server，因为协议是通用的，能够直接开放出来给大家使用，也大幅减少了开发者的重复劳动。

以下是 MCP 与 Function Call 的对比总结：

||MCP|Function Call|
|--|---|-------------|
|定义|模型和其它设备集成的标准接口，Tool 的函数约定了输入输出的协议规范。|将模型连接到外部数据和系统，平铺式的罗列 Tools 工具。|
|协议|JSON-RPC，支持双向通信、可发现性|JSON-Schema，静态函数调用。|
|调用方式|STDIO / SSE / 同进程调用|同进程调用 / 编程语言对应的函数|
|适用场景|更适合动态、复杂的交互场景|单一特定工具、静态函数执行调用|
|集成难度|高|简单|
|工程程度|高|低|

### MCP Demo 演示

以下演示在 Cherry Studio 客户端中进行，[下载地址](https://www.cherry-ai.com/download)

![](images/03MCPandA2A_04.png)

点击右上角右上角设置，在设置页面中可以在模型服务菜单中添加模型，在 MCP 菜单中添加 MCP 服务。

## MCP 核心原理

![](images/03MCPandA2A_03.png)

上图是类比于 USB-C 来看 MCP，核心设计理念上是要成为 AI 的 USB-C，架构上的核心组件则是 MCP server 、 MCP client 、 MCP host。下面将进行详细的介绍。

### MCP 设计理念

### MCP 架构

### 一次 MCP 的完整调用之旅

## 总结与思考

### MCP 的局限性

### MCP 未来展望

## A2A 原理
