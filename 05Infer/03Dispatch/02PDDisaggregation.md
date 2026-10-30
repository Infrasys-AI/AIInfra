<!--Copyright © AIInfra 贡献者 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# PD 分离推理：从原理到 vLLM 部署

预填充（Prefill，P）处理输入提示词，解码（Decode，D）逐步生成后续 token。PD 分离将两个阶段放在不同的推理实例中，通过传输 KV Cache 衔接计算。本节接续[教学模拟](./Code02PDDispatch.md)，讨论真实服务中的传输成本、部署步骤与评测方法。

资料核对日期：2026-10-30。命令以 **vLLM v0.32.0、NVIDIA GPU、NixlConnector、单机两张 GPU** 为范围。PD 分离仍是实验性功能，其他版本、模型架构和硬件组合需重新检查兼容性。本文提供部署和评测步骤，未提供 GPU 实测结果。[<sup>[1]</sup>](#ref1)

## 1. 为什么拆分两个阶段

长提示词的 Prefill 与正在生成回复的 Decode 共享 GPU 时，前者可能拉高后者的尾部延迟。将二者分开，可以分别调整资源规模与并行策略；但每个请求也增加了排队、协调与 KV 传输成本。2026 年 9 月的 vLLM 官方实践强调，应关注满足延迟目标的有效请求吞吐（Goodput），不能假定拆分后总 token 吞吐必然提高。[<sup>[2]</sup>](#ref2)

选择架构前，先建立同等 GPU 数量的共置部署基线。表 1 对比三种优化方式，它们可以组合使用。

<center>表 1 调度与部署优化的作用范围</center>

| 方式 | 主要改变 | 需要验证的成本 |
| :-- | :-- | :-- |
| 连续批处理（Continuous Batching） | 请求完成后及时补入新请求 | 并发增加后的显存占用与排队 |
| 分块预填充（Chunked Prefill） | 将长 Prefill 分成较小调度单元，与 Decode 混合执行 | token 预算对首 token 延迟和生成延迟的影响 |
| PD 分离 | P、D 使用不同实例，通过连接器传输 KV | 传输耗时、两侧负载均衡与额外模型副本 |

vLLM V1 在条件允许时默认启用 Chunked Prefill，优先调度 Decode，再用剩余 token 预算处理 Prefill。`max_num_batched_tokens` 控制每轮预算，较小值通常有利于生成阶段延迟，较大值可能有利于 Prefill；具体取值应由负载实验决定。[<sup>[3]</sup>](#ref3)

## 2. KV 传输需要多少带宽

对于每层 KV 形状一致、使用标准多头注意力或分组查询注意力的模型，单请求提示词 KV 的逻辑数据量可估算为：

$$
S_{KV} = 2 \times L \times T \times H_{KV} \times d_{head} \times b
$$

其中，$L$ 为层数，$T$ 为输入 token 数，$H_{KV}$ 为 KV 头数，$d_{head}$ 为头维度，$b$ 为每个元素的字节数，系数 2 对应 Key 和 Value。这里计算的是整个模型的逻辑 KV 总量；每个并行 rank 的数据量还取决于分片或复制策略。块对齐、元数据与传输协议也会增加实际开销。MLA、混合注意力或状态空间模型需要按各自缓存结构估算。

用一个假设配置手算：$L=32$、$H_{KV}=8$、$d_{head}=128$，KV 使用 BF16，即 $b=2$。每个 token 的 KV 为 128 KiB，8192 个输入 token 对应 1 GiB。

若有效传输带宽为 25 GB/s，单次传输的理想下界约为：

$$
t_{transfer} \geq \frac{1 \times 2^{30}}{25 \times 10^9} \approx 43\text{ ms}
$$

这个结果是容量估算，未计入握手、拷贝、等待与并发争用。也不能把网卡标称 Gb/s 直接当成可用 GB/s。若每个请求都需传输完整提示词 KV、到达率为 $\lambda$，平均传输需求约为 $\lambda S_{KV}$；共享链路上的多个实例需要合并计算。

工程上，PD 分离减少的调度干扰必须足以抵消新增成本。长输入、严格的流式输出延迟目标和较快的互联，是值得开展对比实验的条件；低负载或慢链路上，共置部署可能更合适。[<sup>[2]</sup>](#ref2)

## 3. 用 vLLM 建立最小链路

### 3.1 环境与版本

以下步骤要求 Linux CUDA 环境，两张 GPU 各自能够容纳一份模型与 KV Cache。先在独立 Python 环境中按对应版本的官方安装说明准备 vLLM。模型权重、tokenizer、精度和缓存配置应在 P、D 两侧保持一致。[<sup>[4]</sup>](#ref4)

在独立环境中准备指定版本的运行包、连接器与示例源码。v0.32.0 的连接器依赖文件指定 `nixl==1.5.0`；NIXL 的可用传输后端仍取决于 CUDA、UCX 和系统环境。[<sup>[5]</sup>](#ref5)

```bash
python -m pip install 'vllm==0.32.0' 'nixl==1.5.0'
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git vllm-pd-example
cd vllm-pd-example
vllm --version
nvidia-smi topo -m
```

这里使用小模型 `Qwen/Qwen3-0.6B` 检查链路。小模型的启动成本较低，但不能据此证明 PD 分离能提升生产负载的性能。[<sup>[2]</sup>](#ref2)

### 3.2 启动 P、D 与代理

分别在三个终端执行下面的命令。P 生成 KV，D 消费 KV；同机两个进程使用不同的侧通道端口。`--enforce-eager` 用于与官方最小示例保持一致，正式性能实验应记录并统一执行模式。[<sup>[4]</sup>](#ref4)

```bash
# 终端 1：GPU 0 上的 Prefill 实例
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve Qwen/Qwen3-0.6B --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

```bash
# 终端 2：GPU 1 上的 Decode 实例
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve Qwen/Qwen3-0.6B --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

等待两个实例完成模型加载，再从上述源码目录启动代理。该脚本位于 vLLM 源码的测试目录中，不能假定安装 wheel 后就能直接找到它。[<sup>[6]</sup>](#ref6)

```bash
# 终端 3：请求协调代理
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --host 127.0.0.1 --port 8192 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200
```

验证请求从代理入口完成：

```bash
curl --fail-with-body http://127.0.0.1:8192/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Explain the purpose of KV cache:","max_tokens":32,"temperature":0,"stream":true}'
```

代理先让 P 处理提示词并取得传输元数据，再将请求交给 D。D 依据元数据读取 KV 后继续生成。一次成功响应只能证明基本链路连通，还需结合连接器日志确认实际传输，并运行对应版本的正确性测试。这个代理是开发示例；生产环境还需要处理路由、背压、超时与实例故障。[<sup>[6]</sup>](#ref6)

## 4. 如何公平比较性能

### 4.1 固定资源与工作负载

同机两张 GPU 做 PD 分离时，共置基线也应使用两张 GPU，例如两个独立副本加负载均衡。两组按先后顺序测试，避免同时争抢 GPU。记录模型及权重版本、GPU 型号、互联拓扑、vLLM/NIXL 版本、并行度、精度、最大上下文和调度参数。

预热后分别测短输入、长输入，以及长短混合流量，并逐步增加请求到达率。两组使用相同的输入、输出长度、随机种子与前缀缓存设置。测试完整 KV 传输成本时，两侧可显式关闭前缀缓存；测试业务实际收益时，再按真实复用比例启用。

### 4.2 区分首 token、平均生成时间与尾延迟

首 token 延迟（Time To First Token，TTFT）是客户端发出请求到收到首个输出的时间。单请求平均输出 token 时间（Time Per Output Token，TPOT）在输出 token 数 $N>1$ 时可按下式计算：

$$
TPOT = \frac{t_{last}-t_{first}}{N-1}
$$

逐 token 延迟（Inter-Token Latency，ITL）描述相邻流式输出之间的间隔。平均 TPOT 可能掩盖一次较长停顿，因此应同时报告 ITL 的 P95/P99。流式输出块也未必严格对应一个 token，比较客户端和服务端指标时应确认统计口径。[<sup>[7]</sup>](#ref7)

下面是代理入口的合成负载命令。TTFT 1000 ms、TPOT 50 ms 是演示阈值，需按业务目标修改；它们不是性能承诺。`--goodput` 的阈值单位为毫秒，其 `tpot` 约束是请求级指标，不能代替 P99 ITL 约束。[<sup>[8]</sup>](#ref8)

```bash
vllm bench serve --backend vllm --base-url http://127.0.0.1:8192 \
  --endpoint /v1/completions --model Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 8192 --random-output-len 256 \
  --num-prompts 200 --request-rate 1 --seed 42 --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
  --goodput ttft:1000 tpot:50 --save-result --result-filename pd-result.json
```

对共置基线的负载均衡入口重复相同负载，并单独保存结果。比较不同到达率下的成功请求数、错误率、输出 token 吞吐、TTFT、TPOT 和尾部 ITL。Goodput 的一个明确口径是“同时满足所选延迟阈值的成功请求数 / 测量时间”；报告中需要写清使用了哪些阈值。

如果仅 ITL 改善而 TTFT 超过目标，就不能宣称服务整体达到预期。两侧服务的 `/metrics` 有助于定位问题，但用户体验应以包含代理与网络开销的客户端数据为准。[<sup>[7]</sup>](#ref7)

## 5. 常见问题与排查顺序

1. **先检查拓扑和传输耗时。** GPU 间不能高效传输时，应先定位互联和拷贝开销，再比较调度收益。长提示词的 KV 数据量可用第 2 节估算值交叉核对。
2. **检查版本和模型兼容性。** 固定运行包与示例脚本版本，再核对 NixlConnector 的模型、并行布局和特性交互限制。[<sup>[9]</sup>](#ref9)
3. **检查 UCX 配置。** 使用 UCX 后端时，NCCL 网卡环境变量不能代替 UCX 配置；跨机还需设置可达的侧通道地址并核对网络设备。[<sup>[4]</sup>](#ref4)
4. **检查 KV 生命周期。** D 排队时间、P 的缓存保留时间与实例故障都影响传输能否完成。不要只看到 HTTP 服务存活就认为链路正常。[<sup>[4]</sup>](#ref4)
5. **区分失败与重算。** `kv_load_failure_policy=fail` 在 KV 加载失败时返回错误，`recompute` 允许 D 重算缺失缓存，但会重新引入 Prefill 干扰。评测时必须报告所选策略和传输失败情况。[<sup>[4]</sup>](#ref4)

## 6. 总结与思考

- PD 分离的收益依赖工作负载、传输链路和延迟目标，新增模型副本与 KV 搬运都需要计入资源预算。
- 从同等资源的共置基线开始，先验证正确性，再比较完整客户端链路的 Goodput 与尾延迟。
- 思考：如果输入长度翻倍但输出长度不变，KV 传输需求和 P/D 资源比例会怎样变化？若大量请求共享长前缀，评测方法又应如何调整？

## 参考文献

<div id="ref1"></div>

[1]. vLLM. [Disaggregated Prefilling（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/features/disagg_prefill/).

<div id="ref2"></div>

[2]. Martin Hickey. [Taking vLLM Apart: A Practical Guide to Disaggregated Serving](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html). 2026-09-29.

<div id="ref3"></div>

[3]. vLLM. [Optimization and Tuning（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/configuration/optimization/).

<div id="ref4"></div>

[4]. vLLM. [NixlConnector Usage Guide（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/features/nixl_connector_usage/).

<div id="ref5"></div>

[5]. vLLM. [KV connector requirements（v0.32.0）](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/kv_connectors.txt).

<div id="ref6"></div>

[6]. vLLM. [NIXL integration toy proxy（v0.32.0）](https://github.com/vllm-project/vllm/blob/v0.32.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py).

<div id="ref7"></div>

[7]. vLLM. [Metrics（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/design/metrics/).

<div id="ref8"></div>

[8]. vLLM. [Online serving benchmark CLI（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/cli/bench/serve/).

<div id="ref9"></div>

[9]. vLLM. [NixlConnector Compatibility Matrix（v0.32.0）](https://docs.vllm.ai/en/v0.32.0/features/nixl_connector_compatibility/).
