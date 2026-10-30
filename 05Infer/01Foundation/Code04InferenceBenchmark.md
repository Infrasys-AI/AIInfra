<!--Copyright © AIInfra 贡献者 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 实践：推理性能与成本评测

本实验连接[推理性能指标](./03Performance.md)与真实数据：先生成无需 GPU 的交互报告，再下载 InferenceX 的公开观测点，最后对自己的 vLLM/SGLang 服务扫描并发。三个案例使用同一套输出格式，可以查看交互速度与吞吐的关系，并输入小时成本做敏感性分析。

**环境**：Python 3.10 或更新版本。下载、分析、报告和测试只使用 Python 标准库；自有服务压测另需安装 vLLM v0.32.0 客户端和部署好的兼容服务。[示例代码](./code/inference_benchmark/demo.py)、[报告模板](./code/inference_benchmark/report.html)、[测试](./code/inference_benchmark/test_demo.py)均在仓库内。

## 1. 评测口径

[InferenceX DeepSeek V4.1 Flash 计算器](https://inferencex.semianalysis.com/calculator/deepseek-v41-flash)根据实测点估算不同交互速度目标下的吞吐与成本。本实验采用**满足目标的观测点筛选**：在同一个配置系列中，选择满足交互速度下限且吞吐最高的实际观测点。没有合格点时显示空值，不做插值或外推。

本文资料核对日期为 2026-10-30。该日此模型的公开原始 API 返回 Agentic Traces 多轮负载，固定 8192/1024 的计算器投影返回空数组。因此，下面的公开数据案例明确选择 `agentic_traces`，自有服务案例明确使用固定输入/输出长度。运行时会记录实际可用数据，不用其他负载替代缺失结果。

表 1 给出两个数据入口的共同口径。实现根据 [InferenceX 原始 API](https://inferencex.semianalysis.com/api)与 [vLLM v0.32.0 压测客户端](https://docs.vllm.ai/en/v0.32.0/cli/bench/serve/)的字段进行单位转换。

<center>表 1 数据与成本口径</center>

| 项目 | 本实验的定义 |
| :-- | :-- |
| 交互速度 | `1000 / p90_itl_ms`，单位为输出 tok/s；这是 P90 ITL 的倒数，不能解释成所有用户的保底速度 |
| 吞吐 | 成功生成的输出 token/s/GPU；也可以切换为输入加输出 token/s/GPU |
| GPU 数量 | 共置部署使用单份物理 GPU 数，镜像记录的 P/D 数量不相加；分离部署将 P、D 两侧 GPU 相加 |
| 公开数据时间单位 | API 的 ITL/TTFT 单位为秒，转换为毫秒后分析 |
| 本地数据时间单位 | vLLM 导出的 `*_ms` 字段直接使用 |
| 小时成本 | 用户输入的 USD/GPU-hour，缺失时成本显示未知 |
| 来源与系列 | 按来源、模型、负载、硬件、框架、精度、推测解码、卸载模式、PD 模式、日期与软件镜像分别分析 |

设每 GPU 小时成本为 $h$，每 GPU 吞吐为 $q$，计费时间内有效负载比例为 $u$，则本实验的每百万 token 成本为：

$$
C = \frac{h \times 10^6}{3600 \times q \times u}
$$

这里的小时成本和有效负载比例都是用户假设；只有当费率已包含折旧、电力、网络等对应范围的成本时，才能据此讨论相应范围的拥有成本。该公式不计算 API 售价或利润。Agentic 负载中的长前缀复用也会改变输入 token 与输出 token 的比例，切换为总 token 时应保留这一背景。

## 2. 案例一：离线体验报告

从仓库根目录进入示例目录，执行：

```bash
cd 05Infer/01Foundation/code/inference_benchmark
python3 demo.py sample --output-dir /tmp/aiinfra-sample
```

打开 `/tmp/aiinfra-sample/report.html`。报告包含八个明确标记为 **SYNTHETIC** 的模拟点，用来练习调整交互速度下限、切换 token 口径，以及输入小时成本。它们不代表任何真实 GPU 或模型的性能。

每次运行使用一个新的空输出目录。报告是自包含 HTML，不需要服务器或外部 JavaScript 库，断网也可以查看。

## 3. 案例二：分析 DeepSeek V4.1 Flash 公开数据

在示例目录执行：

```bash
python3 demo.py fetch --model DeepSeek-V4.1-Flash \
  --date 2026-10-30 --workload agentic_traces \
  --target 50 --output-dir /tmp/aiinfra-deepseek-public
```

`--date` 使用 API 的截至日期查询，返回各配置在该日或此前的最新观测，**不表示所有点在同一天测量**。报告保留每个点的日期与原始运行链接；原始响应另存为 `raw.json`，下载 URL、下载时间与 SHA-256 保存在 `metadata.json` 中。缺少必要吞吐或 GPU 数量字段的行会被跳过，原因和数量写入元数据。

报告按配置系列展示观测点，悬停可查看并发、日期和运行标识。高于交互速度下限的点才参与吞吐选择。如果想分析其他模型或负载，可以修改 `--model` 和 `--workload`；请求不存在的负载会明确报错并保留原始数据。

原始公开数据中的错误计数可能未知，本实验不会据此宣称其错误率为零。下载和转换公开观测属于数据分析，没有在本机重新运行该模型，也不会变成 InferenceX 官方提交。

### 3.1 输入成本并重算

创建一个费率文件。下面使用虚构名称演示 JSON 格式；分析真实数据时，应替换成报告中的硬件键和自己的费率来源。

```json
{
  "SYNTHETIC-A": 2.0,
  "SYNTHETIC-B": 3.0
}
```

将文件保存为 `prices.json`，对案例一的固定快照重新分析：

```bash
python3 demo.py report --input /tmp/aiinfra-sample/points.json \
  --prices prices.json --utilization 0.7 --target 50 \
  --output-dir /tmp/aiinfra-sample-repriced
```

同样可以把 `--input` 换成公开数据或本地压测的 `points.json`。保存的快照不会重新联网下载。JSON/CSV 中记录本次使用的费率与结果；在 HTML 中手动修改参数只影响当前页面显示，不会回写磁盘文件。

## 4. 案例三：压测自己的 vLLM/SGLang 服务

### 4.1 建立服务与准备客户端

以 Linux CUDA 环境中一张显存足够的 GPU 和小模型为例。先按官方说明安装 [vLLM](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)，然后在服务端执行：

```bash
vllm serve Qwen/Qwen3-0.6B --port 8000 --dtype bfloat16 \
  --max-model-len 16384 --no-enable-prefix-caching
```

等待模型加载完成。压测客户端可运行在另一台机器，需准备 `vllm==0.32.0` 和对应 tokenizer 的访问条件。小模型用于验证流程，获得的性能结论仅适用于实际记录的模型与部署。

已有 SGLang 服务也可以使用相同客户端访问其 `/v1/completions` 兼容入口；设置 `--framework sglang`，并记录真实服务端的镜像或 commit、并行方式和缓存设置。服务端支持的额外请求字段及输出长度控制应在正式扫描前验证。

### 4.2 先查看命令，再执行扫描

先生成执行计划，不启动客户端，也不向服务发送请求：

```bash
python3 demo.py bench --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B --hardware my-gpu --num-gpus 1 \
  --framework vllm --precision bf16 \
  --server-note 'vllm=0.32.0; TP=1; prefix-cache=off; max-len=16384' \
  --concurrency 1,4,8 --num-prompts 32 --isl 1024 --osl 128 \
  --output-dir /tmp/aiinfra-local-plan --dry-run
```

确认服务配置和请求量后，使用一个新目录执行实测：

```bash
python3 demo.py bench --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B --hardware my-gpu --num-gpus 1 \
  --framework vllm --precision bf16 \
  --server-note 'vllm=0.32.0; TP=1; prefix-cache=off; max-len=16384' \
  --concurrency 1,4,8 --num-prompts 32 --isl 1024 --osl 128 \
  --output-dir /tmp/aiinfra-local-measured
```

示例对每个并发档位先预热，再运行 32 个测量请求。客户端还会进行连接测试，因此服务实际收到的请求数会多于测量请求数。认证服务通过客户端支持的 `OPENAI_API_KEY` 环境变量传递凭据；脚本不将其加入命令或元数据。

`--request-rate inf` 与 `--max-concurrency` 组合用于并发饱和扫描。客户端等待并发名额的时间不计入报告采用的 ITL/TTFT；这不是固定请求到达率下的排队延迟实验。正式容量规划需要增加样本、重复运行，并补充实际请求到达分布与长短输入混合负载。

### 4.3 扩展到 8K/1K 与 PD 分离

固定 8K/1K 实验可设置 `--isl 8192 --osl 1024`，并确认服务端最大上下文足够。该固定单轮负载与公开 Agentic Traces 不能作为相同工作负载直接排序。

PD 分离时，`--base-url` 指向请求代理，添加 `--disagg`，`--num-gpus` 填写 P、D 两侧全部 GPU 的物理数量；框架、精度、推测解码方式和镜像应据实填写。一个服务被多个客户端并行压测时，也应记录共享资源情况。

本地扫描由客户端提供请求级 Goodput，默认使用 TTFT 1000 ms、TPOT 50 ms 的示例阈值，分别由 `--ttft-slo-ms`、`--tpot-slo-ms` 设置。报告中按 P90 ITL 的配置点筛选与请求级 Goodput 是不同指标，不应相互代替。

## 5. 输出与验证

表 2 列出输出文件。客户端失败、请求计数不一致或出现失败请求时，扫描会以非零状态退出，保留原始结果和日志，并将元数据标记为 `failed`，不会生成成功报告。

<center>表 2 实验产物</center>

| 文件 | 内容 |
| :-- | :-- |
| `metadata.json` | 来源、校验哈希或客户端执行计划、服务配置与执行状态 |
| `raw.json` | 公开 API 原始响应 |
| `raw-conc-*.json`、`client-conc-*.log` | 每个本地并发档位的原始结果与客户端日志 |
| `points.json` | 来源、指标与单位统一后的观测点快照，以及成本假设 |
| `summary.json`、`summary.csv` | 每个系列满足目标的观测点与成本计算结果 |
| `report.html` | 可离线查看的交互报告 |

在示例目录运行 CPU 测试：

```bash
python3 -m unittest -v
```

测试覆盖物理 GPU 计数、秒/毫秒转换、token 成本分母、缺失指标、不可达目标、不同负载隔离，以及模拟客户端的成功/失败处理。模拟客户端只验证执行与文件处理流程，不能替代真实 GPU 测量。

## 6. 思考

- 为什么并发增加时总吞吐可能提高，而单用户交互速度下降？
- 同样的输入/输出长度下，前缀缓存、推测解码和精度分别会改变哪些指标？
- 每 GPU 小时费率更低的配置，是否一定有更低的每百万输出 token 成本？

## 参考资料

- [InferenceX 原始数据 API 与字段说明](https://inferencex.semianalysis.com/api)
- [InferenceX 官方基准项目](https://github.com/SemiAnalysisAI/InferenceX)
- [InferenceX 计算器的方法说明](https://github.com/SemiAnalysisAI/InferenceX-app/blob/master/docs/tco-calculator.md)
- [vLLM v0.32.0 在线服务压测命令](https://docs.vllm.ai/en/v0.32.0/cli/bench/serve/)
