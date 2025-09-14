<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 01: KV Cache 缓存优化

本文将围绕 Transformer 模型中的 KVCache 技术展开，通过实验对比关闭 KVCache、开启 KVCache 和使用 PagedAttention 三种场景下的性能表现。

我们会重点关注**显存占用**和**推理延迟**这两个关键指标，并使用 Python 代码进行实际测量和分析。

## 1. 实验环境设置

首先设置实验环境，确保结果的可重现性。我们使用 Hugging Face 的 Transformers 库来加载一个适中的模型，以便在消费级 GPU 上运行实验。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
```

接下来加载一个适中的模型进行实验。我们选择 GPT-2 模型，它在保持 Transformer 架构完整性的同时，计算需求相对较小。

```python
# 加载模型和分词器
model_name = "gpt2"  # 使用较小的 GPT-2 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 添加填充令牌（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"模型 {model_name} 加载完成")
```

## 2. KVCache 技术原理

在深入代码之前，理解 KVCache 的技术原理至关重要。在 Transformer 的自注意力机制中，每个输入序列都需要计算键(Key)和值(Value)向量。对于生成任务，当我们逐步生成 token 时，重复计算先前所有 token 的 KV 值会导致大量冗余计算。

KVCache 的核心思想是将先前计算过的 KV 值存储起来，避免在生成新 token 时重复计算。数学上，自注意力机制可以表示为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$, $K$, $V$ 分别表示查询(Query)、键(Key)和值(Value)矩阵。在生成过程中，只有最新 token 的 $Q$ 需要与所有先前 token 的 $K$ 和 $V$ 进行计算。

大语言模型生成文本的核心原理是基于深度学习技术，通过训练大规模语料库来学习语言规律，并生成具有相似统计特征的新文本。这些模型的核心是建立一个统计模型，用来估计文本序列中每个词语或字符出现的概率。

## 3. 关闭 KVCache

在第一个实验中，我们完全关闭 KVCache 功能，每次生成新 token 时都重新计算所有先前 token 的 KV 值。这种方法计算效率最低，但可以帮助我们理解 KVCache 的价值。

```python
def generate_without_kv_cache(model, input_ids, max_length=50):
    """
    不使用 KVCache 的生成函数
    每次生成新 token 时都重新计算所有先前 token 的 KV 值
    """
    generated = input_ids
    past_key_values = None  # 明确不使用缓存
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                generated, 
                past_key_values=past_key_values,
                use_cache=False  # 强制不使用缓存
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 始终不使用缓存，所以 past_key_values 保持为 None
        
    return generated

# 准备输入
input_text = "深度学习中的注意力机制是"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 测量显存和延迟
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start_time = time.time()
output_ids = generate_without_kv_cache(model, input_ids)
end_time = time.time()

memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为 MB
latency = end_time - start_time

print(f"关闭 KVCache - 生成文本: {tokenizer.decode(output_ids[0])}")
print(f"关闭 KVCache - 显存占用: {memory_used:.2f} MB")
print(f"关闭 KVCache - 推理延迟: {latency:.4f} 秒")
```

这个实验展示了最基础的生成方式，每次都需要重新计算整个序列的注意力，计算复杂度为 $O(n^2)$，其中 n 是序列长度。大语言模型通过概率方法生成文本，即根据输入或上下文为每个可能的词或句子分配一个概率，然后选择概率最高的词或句子，或者从概率分布中采样，来生成输出文本。

## 4. 开启 KVCache

现在，我们启用 KVCache 功能。这将显著减少计算量，因为只需要计算最新 token 的注意力权重。

```python
def generate_with_kv_cache(model, input_ids, max_length=50):
    """
    使用 KVCache 的生成函数
    缓存先前计算的 KV 值以避免重复计算
    """
    generated = input_ids
    past_key_values = None  # 初始化为 None
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                generated, 
                past_key_values=past_key_values,
                use_cache=True  # 启用缓存
            )
            
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # 更新 KVCache 以供下一次迭代使用
        past_key_values = outputs.past_key_values
        
    return generated

# 测量显存和延迟
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start_time = time.time()
output_ids = generate_with_kv_cache(model, input_ids)
end_time = time.time()

memory_used_kv = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为 MB
latency_kv = end_time - start_time

print(f"开启 KVCache - 生成文本: {tokenizer.decode(output_ids[0])}")
print(f"开启 KVCache - 显存占用: {memory_used_kv:.2f} MB")
print(f"开启 KVCache - 推理延迟: {latency_kv:.4f} 秒")
```

使用 KVCache 后，计算复杂度降低到 $O(n)$，因为只需要计算最新 token 与所有缓存 key 的点积。但是，KVCache 可能占用大量显存，尤其是对于长序列。大语言模型具有上下文感知能力，可以根据上下文信息进行文本生成和理解，从而更好地适应不同的语言环境。

## 5. KVCache 内存挑战

虽然 KVCache 显著提高了计算效率，但它也带来了内存挑战。对于生成长序列，KVCache 可能占用大量显存。具体来说，缓存大小与序列长度、批处理大小、注意力头数和头维度成正比：

$$\text{缓存大小} = 2 \times b \times h \times l \times d$$

其中 $b$ 是批处理大小，$h$ 是注意力头数，$l$ 是序列长度，$d$ 是每个头的维度。

传统 KVCache 需要连续的内存空间，当生成长序列时可能找不到足够大的连续内存块，导致内存碎片化。大语言模型通常是巨型模型，包含数以亿计的参数，以便处理大量的语言数据，这使得内存管理变得尤为重要。

## 6. PagedAttention

PagedAttention 是一种高级优化技术，灵感来自操作系统中的虚拟内存和分页概念。它将 KVCache 分成固定大小的块（页），并在非连续的内存空间中管理这些页。

由于直接实现 PagedAttention 需要复杂的底层优化，我们使用 vLLM 库来实现这一功能。vLLM 是一个高效推理引擎，内置了对 PagedAttention 的支持。

```python
# 安装 vLLM 库
# !pip install vllm

from vllm import LLM, SamplingParams

# 使用 vLLM 的 PagedAttention 实现
def generate_with_paged_attention(input_text, max_length=50):
    """
    使用 vLLM 的 PagedAttention 进行生成
    """
    # 初始化 vLLM 模型
    llm = LLM(model=model_name)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_length,
        ignore_eos=False  # 不忽略结束符
    )
    
    # 测量显存和延迟
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    # 使用 vLLM 生成文本
    outputs = llm.generate([input_text], sampling_params)
    end_time = time.time()
    
    memory_used_paged = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为 MB
    latency_paged = end_time - start_time
    
    generated_text = outputs[0].outputs[0].text
    
    print(f"Paged Attention - 生成文本: {generated_text}")
    print(f"Paged Attention - 显存占用: {memory_used_paged:.2f} MB")
    print(f"Paged Attention - 推理延迟: {latency_paged:.4f} 秒")
    
    return generated_text, memory_used_paged, latency_paged

# 运行 PagedAttention 实验
paged_text, memory_used_paged, latency_paged = generate_with_paged_attention(input_text)
```

PagedAttention 通过分页机制解决了 KVCache 的内存碎片问题。它将 KVCache 分成固定大小的页面，允许非连续存储，提高了内存利用率。大语言模型通过自监督学习进行训练，即通过预测下一步文本来学习语言模式，这种学习方法使大语言模型可以在没有人工标注数据的情况下进行训练。

## 7. 实验结果分析与可视化

从显存占用来看，关闭 KVCache 时显存占用最少，因为不需要额外空间存储 KV 值，但这是以计算时间为代价的。开启 KVCache 后，显存占用明显增加，因为需要存储先前所有 token 的键值对。使用 PagedAttention 后，显存占用进一步增加，这是因为分页机制需要额外的元数据来管理内存页面，但这种方法能够支持更长的序列生成。

在推理延迟方面，关闭 KVCache 的方案延迟最高，因为每次生成都需要重新计算整个序列的注意力。开启 KVCache 后，延迟显著降低，因为只需要计算最新 token 的注意力权重。PagedAttention 在延迟方面表现最佳，因为它不仅利用了 KVCache，还通过优化的内存访问模式减少了内存碎片和访问延迟。

在实际应用中，KVCache 优化通常与其他技术结合使用，如量化、剪枝和蒸馏等。对于极长序列生成，还可以考虑稀疏注意力只计算与最近 token 的注意力，减少计算量；线性注意力使用线性复杂度的注意力变体；内存换计算在内存充足时缓存更多中间结果。


```python
def practical_kv_cache_usage():
    """
    演示在实际项目中使用 KVCache 的最佳实践
    """
    input_text = "深度学习中的优化技术包括"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # HuggingFace 模型默认启用 use_cache
    outputs = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"实际应用生成文本: {generated_text}")

# 运行实际应用示例
practical_kv_cache_usage()
```

## 8. 总结与思考

通过本实验，我们验证了 KVCache 技术在提高大语言模型推理效率方面的重要作用。从完全关闭缓存到启用缓存，再到更高级的 PagedAttention 优化，每一步都带来了显著的性能提升。
