<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 02: Min-P vs Top-P 探索

在自然语言生成领域，采样策略的选择对生成文本的质量和多样性影响深远。本文将探讨一种新兴的采样方法——最小 P 采样（Min-P Sampling），并与广泛使用的 Top-P（核）采样进行对比实验。

通过直观的代码示例和深入分析，我们希望帮助读者理解这两种方法的原理与差异，特别关注它们在 Qwen3 4B 模型上的表现。

## 1. Top-P 采样实现

Top-P 采样从概率分布中选择概率累积和超过阈值 p 的最小 token 集合，然后从这个集合中重新归一化概率并采样。这种方法确保只考虑概率较高的 token，同时保持一定的多样性。

数学表达式为：$V_{\text{Top-P}} = \{v_i \in V \mid \sum_{j=1}^{i} p(v_j) \geq p\}$，其中 $V$ 是按概率降序排列的词汇表。

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9):
    """
    实现 Top-P 采样策略
    
    参数:
        logits: 模型输出的原始 logits
        p: 累积概率阈值
        
    返回:
        采样得到的 token 索引
    """
    # 将 logits 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 对概率进行降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 移除累积概率超过 p 的 token
    indices_to_remove = cumulative_probs > p
    # 确保至少保留一个 token
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = False
    
    # 创建过滤后的概率分布
    sorted_indices_to_remove = sorted_indices[indices_to_remove]
    probs[sorted_indices_to_remove] = 0
    
    # 重新归一化概率
    filtered_probs = probs / probs.sum()
    
    # 从过滤后的分布中采样
    next_token = torch.multinomial(filtered_probs, num_samples=1)
    
    return next_token.item()
```

## 2. Min-P 采样实现

Min-P 采样是 Top-P 采样的变体，它设置一个最小概率阈值而不是累积概率阈值。具体来说，它保留所有概率大于最小阈值 $p_{\text{min}}$ 的 token，然后从这些 token 中采样。这种方法提供了一种更直接的概率阈值控制方式。

数学表达式为：$V_{\text{min-p}} = \{v_i \in V \mid p(v_i) \geq p_{\text{min}}\}$

```python
def min_p_sampling(logits, min_prob=0.05):
    """
    实现 Min-P 采样策略
    
    参数:
        logits: 模型输出的原始 logits
        min_prob: 最小概率阈值
        
    返回:
        采样得到的 token 索引
    """
    # 将 logits 转换为概率
    probs = F.softmax(logits, dim=-1)
    
    # 移除概率低于阈值的 token
    indices_to_remove = probs < min_prob
    probs[indices_to_remove] = 0
    
    # 如果所有 token 都被移除，则保留概率最高的 token
    if probs.sum() == 0:
        probs = F.softmax(logits, dim=-1)
        probs[1:] = 0  # 只保留概率最高的 token
        probs[0] = 1
    
    # 重新归一化概率
    filtered_probs = probs / probs.sum()
    
    # 从过滤后的分布中采样
    next_token = torch.multinomial(filtered_probs, num_samples=1)
    
    return next_token.item()
```

## 3. 实验设置

为了对比这两种采样策略在现代化模型上的效果，我们选择使用 Qwen3 4B 模型进行实验。这个模型在参数量和性能之间取得了良好平衡，适合进行采样策略的对比研究。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 Qwen3 4B 模型和分词器
model_name = "Qwen/Qwen2-4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 设置填充 token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 将模型设置为评估模式
model.eval()
```

## 4. 文本生成

我们编写一个通用的文本生成函数，它可以接受不同的采样策略，并在 Qwen3 4B 模型上进行文本生成。

```python
def generate_text(prompt, sampling_function, sampling_param, max_length=50):
    """
    使用指定的采样策略生成文本
    
    参数:
        prompt: 输入文本提示
        sampling_function: 采样函数
        sampling_param: 采样参数
        max_length: 生成的最大长度
        
    返回:
        生成的文本
    """
    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # 存储生成的 token
    generated = input_ids
    
    # 使用 no_grad 避免计算梯度
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            
            # 使用指定的采样策略获取下一个 token
            next_token = sampling_function(next_token_logits, sampling_param)
            
            # 将新 token 添加到已生成序列中
            generated = torch.cat([generated, torch.tensor([[next_token]], device=model.device)], dim=-1)
            
            # 如果生成了结束 token，停止生成
            if next_token == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

## 5. 对比实验

现在我们来对比两种采样策略在不同参数下的文本生成效果。首先定义一个测试函数来批量生成文本。

```python
def compare_sampling_strategies(prompt, top_p_values, min_p_values):
    """
    对比不同参数下的采样策略效果
    
    参数:
        prompt: 输入提示
        top_p_values: 要测试的 Top-P 值列表
        min_p_values: 要测试的 min-p 值列表
    """
    print(f"输入提示: '{prompt}'\n")
    
    # 测试 Top-P 采样
    print("Top-P 采样结果:")
    for p in top_p_values:
        generated_text = generate_text(prompt, top_p_sampling, p)
        print(f"p={p}: {generated_text[len(prompt):]}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试 Min-P 采样
    print("Min-P 采样结果:")
    for min_prob in min_p_values:
        generated_text = generate_text(prompt, min_p_sampling, min_prob)
        print(f"min_prob={min_prob}: {generated_text[len(prompt):]}")
```

使用相同的提示文本，我们观察两种采样策略在 Qwen3 4B 模型上的生成效果。选择适当的参数范围对于观察差异至关重要。

```python
# 设置测试参数
test_prompt = "人工智能的未来发展将"
top_p_values = [0.5, 0.8, 0.9]
min_p_values = [0.01, 0.05, 0.1]

# 运行对比实验
compare_sampling_strategies(test_prompt, top_p_values, min_p_values)
```

从生成结果中，我们可以观察到一些有趣的现象。较低的 p 值（如 0.5）在 Top-P 采样中会导致更保守的生成，只考虑概率最高的几个 token，而较高的 p 值（如 0.9）会允许更多样化的生成，但可能包含一些不太相关的 token。生成结果的质量和多样性对 p 值非常敏感，这反映了 Top-P 采样的动态特性。

对于 Min-P 采样，较低的最小概率阈值（如 0.01）会保留更多 token，导致更多样化的生成，而较高的最小概率阈值（如 0.1）会更严格，只保留概率较高的 token。相对于 Top-P，Min-P 提供了一种更直接的概率阈值控制方式，使生成过程更加稳定和可预测。

## 6. 可视化分析

为了更直观地理解两种方法的差异，我们可以可视化它们对概率分布的过滤效果。

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sampling_effect(logits, top_p=0.9, min_p=0.05):
    """
    可视化两种采样策略对概率分布的影响
    """
    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    
    # 对概率进行排序
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Top-P 可视化
    top_p_mask = cumulative_probs <= top_p
    ax1.bar(range(len(sorted_probs)), sorted_probs, alpha=0.7)
    ax1.bar(range(len(sorted_probs))[top_p_mask], sorted_probs[top_p_mask], color='red', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_title(f'Top-P Sampling (p={top_p})')
    ax1.set_xlabel('Token Rank')
    ax1.set_ylabel('Probability')
    
    # Min-P 可视化
    min_p_mask = sorted_probs >= min_p
    ax2.bar(range(len(sorted_probs)), sorted_probs, alpha=0.7)
    ax2.bar(range(len(sorted_probs))[min_p_mask], sorted_probs[min_p_mask], color='green', alpha=0.7)
    ax2.axhline(y=min_p, color='k', linestyle='--', alpha=0.7, label=f'Min-p threshold ({min_p})')
    ax2.set_title(f'Min-P Sampling (min_p={min_p})')
    ax2.set_xlabel('Token Rank')
    ax2.set_ylabel('Probability')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 获取一个示例 logits 分布
with torch.no_grad():
    test_input = tokenizer.encode("人工智能的", return_tensors="pt").to(model.device)
    outputs = model(test_input)
    sample_logits = outputs.logits[:, -1, :].squeeze().cpu()

# 可视化
visualize_sampling_effect(sample_logits, top_p=0.9, min_p=0.05)
```

通过可视化分析，我们可以清楚地看到两种采样方法如何过滤概率分布。Top-P 采样保留累积概率达到阈值的最少 token 集合，而 Min-P 采样保留所有概率超过最小阈值的 token。这种差异在实际生成过程中会导致不同的文本质量和多样性特征。

## 7. 总结与思考

从计算效率角度来看，两种方法在复杂度上相似，都需要对概率进行排序和过滤。在实际应用中，最佳采样策略和参数往往需要根据具体任务和模型进行调优。

Min-P 采样作为 Top-P 采样的新兴变体，提供了另一种控制文本生成质量的方式。随着大语言模型技术的不断发展，采样策略的研究也将继续深入，为自然语言生成任务提供更多灵活性和控制能力。
