<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 03:多模态输出采样与控制

> 多模态生成模型正在改变我们创造和理解内容的方式，而采样策略则是控制这一过程的隐形艺术家。

在当前的人工智能浪潮中，多模态生成模型已经成为内容创作的重要工具。无论是通过文字描述生成图像，还是根据静态图像创建视频，这些模型都展现出了惊人的能力。

但你是否曾经好奇，为什么同样的输入提示，有时会产生令人惊艳的结果，有时却平平无奇？这背后往往取决于采样策略的选择。

今天我们将深入探讨温度（Temperature）和 Top-P（核采样）两种采样策略如何影响多模态生成模型的输出结果，以及如何通过调整这些参数来平衡生成结果的**多样性**、**创造性**和**质量**。

### 1. Temperature 采样

温度参数或许是控制生成随机性最直观的方式。它在数学上调整了模型输出概率分布的平滑程度。

给定原始概率分布 $P(x_i|x_{<i})$，应用温度参数 $T$ 后的新概率分布为：

$$\hat{P}(x_i|x_{<i}) = \frac{\exp(\frac{z_i}{T})}{\sum_j \exp(\frac{z_j}{T})}$$

其中 $z_i$ 是模型输出的 logits 值。

**温度参数的影响**：

- 当 $T > 1$ 时，概率分布变得更加平滑，生成结果更加多样但可能不够准确
- 当 $T < 1$ 时，概率分布更加尖锐，生成结果更加确定但可能缺乏创造性

为了更好地理解这一概念，让我们通过代码来实现温度采样的效果：

```python
import torch
import torch.nn.functional as F

def apply_temperature(logits, temperature):
    if temperature > 0:
        # 应用温度缩放
        scaled_logits = logits / temperature
        # 重新计算 softmax 概率
        probs = F.softmax(scaled_logits, dim=-1)
        return probs
    else:
        raise ValueError("温度参数必须大于 0")

# 示例用法
logits = torch.tensor([2.0, 1.0, 0.5, 0.2])
print("原始概率:", F.softmax(logits, dim=-1).numpy())

for temp in [0.5, 1.0, 2.0]:
    temp_probs = apply_temperature(logits, temp)
    print(f"温度 {temp} 后的概率: {temp_probs.numpy()}")
```

这段代码展示了温度参数如何影响概率分布。当温度值较低时（如 0.5），概率分布更加集中，模型更倾向于选择最高概率的选项；而当温度值较高时（如 2.0），概率分布更加平滑，模型的选择更加多样化。

## 2. Top-P 采样

Top-P 采样，也称为核采样，选择概率累积超过阈值 p 的最小可能词元集合，然后从这个集合中重新归一化概率并采样。

形式上，给定概率分布 $P$ 和阈值 $p ∈ (0, 1]$，我们按概率降序排列，找到最小的集合 $S$ 使得：

$$\sum_{x_i \in S} P(x_i) \geq p$$

然后从集合 $S$ 中按照重新归一化的概率进行采样。

```python
def top_p_sampling(probs, p):
    # 对概率进行排序
    sorted_probs, indices = torch.sort(probs, descending=True)
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 移除累积概率超过 p 的部分
    indices_to_remove = cumulative_probs > p
    # 确保至少保留一个 token
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = 0
    
    # 将需要移除的 token 概率设为 0
    sorted_probs[indices_to_remove] = 0
    # 重新归一化概率
    sorted_probs /= sorted_probs.sum()
    
    # 恢复到原始顺序
    return sorted_probs.scatter(-1, indices, sorted_probs)

# 示例用法
probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
print("原始概率:", probs.numpy())

for p in [0.7, 0.9]:
    top_p_probs = top_p_sampling(probs, p)
    print(f"Top-P (p={p}) 后的概率: {top_p_probs.numpy()}")
```

Top-P 采样的优势在于它能够动态调整候选集的大小，既保证了多样性，又避免了选择概率极低的选项。

## 3. 文生图应用实践

了解了采样策略的基本原理后，让我们看看它们在实际的多模态生成任务中如何应用。我们将使用 Stable Diffusion 模型进行文本到图像的生成实验。

首先，我们需要设置环境并加载模型：

```python
from diffusers import StableDiffusionXLPipeline
import torch

# 加载 SDXL 管道
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# 将管道移动到 GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
```

虽然 Stable Diffusion 本身不直接暴露温度参数，但我们可以通过修改生成过程中的随机性来模拟类似效果。实际上，在扩散模型中，类似的随机性控制可以通过调整 guidance_scale 和随机种子来实现。

```python
def generate_images_with_variation(prompt, num_images=4, guidance_scale=7.5):
    images = []
    for i in range(num_images):
        # 使用不同的随机种子
        generator = torch.Generator(device=device).manual_seed(i)
        
        # 生成图像
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        images.append(image)
    
    return images

# 设置生成参数
prompt = "一个美丽的日落海滩，有椰子树和金色的沙滩"
negative_prompt = "模糊，失真，低质量"

# 生成图像
images = generate_images_with_variation(prompt)
```

在实际应用中，我们往往需要更精细的控制，而不仅仅是调整随机种子。多模态生成的可控性技术正在不断发展，例如通过 ControlNet 实现空间精准定位，通过 LoRA 注入特定规则，以及通过 CLIP 进行情感校准等方法。

## 4. 多模态实验

除了文生图应用，采样策略也对多模态语言模型的输出有重要影响。让我们以视觉语言模型为例，看看不同采样参数如何影响模型生成的描述。

我们将使用 Qwen-VL 模型进行图文对话实验，观察不同温度和 Top-P 参数如何影响生成的图像描述。

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# 初始化模型和处理器
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 初始化 LLM
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

def generate_descriptions_with_sampling(image_path, prompt_text, sampling_configs):
    # 构建消息
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    
    # 应用聊天模板
    prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    descriptions = []
    for config in sampling_configs:
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=256,
        )
        
        # 生成描述
        outputs = llm.generate([{"prompt": prompt}], sampling_params=sampling_params)
        description = outputs[0].outputs[0].text.strip()
        descriptions.append(description)
    
    return descriptions

# 定义不同的采样配置
sampling_configs = [
    {"temperature": 0.1, "top_p": 0.9, "name": "低温度，高 Top-P"},
    {"temperature": 0.7, "top_p": 0.9, "name": "中等温度，高 Top-P"},
    {"temperature": 1.2, "top_p": 0.9, "name": "高温度，高 Top-P"},
    {"temperature": 0.7, "top_p": 0.3, "name": "中等温度，低 Top-P"},
]

# 生成描述（需要实际图像路径）
image_path = "path_to_your_image.jpg"
prompt_text = "描述这张图片中的场景和细节"
descriptions = generate_descriptions_with_sampling(image_path, prompt_text, sampling_configs)
```

通过这个实验，我们可以观察到不同采样配置下模型生成的描述有何不同。低温度配置往往产生更加保守和确定的描述，而高温度配置则可能产生更加创造性和多样化的描述，但也可能增加不相关或虚构内容的风险。

## 5. 评估生成结果

评估多模态生成结果的质量是一个复杂的任务，需要从多个维度进行考量。常用的评估指标包括：

1.  **模态对齐度（MDA）**：衡量生成内容与输入提示之间的一致性。
2.  **细节保真度（DF）**：评估生成内容的细节丰富程度和准确性。
3.  **多样性**：衡量不同生成结果之间的差异程度。
4.  **创造性**：评估生成内容的新颖性和创新程度。

我们可以通过计算一些定量指标来评估生成结果的多样性：

```python
from collections import Counter
import math

def calculate_diversity(descriptions):
    """
    计算生成描述的多样性指标
    """
    all_words = []
    for desc in descriptions:
        words = desc.lower().split()
        all_words.extend(words)
    
    # 计算词汇总量和唯一词汇量
    total_words = len(all_words)
    unique_words = len(set(all_words))
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    
    return {
        "lexical_diversity": lexical_diversity,
        "unique_words": unique_words,
        "total_words": total_words
    }

# 计算生成描述的多样性
diversity_metrics = calculate_diversity(descriptions)
```

除了这些定量指标，人工评估仍然是评估生成质量的重要方式，特别是对于创造性和审美价值的判断。

## 6. 总结与思考

采样策略在多模态生成中扮演着至关重要的角色，它们像是隐形的艺术家，默默地影响着生成结果的多样性、创造性和质量。通过理解和掌握温度参数和 Top-P 采样等策略，我们能够更好地驾驭多模态生成模型，创造出更加符合期望的内容。

需要注意的是，参数调整并非万能，它需要在模型能力、任务需求和使用场景之间找到平衡点。有时候，**创造力的提升可能会以降低精确性为代价**，而**过于追求确定性又可能抑制创新**。这正是多模态生成既是一门科学也是一门艺术的原因。

希望本文能够为你理解和应用采样策略提供有益的指导，帮助你在多模态生成的探索之旅中走得更远。记住，最好的参数配置往往来自于不断的实验和调整，而不是一成不变的公式。
