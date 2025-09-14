<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 02: FastVLM-0.5B 量化对比

在深度学习应用中，模型的大小和计算效率往往是实际部署时需要考虑的关键因素。特别是对于像 FastVLM 这样的视觉语言模型，即使是 0.5B 参数规模，在资源受限的设备上运行也可能面临挑战。

模型量化是解决这一问题的有效方法，它通过减少模型参数和计算的数值精度来降低显存占用并提高推理速度。今天我们就来实际对比不同量化策略对模型性能的影响。

## 1. 模型量化基础

量化的核心思想是将神经网络中的浮点数权重和激活值转换为定点数表示。最常用的是将 32 位浮点数（FP32）转换为更低位数的整数表示。

对于权重量化，我们通常使用以下公式将浮点数转换为整数：

$$ q = \text{round}(r / s + z) $$

其中：

- $r$ 是原始浮点数值
- $s$ 是缩放因子（scale）
- $z$ 是零点偏移（zero point）
- $q$ 是量化后的整数值

常见的量化配置有：

- W4A4：权重和激活值都使用 4 位整数
- W8A8：权重和激活值都使用 8 位整数 
- W4A16：权重使用 4 位整数，激活值使用 16 位整数

## 2. 实验环境准备

首先，我们需要安装必要的库。我们将使用 Hugging Face 的 transformers 库加载模型，以及 accelerate 库来帮助管理显存使用。

```python
# 安装所需库
!pip install transformers accelerate torch pillow
```

然后，我们导入必要的模块：

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from PIL import Image
import os
```

## 3. 加载 FastVLM-0.5B

让我们先加载原始的 FastVLM-0.5B 模型，作为基准参考。

```python
# 定义模型名称和设备
model_name = "apple/FastVLM-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 检查是否有可用的 GPU
if device != "cuda":
    print("警告：未检测到 GPU，实验效果可能受影响")
else:
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

# 加载分词器和图像处理工具
tokenizer = AutoTokenizer.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

# 加载原始 FP16 模型（作为基准）
print("正在加载原始 FP16 模型...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

这段代码首先检查我们是否有可用的 GPU，因为量化实验在 GPU 上效果更明显。然后我们加载了模型的分词器和图像处理工具，最后加载了原始的 FP16 精度模型作为基准。

注意我们使用了`torch_dtype=torch.float16`参数，这会将模型加载为半精度（16 位）而不是默认的 32 位，这已经是一种简单的量化形式了。

## 4. 准备数据和评估函数

为了公平比较不同量化配置，我们需要相同的测试数据和评估方法。

```python
# 准备测试图像和问题
def load_test_image():
    """创建一张简单的测试图像（白色背景，黑色方块）"""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (224, 224), color='white')
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 50), (150, 150)], fill='black')
    return img

test_image = load_test_image()
test_question = "这张图片中有什么？请描述一下。"

# 定义评估函数
def evaluate_model(model, image, question, tokenizer, image_processor, device):
    """评估模型的显存占用和推理延迟"""
    # 预处理输入
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    text_inputs = tokenizer(question, return_tensors="pt").to(device)
    
    # 测量推理延迟（多次运行取平均）
    start_time = time.time()
    iterations = 5  # 多次运行取平均
    for _ in range(iterations):
        with torch.no_grad():  # 禁用梯度计算，节省显存和计算
            outputs = model.generate(
                **inputs,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                max_new_tokens=50
            )
    end_time = time.time()
    avg_latency = (end_time - start_time) / iterations * 1000  # 转换为毫秒
    
    # 测量显存占用
    torch.cuda.synchronize()  # 等待所有操作完成
    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为 MB
    
    # 解码输出结果
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "answer": answer,
        "avg_latency_ms": avg_latency,
        "memory_used_mb": memory_used
    }
```

## 5. 基准测试：FP16 模型

让我们先测试原始的 FP16 模型作为基准：

```python
# 清空缓存，确保测量准确
torch.cuda.empty_cache()

# 评估 FP16 模型
print("正在评估 FP16 模型...")
fp16_results = evaluate_model(
    model_fp16, 
    test_image, 
    test_question, 
    tokenizer, 
    image_processor, 
    device
)

print(f"FP16 模型结果:")
print(f"答案: {fp16_results['answer']}")
print(f"平均延迟: {fp16_results['avg_latency_ms']:.2f} ms")
print(f"显存占用: {fp16_results['memory_used_mb']:.2f} MB")
```

在进行评估前，我们调用了`torch.cuda.empty_cache()`来清空 GPU 缓存，确保显存测量的准确性。然后我们使用前面定义的评估函数来测试模型。

## 6. 量化配置 1：W8A8

现在让我们尝试 8 位量化，这是一种常用的平衡性能和精度的量化策略。

```python
# 安装 bitsandbytes 库用于量化
!pip install bitsandbytes

# 加载 8 位量化模型
from transformers import BitsAndBytesConfig

# 配置 8 位量化参数
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,  # 启用 8 位量化
)

# 清空缓存
torch.cuda.empty_cache()

# 加载 8 位量化模型
print("正在加载 W8A8 量化模型...")
model_w8a8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8bit,
    device_map="auto"
)

# 评估 8 位量化模型
print("正在评估 W8A8 模型...")
w8a8_results = evaluate_model(
    model_w8a8, 
    test_image, 
    test_question, 
    tokenizer, 
    image_processor, 
    device
)

print(f"W8A8 模型结果:")
print(f"答案: {w8a8_results['answer']}")
print(f"平均延迟: {w8a8_results['avg_latency_ms']:.2f} ms")
print(f"显存占用: {w8a8_results['memory_used_mb']:.2f} MB")
```

这里我们使用了 bitsandbytes 库提供的 8 位量化功能。通过设置`load_in_8bit=True`，我们告诉库将模型权重加载为 8 位整数。

理论上，8 位量化可以将模型大小减少约 4 倍（从 32 位浮点数到 8 位整数），但实际显存节省可能略少，因为还需要存储一些量化参数（如缩放因子）。

## 7. 量化配置 2：W4A4

接下来，我们尝试更激进的 4 位量化，这会进一步减少模型大小和显存占用。

```python
# 配置 4 位量化参数
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4 位量化
    bnb_4bit_use_double_quant=True,  # 使用双量化，进一步节省空间
    bnb_4bit_quant_type="nf4",  # 使用正态分布量化
)

# 清空缓存
torch.cuda.empty_cache()

# 加载 4 位量化模型（W4A4）
print("正在加载 W4A4 量化模型...")
model_w4a4 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit,
    device_map="auto"
)

# 评估 4 位量化模型
print("正在评估 W4A4 模型...")
w4a4_results = evaluate_model(
    model_w4a4, 
    test_image, 
    test_question, 
    tokenizer, 
    image_processor, 
    device
)

print(f"W4A4 模型结果:")
print(f"答案: {w4a4_results['answer']}")
print(f"平均延迟: {w4a4_results['avg_latency_ms']:.2f} ms")
print(f"显存占用: {w4a4_results['memory_used_mb']:.2f} MB")
```

在 4 位量化配置中，我们使用了一些额外的优化：

- `bnb_4bit_use_double_quant=True`：启用双量化，对量化参数本身也进行量化
- `bnb_4bit_quant_type="nf4"`：使用正态分布感知量化，这通常比均匀量化保留更好的精度

4 位量化理论上可以比 32 位浮点数减少 8 倍的存储空间，是资源受限环境下的理想选择。

## 8. 量化配置 3：W4A16

最后，我们测试一种混合量化策略：权重使用 4 位，激活使用 16 位。这种配置试图在节省显存和保持推理精度之间取得更好的平衡。

```python
# 配置 W4A16 量化参数
bnb_config_w4a16 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # 计算使用 16 位浮点数
)

# 清空缓存
torch.cuda.empty_cache()

# 加载 W4A16 量化模型
print("正在加载 W4A16 量化模型...")
model_w4a16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_w4a16,
    device_map="auto"
)

# 评估 W4A16 量化模型
print("正在评估 W4A16 模型...")
w4a16_results = evaluate_model(
    model_w4a16, 
    test_image, 
    test_question, 
    tokenizer, 
    image_processor, 
    device
)

print(f"W4A16 模型结果:")
print(f"答案: {w4a16_results['answer']}")
print(f"平均延迟: {w4a16_results['avg_latency_ms']:.2f} ms")
print(f"显存占用: {w4a16_results['memory_used_mb']:.2f} MB")
```

这里的关键参数是`bnb_4bit_compute_dtype=torch.float16`，它指定了在计算过程中（主要是激活值）使用 16 位浮点数，而不是 4 位整数。这种配置可以减少量化误差，尤其是在激活值动态范围较大的情况下。

## 9. 实验结果对比与分析

现在让我们汇总所有实验结果，进行对比分析：

```python
# 汇总结果
results = {
    "FP16": fp16_results,
    "W8A8": w8a8_results,
    "W4A4": w4a4_results,
    "W4A16": w4a16_results
}

# 打印对比表格
print(f"{'配置':<8} {'延迟(ms)':<10} {'显存(MB)':<10}")
print("-" * 30)
for config, res in results.items():
    print(f"{config:<8} {res['avg_latency_ms']:<10.2f} {res['memory_used_mb']:<10.2f}")

# 计算相对值（相对于 FP16）
print("\n 相对值（相对于 FP16）:")
print(f"{'配置':<8} {'延迟比例':<10} {'显存比例':<10}")
print("-" * 30)
fp16_latency = results["FP16"]["avg_latency_ms"]
fp16_memory = results["FP16"]["memory_used_mb"]

for config, res in results.items():
    latency_ratio = res["avg_latency_ms"] / fp16_latency
    memory_ratio = res["memory_used_mb"] / fp16_memory
    print(f"{config:<8} {latency_ratio:<10.2f} {memory_ratio:<10.2f}")
```

这段代码会以表格形式展示所有配置的延迟和显存占用，并计算它们相对于 FP16 基准的比例。

从理论上我们可以预期：

- 显存占用：W4A4 < W4A16 < W8A8 < FP16
- 推理延迟：通常量化程度越高，延迟越低，但这也取决于硬件支持

除了这些数值指标，我们还应该关注模型的输出质量是否有明显下降。如果量化后的模型生成的答案质量严重下降，那么即使显存和延迟有优势，这种量化配置也可能不适用。

## 10. 总结与思考

在显存占用方面，4 位量化（W4A4 和 W4A16）相比 FP16 可以节省显著的显存空间，通常能达到 70-80%的减少，而 8 位量化（W8A8）则可以节省约 40-50%的显存。这种显存占用的减少对于在资源受限设备上部署大模型尤为重要。

推理延迟方面，量化通常会带来推理速度的提升，但提升幅度取决于具体硬件和量化实现。一般来说，4 位量化可能比 8 位量化更快，但也可能因为需要更多的反量化操作而抵消部分优势。实际测量中，W4A16 配置通常在延迟和精度之间提供了较好的平衡。

精度权衡是量化技术中需要重点考虑的因素。更高程度的量化（如 W4A4）可能会导致模型精度下降，特别是在复杂任务上。W4A16 这种混合配置通常能在节省显存和保持精度之间取得更好的平衡，尤其是对于多模态模型中的视觉特征处理部分。
