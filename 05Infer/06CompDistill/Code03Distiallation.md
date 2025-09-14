<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 02: 大模型 Qwen3 蒸馏

模型蒸馏（Knowledge Distillation）是一种让小型学生模型（Student Model）学习大型教师模型（Teacher Model）的知识和行为的技术，旨在让小模型以更少的参数实现接近大模型的性能。

本次实验使用 Qwen3-4B 作为教师模型，指导 Qwen3-0.6B 学生模型进行训练。通过蒸馏，我们希望 Qwen3-0.6B 能在特定任务（如数学推理、代码生成）上获得接近 Qwen3-4B 的表现，同时保持较小的参数规模和计算开销。

## 1. 环境准备

首先安装必要的库：PyTorch、Transformers、Hugging Face Hub 和 Datasets。以下代码块用于设置环境：

```python
# 安装依赖库
!pip install torch transformers huggingface_hub datasets

# 导入所需模块
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import numpy as np
```

## 2. 蒸馏的核心思想

模型蒸馏的目的是将教师模型（Teacher）的知识“转移”到学生模型（Student）中。这里的关键在于**软标签（Soft Targets）**：教师模型输出的概率分布比原始数据的硬标签包含更多信息，例如类别间的相似性（即“暗知识”）。蒸馏通过最小化学生模型与教师模型输出的差异来实现知识转移。

蒸馏通常结合两种损失：

1.  **蒸馏损失（Distillation Loss）**：使用 KL 散度（Kullback-Leibler Divergence）衡量学生模型与教师模型输出的概率分布差异。
2.  **学生损失（Student Loss）**：学生模型与真实标签的交叉熵损失。

总损失是两者的加权和：  

$$
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{KL} + (1 - \alpha) \cdot \mathcal{L}_{CE}
$$  

其中 $\alpha$ 是权重系数（通常设为 0.5-0.7），$\mathcal{L}_{KL}$ 是 KL 散度损失，$\mathcal{L}_{CE}$ 是交叉熵损失。

在 Softmax 函数中引入温度 $T$ 可以平滑概率分布：  

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$  

更高的 $T$ 值会使分布更平滑，揭示更多类别间关系。

## 3. 数据准备

我们使用简单的指令跟随数据集进行演示（如数学问题或代码生成任务）。这里以 `timdettmers/openassistant-guanaco` 数据集为例（包含指令-响应对）：

```python
# 加载数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 预处理函数：将数据转换为模型输入的格式
def preprocess_function(examples):
    prompts = ["Instruction: " + q + "\nResponse: " for q in examples['instruction']]
    return {"text": prompts}

# 选取子集以简化实验（500 条样本）
small_dataset = dataset.select(range(500)).map(preprocess_function, batched=True)
```

`load_dataset` 从 Hugging Face 加载数据集。`preprocess_function` 将指令和响应格式化为模型输入（例如："Instruction: What is 2+2?\nResponse: 4"）。

## 4. 教师和学生模型

使用 Hugging Face 的 `AutoModelForCausalLM` 加载 Qwen3-4B（教师）和 Qwen3-0.6B（学生）：

```python
# 定义模型名称
teacher_model_name = "Qwen/Qwen3-4B"
student_model_name = "Qwen/Qwen3-0.6B"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌

# 加载教师模型（使用 float16 节省显存）
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, torch_dtype=torch.float16, device_map="auto"
)

# 加载学生模型（同样使用 float16）
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name, torch_dtype=torch.float16, device_map="auto"
)
```
`device_map="auto"` 自动将模型分配到可用设备（GPU/CPU）。`torch.float16` 减少显存占用，但可能略微影响精度（蒸馏中可接受）。分词器使用教师模型的版本，确保输入处理一致。

## 5. 定义蒸馏损失函数

我们需要自定义损失函数，结合 KL 散度和交叉熵损失：

```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=5.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 计算蒸馏损失（KL 散度）
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 计算学生损失（交叉熵）
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        # 结合损失
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss
```

其中，

- `alpha` 控制蒸馏损失与交叉熵损失的权重。
- `temperature` 平滑概率分布（更高值使教师输出更柔和）。
- `kl_loss` 计算学生与教师软标签的 KL 散度。
- `ce_loss` 计算学生输出与真实标签的交叉熵。

## 6. 微调蒸馏循环

下面实现蒸馏训练循环（简化版）：

```python
# 定义优化器（AdamW）和损失函数
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
distill_loss_fn = DistillationLoss(alpha=0.7, temperature=5.0)

# 训练参数
epochs = 3  # 蒸馏通常需要较少轮次
batch_size = 4  # 小批量以节省显存

# 训练循环
for epoch in range(epochs):
    student_model.train()
    total_loss = 0.0

    for i in range(0, len(small_dataset), batch_size):
        # 准备批量数据
        batch_texts = small_dataset["text"][i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(student_model.device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()

        # 教师模型推理（禁用梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=False)

        # 学生模型推理
        student_outputs = student_model(**inputs, labels=labels)

        # 计算蒸馏损失
        loss = distill_loss_fn(
            student_outputs.logits, teacher_outputs.logits, labels
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Average Loss: {total_loss / (len(small_dataset)/batch_size):.4f}")
```

教师模型在推理时禁用梯度（`torch.no_grad()`），以减少计算和显存开销。使用小批量（`batch_size=4`）适应有限显存。损失函数同时考虑教师输出（软标签）和真实标签。

## 7. 评估蒸馏效果

训练后，我们在测试集上比较学生模型与教师模型的性能。使用简单的准确率（Accuracy）或困惑度（Perplexity）作为指标：

```python
# 评估函数
def evaluate_model(model, test_data):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for text in test_data["text"]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
            labels = inputs["input_ids"]
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(test_data))).item()
    return perplexity

# 加载测试数据
test_dataset = load_dataset("timdettmers/openassistant-guanaco", split="test").select(range(100))

# 计算教师和学生的困惑度
teacher_ppl = evaluate_model(teacher_model, test_dataset)
student_ppl = evaluate_model(student_model, test_dataset)

print(f"Teacher Perplexity: {teacher_ppl:.2f}")
print(f"Student Perplexity: {student_ppl:.2f}")
```

**困惑度（Perplexity）** 衡量模型预测能力（越低越好）。蒸馏后，学生模型的困惑度应接近教师模型。实际应用中还可使用任务特定指标（如数学问题的准确率）。

## 8. 总结与思考

在本实验中，我们期望蒸馏后的 Qwen3-0.6B 性能显著提升。例如，在测试集上，学生模型的困惑度可能从原始值（例如 30+）降低到接近教师模型的水平（例如 15-20）。然而，蒸馏效果受多种因素影响：

1.  **数据质量**：高质量、多样化的数据能提升蒸馏效果。Qwen3 预训练数据涵盖多语言和多种领域（如代码、数学），这有助于蒸馏。
2.  **超参数选择**：温度参数 $\alpha$ 和 $T$ 需要调优。过高的 $T$ 可能使分布过于平滑，而过低的 $\alpha$ 可能忽略教师知识。
3.  **模型容量差距**：学生模型过小可能无法完全吸收教师知识（Qwen3-0.6B 与 Qwen3-4B 的参数量比约为 1:6.7，差距适中）。
