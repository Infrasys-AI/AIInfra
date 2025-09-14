<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 01: Qwen3-4B 模型微调

大型语言模型（LLM）的微调技术是将预训练模型适配到特定任务的关键环节。面对不同的数据特性和资源约束，选择合适的微调方法至关重要。

本文将使用**Qwen3-4B**模型作为基础模型，对比全参数微调、LoRA（Low-Rank Adaptation）、Prompt Tuning 和指令微调四种主流技术，分析它们在**效果、效率和数据需求**方面的差异，并探索**数据集类型**（通用/领域/小样本）与微调技术的适配关系。

## 2. 实验设置

首先安装必要的库：

```python
# 安装所需库
!pip install transformers peft datasets torch accelerate unsloth
```

```python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PromptTuningConfig, TaskType
from datasets import load_dataset
import torch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载 Qwen3-4B 模型和 tokenizer
model_name = "Qwen/Qwen3-4B-Instruct"  # 使用 Qwen3-4B 指令微调版本
max_seq_length = 2048  # 最大序列长度
load_in_4bit = True    # 使用 4bit 量化减少显存占用

# 使用 Unsloth 优化加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True  # Qwen 模型需要此参数
)

# 添加 pad_token 以便于批处理
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("模型和分词器加载完成")
print(f"模型参数量: {model.num_parameters()}")
```

## 3. 数据集构建

不同微调技术对数据集格式有不同要求。以下是**指令微调**所需的数据集格式示例：

```python
# 指令微调数据集格式示例
instruction_dataset = [
    {
        "instruction": "判断情感倾向",
        "input": "这部电影的视觉效果很棒，但剧情有些乏味",
        "output": "混合情感：正面评价视觉效果，负面评价剧情",
        "system": "你是一个专业的情感分析助手",
        "history": []
    },
    {
        "instruction": "生成产品描述",
        "input": "智能手机，品牌：Apple，型号：iPhone 15，特点：A17 芯片、4800 万像素相机",
        "output": "Apple iPhone 15 搭载强大的 A17 芯片和 4800 万像素高清相机，提供卓越性能和拍摄体验。",
        "system": "你是一个产品描述生成器",
        "history": []
    },
    {
        "instruction": "翻译成英文",
        "input": "今天天气很好，我们一起去公园吧",
        "output": "The weather is nice today, let's go to the park together.",
        "system": "你是一个翻译助手",
        "history": []
    }
]

# 将示例数据集保存为 JSON 文件
import json
with open("instruction_dataset.json", "w", encoding="utf-8") as f:
    json.dump(instruction_dataset, f, ensure_ascii=False, indent=2)

print("指令微调数据集示例已保存")
```

对于**通用文本生成**任务，数据集格式可以更简单：

```python
# 通用文本数据集格式示例
general_dataset = [
    {
        "text": "大型语言模型（LLM）是人工智能领域的重要突破，它们通过在大量文本数据上进行预训练，学习语言的统计规律和语义表示。"
    },
    {
        "text": "迁移学习使模型能够将在一个任务上学到的知识应用到其他相关任务上，大大减少了数据需求和训练时间。"
    }
]

with open("general_dataset.json", "w", encoding="utf-8") as f:
    json.dump(general_dataset, f, ensure_ascii=False, indent=2)

print("通用文本数据集示例已保存")
```

## 4. 数据预处理

我们需要根据不同的微调方法对数据进行相应处理：

```python
def preprocess_instruction_data(examples):
    """处理指令微调数据"""
    instructions = []
    
    for i in range(len(examples["instruction"])):
        instruction = str(examples["instruction"][i])
        input_text = str(examples["input"][i]) if "input" in examples and examples["input"][i] else ""
        output_text = str(examples["output"][i])
        system_text = str(examples["system"][i]) if "system" in examples and examples["system"][i] else ""
        
        # 构建符合 Qwen3 格式的输入
        if system_text:
            text = f"<|im_start|>system\n{system_text}<|im_end|>\n"
        else:
            text = ""
            
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction
            
        text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output_text}<|im_end|>"
        
        instructions.append(text)
    
    return tokenizer(
        instructions,
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

def preprocess_general_data(examples):
    """处理通用文本数据"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )

# 加载示例数据集（实际应用中替换为真实数据）
dataset = load_dataset("json", data_files="instruction_dataset.json", split="train")

# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_instruction_data,
    batched=True,
    remove_columns=dataset.column_names
)

# 分割训练集和验证集
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"训练集大小: {len(train_dataset)}")
print(f 验证集大小: {len(eval_dataset)}")
```

## 3. 全参数微调

全参数微调通过**反向传播算法更新模型的所有可训练参数**。其数学本质可以表示为：

θ_min = argmin_θ (1/N) * Σ_{i=1}^N L(f_θ(x_i), y_i)

其中 f_θ表示参数化模型，L 为损失函数，N 为样本数量。

这种方法的主要优势是能够充分利用所有模型参数进行任务适配，但缺点是**计算成本高**，对于大模型来说需要大量的显存和计算资源。

```python
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./full_finetune_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 较小的批大小以适应显存
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-6,  # 全参数微调使用较小的学习率
    weight_decay=0.01,
    save_steps=500,
    report_to="none",
    fp16=True,  # 使用混合精度训练节省显存
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练（注释掉实际训练代码以便演示）
# trainer.train()

print("全参数微调完成")
```

全参数微调的主要优点是能够**充分利用模型的全部能力**，通常在数据充足的情况下能达到最佳性能。然而，它的计算成本非常高——对于 Qwen3-4B 这样的模型，需要大量的 GPU 显存和计算时间。

此外，全参数微调还容易导致**灾难性遗忘**，即模型在适应新任务时丢失了预训练中获得的一般知识。

## 4. LoRA 微调

LoRA 是一种**参数高效微调**（PEFT）技术，其核心思想是通过**低秩分解**来限制可训练参数的数量。具体而言，LoRA 将权重更新矩阵ΔW 分解为两个低秩矩阵的乘积：

W + ΔW = W + BA

其中 W 是预训练权重矩阵，A ∈ R^{r×d}和 B ∈ R^{d×r}是低秩矩阵，r 是秩（r << d）。

这种分解的数学基础是**奇异值分解**（SVD）定理，该定理表明任何矩阵都可以被分解为奇异值和奇异向量的乘积，而低秩近似则保留了矩阵中最重要的信息。

```python
# 定义 LoRA 配置
lora_config = LoraConfig(
    r=16,  # 秩
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 创建 LoRA 模型
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # LoRA 可以使用更大的批大小
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=2e-4,  # LoRA 可以使用更大的学习率
    weight_decay=0.01,
    report_to="none",
    fp16=True,
)

# 创建 Trainer
lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练（注释掉实际训练代码以便演示）
# lora_trainer.train()

print("LoRA 微调完成")
```

LoRA 的主要优势在于：
1.  **参数效率**：只需要训练极少量参数（通常小于原模型参数的 1%）
2.  **内存友好**：大幅降低显存需求，使得在消费级 GPU 上微调大模型成为可能
3.  **模块化**：可以为不同任务训练多个适配器，然后灵活切换

实验表明，LoRA 能够保持原始模型大部分性能，同时显著减少训练时间和计算资源需求。

## 5. Prompt 微调

Prompt Tuning 是一种**轻量级微调方法**，它在输入层插入**可训练的虚拟令牌**（virtual tokens），而保持预训练模型的参数不变。这些虚拟令牌作为连续提示，引导模型更好地执行特定任务。

形式上，Prompt Tuning 将原始输入 x 转换为模板化提示 x'，通过构造映射函数 P: X → X'来实现。

```python
# 定义 Prompt Tuning 配置
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # 虚拟令牌数量
    tokenizer_name=model_name
)

# 创建 Prompt Tuning 模型
prompt_model = get_peft_model(model, prompt_config)
prompt_model.print_trainable_parameters()

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./prompt_tuning_results",
    num_train_epochs=5,  # Prompt Tuning 通常需要更多训练轮次
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=3e-3,  # Prompt Tuning 通常需要更高学习率
    weight_decay=0.01,
    report_to="none",
    fp16=True,
)

# 创建 Trainer
prompt_trainer = Trainer(
    model=prompt_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练（注释掉实际训练代码以便演示）
# prompt_trainer.train()

print("Prompt Tuning 完成")
```

Prompt Tuning 的优势在于：
1.  **极高的参数效率**：只需要训练极少量的参数（仅虚拟令牌对应的参数）
2.  **避免灾难性遗忘**：由于原始模型参数被冻结，预训练知识得到保留
3.  **多任务学习**：可以为不同任务学习不同的提示，然后共享同一基础模型

Prompt Tuning 特别适合**少样本学习**场景，但在复杂任务上可能性能不如其他方法。

## 6. 指令微调

指令微调是**监督微调**（SFT）的一种形式，它使用**标注的输入-输出对**进行有监督训练，损失函数通常采用交叉熵（语言建模目标）。与全参数微调不同，指令微调通常专注于使模型遵循指令和完成特定任务格式。

指令微调的核心思想是通过高质量的指令-回答对来训练模型，使其能够更好地理解和遵循人类指令。

```python
# 指令微调需要特定的数据格式
# 这里我们使用前面创建的指令数据集

# 使用 LoRA 进行指令微调（指令微调通常与参数高效方法结合）
instruct_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

instruct_model = get_peft_model(model, instruct_lora_config)
instruct_model.print_trainable_parameters()

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./instruction_tuning_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    weight_decay=0.01,
    report_to="none",
    fp16=True,
)

# 创建 Trainer
instruct_trainer = Trainer(
    model=instruct_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练（注释掉实际训练代码以便演示）
# instruct_trainer.train()

print("指令微调完成")
```

指令微调的优势包括：

1.  **任务特异性**：能够使模型更好地适应特定任务格式和指令
2.  **数据效率**：通常比全参数微调需要更少的数据
3.  **可组合性**：可以与 LoRA 等参数高效方法结合使用

然而，指令微调**依赖高质量标注数据**，如果指令-回答对质量不高，可能会限制模型性能。

## 7. 实验结果与分析

为了评估不同微调方法的性能，我们需要在测试集上计算模型的困惑度（perplexity）或任务特定指标：

```python
def evaluate_model(trainer, dataset):
    # 评估模型性能
    eval_results = trainer.evaluate(eval_dataset=dataset)
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    return {
        "eval_loss": eval_results["eval_loss"],
        "perplexity": perplexity.item()
    }

# 假设我们已经训练了所有模型
# full_finetune_results = evaluate_model(trainer, eval_dataset)
# lora_results = evaluate_model(lora_trainer, eval_dataset)
# prompt_tuning_results = evaluate_model(prompt_trainer, eval_dataset)
# instruction_tuning_results = evaluate_model(instruct_trainer, eval_dataset)

print("性能评估完成")
```

除了性能外，训练效率和资源消耗也是选择微调方法的重要考量因素：

```python
import time
import psutil

def measure_training_efficiency(trainer, model_name):
    # 测量训练时间和内存使用
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # 实际训练代码会在这里执行
    # trainer.train()
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    return {
        "training_time": end_time - start_time,
        "memory_used": end_memory - start_memory,
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in trainer.model.parameters())
    }

# 测量各方法的效率
# full_finetune_efficiency = measure_training_efficiency(trainer, "Full Finetune")
# lora_efficiency = measure_training_efficiency(lora_trainer, "LoRA")
# prompt_tuning_efficiency = measure_training_efficiency(prompt_trainer, "Prompt Tuning")
# instruction_tuning_efficiency = measure_training_efficiency(instruct_trainer, "Instruction Tuning")

print("效率评估完成")
```

根据理论分析和实验经验，我们可以总结出数据集特性与微调技术的适配关系：

| **微调方法** | **数据需求** | **计算效率** | **适合场景** | **实现难度** |
|------------|------------|------------|------------|------------|
| **全参数微调** | 大量高质量数据 | 低 | 数据充足且与预训练数据相似度高 | 中等 |
| **LoRA** | 中等规模数据 | 高 | 计算资源有限，需要快速适配 | 低 |
| **Prompt Tuning** | 少样本学习 | 极高 | 数据稀缺，需要快速部署 | 低 |
| **指令微调** | 高质量指令-回答对 | 中等 | 任务特定格式和指令遵循 | 中等 |

具体来说：

1.  **数据量少，数据相似度高**：适合 Prompt Tuning 或 LoRA，只需要修改最后几层或添加少量参数。

2.  **数据量少，数据相似度低**：适合 LoRA 或 Adapter 方法，可以冻结预训练模型的初始层，只训练较高层。

3.  **数据量大，数据相似度低**：考虑全参数微调或领域自适应预训练（DAPT），但由于数据差异大，可能需要更多训练时间。

4.  **数据量大，数据相似度高**：全参数微调通常能获得最佳性能，这是最理想的情况。

## 8. 总结与思考

在实际应用中，选择微调技术时需要综合考虑数据特性（数量、质量、与预训练数据的相似度）、计算资源约束、任务要求和部署环境等因素。对于大多数实际应用场景，**LoRA**提供了最佳的权衡，而**Prompt Tuning**则在极端资源约束或数据稀缺环境下更具优势。

未来的研究方向可能包括这些技术的组合使用（如 LoRA+Prompt Tuning）、自适应微调策略（根据数据特性动态选择微调方法）以及更高效的参数高效微调技术。

## 参考文献

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
2. Liu, X., et al. (2019). Multi-Task Deep Neural Networks for Natural Language Understanding. arXiv:1901.11504.
3. Shin, T., et al. (2020). AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. arXiv:2010.15980.
