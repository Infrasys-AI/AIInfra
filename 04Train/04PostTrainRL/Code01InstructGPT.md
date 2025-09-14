<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 01: 经典 InstructGPT 复现

本实验将使用简化代码复现 InstructGPT 的核心训练流程：监督微调(SFT)→奖励模型训练(RM)→PPO 强化学习优化。通过这个实验，我们可以深入理解如何通过人类反馈强化学习将人类偏好传递给语言模型。

InstructGPT 是 OpenAI 提出的基于人类反馈的强化学习(RLHF)框架，它通过三个阶段训练语言模型遵循人类指令。虽然原始实现需要大量计算资源，但我们将使用简化模型和小型数据集来演示核心概念。

## 1. 环境设置

首先安装必要的依赖库：

```python
!pip install torch numpy tqdm matplotlib
```

现在导入实验所需的模块：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

## 2. Stage1：监督微调 SFT

监督微调阶段使用人类标注的指令-回答对来微调预训练语言模型，使其初步学会遵循指令。

在 SFT 阶段，我们使用标准的监督学习方式，使用人类标注的高质量问答对数据来微调预训练语言模型。模型的训练目标是最大化在给定指令情况下，生成人类期望回答的似然概率。

数学上，这个目标可以表示为：

$$ \mathcal{L}_{SFT} = -\mathbb{E}_{(x,y)\sim D}[\log P_{\theta}(y|x)] $$

其中 $x$ 是指令，$y$ 是人类标注的回答，$\theta$ 是模型参数。

### 2.1 数据准备

我们创建一个简化的指令-回答数据集，并实现一个简单的文本编码器：

```python
# 简化的文本编码器
class SimpleTokenizer:
    def __init__(self):
        # 仅保留常见字符+特殊符号（简化版词汇表）
        self.chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ，。？！：；,.?!:;'
        self.char2id = {c: i+1 for i, c in enumerate(self.chars)}  # 1 开始，0 留作 padding
        self.char2id['<pad>'] = 0
        self.vocab_size = len(self.char2id)
    
    def encode(self, text, max_len=32):
        # 文本→token id（截断或补全）
        tokens = [self.char2id.get(c, 0) for c in text[:max_len]]  # 未知字符→0
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))  # 补全
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        # token id→文本
        return ''.join([self.chars[i-1] if i>0 else '' for i in tokens])

# 创建 SFT 数据集类
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instr, resp = self.data[idx]
        text = f"指令：{instr} 回答：{resp}"
        tokens = self.tokenizer.encode(text)
        # 输入是前 n-1 个 token，目标是后 n-1 个 token
        return {
            'input_ids': tokens[:-1],  # [seq_len-1]
            'labels': tokens[1:]       # [seq_len-1]
        }

# 准备 SFT 数据
def create_sft_dataset():
    # 简单的指令-回答对
    return [
        ("什么是 AI？", "AI 是能模拟人类智能的技术。"),
        ("1+1 等于几？", "1+1 等于 2。"),
        ("写一句问候语", "你好，很高兴见到你！"),
        ("什么是机器学习？", "机器学习是让计算机从数据中学习的技术。"),
        ("如何保护环境？", "减少浪费，节约能源，绿色出行。")
    ]

# 初始化 tokenizer 和数据集
tokenizer = SimpleTokenizer()
sft_data = create_sft_dataset()
sft_dataset = SFTDataset(sft_data, tokenizer)
sft_dataloader = DataLoader(sft_dataset, batch_size=2, shuffle=True)
```

这里创建了一个简单的字符级编码器`SimpleTokenizer`，将文本转换为模型可以处理的数字序列。SFTDataset 类将指令和回答组合成模型输入，并构建了合理的训练目标——用前 n-1 个 token 预测第 n 个 token，这是语言模型训练的标准做法。

### 2.2 模型定义

定义一个简化的 Transformer 模型，包含必要的位置编码：

```python
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = self._simple_pos_enc(d_model, max_len=32)  # 简化位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def _simple_pos_enc(self, d_model, max_len):
        # 简化版位置编码：使用正弦函数
        pos = torch.arange(max_len).unsqueeze(1)
        enc = torch.sin(pos / (10000 ** (torch.arange(d_model)/d_model)))
        return enc  # [max_len, d_model]
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        x_emb = self.embedding(x)  # [batch_size, seq_len, d_model]
        x_emb = x_emb + self.pos_enc[:seq_len, :].to(device)  # 添加位置编码
        output = self.transformer(x_emb)  # [batch_size, seq_len, d_model]
        return self.fc_out(output)  # [batch_size, seq_len, vocab_size]

model = SimpleTransformer(vocab_size=tokenizer.vocab_size).to(device)
```

简化 Transformer 模型包含嵌入层、位置编码和 Transformer 编码器。位置编码是关键补充，它让模型能够理解文本中的顺序信息，这是原代码缺失的重要组件。

### 2.3 SFT 训练

```python
# SFT 训练配置
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding 的损失

# SFT 训练循环
def train_sft(model, dataloader, epochs=3):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 准备输入和目标
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(input_ids)  # [batch_size, seq_len-1, vocab_size]
            
            # 计算损失
            loss = criterion(outputs.transpose(1, 2), labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} 完成。平均损失: {avg_loss:.4f}")
    
    return losses

# 开始 SFT 训练
print("开始 SFT 训练阶段...")
sft_losses = train_sft(model, sft_dataloader, epochs=3)
```

训练过程中，我们使用交叉熵损失函数，目标是让模型学会预测下一个 token。注意我们忽略了 padding token(0)的损失，这是处理变长序列的标准做法。随着训练进行，损失应该逐步下降，表明模型正在学习指令和回答之间的关系。

## 3. Stage2：奖励模型 RM

奖励模型用于学习人类对模型回答的偏好，为强化学习阶段提供奖励信号。

奖励模型的目标是学习人类的偏好判断，即对于同一个指令的多个回答，哪个回答更符合人类偏好。训练数据由人类标注员对模型生成回答的质量进行排名。

奖励模型的训练通常使用对比学习框架，通过比较不同回答的相对质量来学习一个标量奖励函数。常用的损失函数是 pairwise ranking loss：

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)\sim D}[\log(\sigma(r_{\phi}(x,y_w) - r_{\phi}(x,y_l)))]
$$

其中 $y_w$ 是偏好的回答，$y_l$ 是不偏好的回答，$r_{\phi}$ 是奖励模型。

### 3.1 数据准备

```python
# 创建奖励模型数据集类
class RMDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        good_text = f"指令：{item['instr']} 回答：{item['good']}"
        bad_text = f"指令：{item['instr']} 回答：{item['bad']}"
        return {
            'good_input_ids': self.tokenizer.encode(good_text),
            'bad_input_ids': self.tokenizer.encode(bad_text)
        }

# 创建奖励模型训练数据
def create_rm_dataset():
    # 模拟人类对回答的偏好比较
    return [
        {
            "instr": "什么是 AI？",
            "good": "AI 是能模拟人类智能的技术，比如语音识别和图像识别。",
            "bad": "AI 就是机器人。"
        },
        {
            "instr": "1+1 等于几？",
            "good": "1+1 的结果是 2。",
            "bad": "1+1 可能等于 3，看情况。"
        },
        {
            "instr": "写一句问候语",
            "good": "你好！很高兴有机会与你交流。",
            "bad": "喂，你好。"
        },
        {
            "instr": "什么是机器学习？",
            "good": "机器学习是人工智能的一个分支，让计算机能从数据中学习并改进。",
            "bad": "机器学习就是教机器读书。"
        }
    ]

# 准备 RM 数据
rm_data = create_rm_dataset()
rm_dataset = RMDataset(rm_data, tokenizer)
rm_dataloader = DataLoader(rm_dataset, batch_size=1, shuffle=True)
```

奖励模型需要成对的偏好数据，每个数据点包含一个指令、一个高质量回答和一个低质量回答。这种数据结构允许我们训练模型区分回答的质量差异。

### 3.2 奖励模型

```python
# 定义奖励模型
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 复用 SFT 阶段的基础模型
        self.reward_head = nn.Linear(64, 1)  # 奖励头，将特征映射为标量奖励
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x_emb = self.base_model.embedding(x)  # [batch_size, seq_len, d_model]
        x_emb = x_emb + self.base_model.pos_enc[:x.shape[1], :].to(device)  # 添加位置编码
        
        # 获取基础模型的特征输出
        features = self.base_model.transformer(x_emb)  # [batch_size, seq_len, d_model]
        
        # 使用最后一个 token 的特征计算奖励
        last_token_features = features[:, -1, :]  # [batch_size, d_model]
        reward = self.reward_head(last_token_features)  # [batch_size, 1]
        return reward

# 初始化奖励模型（基于 SFT 模型构建）
reward_model = RewardModel(model).to(device)
```

奖励模型基于第一阶段训练的 SFT 模型构建，这样可以利用已经学到的指令理解能力。我们添加了一个简单的线性层作为"奖励头"，将模型最后一个 token 的特征转换为一个标量奖励值。

### 3.3 RM 训练

```python
# 奖励模型训练
def train_rm(reward_model, dataloader, epochs=2):
    reward_model.train()
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"RM Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 准备好回答和差回答的输入
            good_inputs = batch['good_input_ids'].to(device)
            bad_inputs = batch['bad_input_ids'].to(device)
            
            # 计算奖励
            good_reward = reward_model(good_inputs)
            bad_reward = reward_model(bad_inputs)
            
            # 计算对比损失：好回答的奖励应高于差回答
            loss = -torch.log(torch.sigmoid(good_reward - bad_reward)).mean()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"RM Epoch {epoch+1} 完成。平均损失: {avg_loss:.4f}")

# 开始奖励模型训练
print("开始奖励模型训练阶段...")
train_rm(reward_model, rm_dataloader, epochs=2)
```

奖励模型的训练使用对比损失函数，目标是让模型对高质量回答给出更高的奖励值，对低质量回答给出更低的奖励值。这种成对比较的方式能够有效学习人类的偏好。随着训练进行，损失应该下降，表明模型逐渐学会了区分回答质量。

## 4. Stage3：PPO 强化学习

在 PPO 阶段，我们使用奖励模型提供的奖励信号来进一步优化语言模型，使其生成更符合人类偏好的回答。

### 4.1 技术原理

PPO(Proximal Policy Optimization)是一种强化学习算法，它通过限制策略更新的步长来确保训练稳定性。在 RLHF 中，PPO 用于优化语言模型，使其生成的回答能获得更高的奖励模型评分。

PPO 的核心目标函数如下：

$$ \mathcal{L}^{PPO} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] $$

其中 $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是概率比，$\hat{A}_t$ 是优势估计（这里简化为奖励模型的输出），$\epsilon$ 是裁剪参数。

### 4.2 PPO 实现

```python
def ppo_train(model, reward_model, tokenizer, epochs=2):
    # 冻结奖励模型的参数
    for param in reward_model.parameters():
        param.requires_grad = False
        
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # 简化 PPO 超参数
    clip_epsilon = 0.2  # 裁剪参数
    instructions = ["什么是 AI？", "1+1 等于几？", "写一句问候语"]  # 用于 PPO 训练的指令
    
    for epoch in range(epochs):
        total_reward = 0
        progress_bar = tqdm(instructions, desc=f"PPO Epoch {epoch+1}/{epochs}")
        
        for instruction in progress_bar:
            # 准备输入：指令 + "回答："前缀
            input_text = f"指令：{instruction} 回答："
            input_ids = tokenizer.encode(input_text)[:-1].unsqueeze(0).to(device)  # [1, seq_len]
            
            # 模型生成回答
            generated_ids = input_ids.clone()
            for _ in range(10):  # 生成 10 个 token
                logits = model(generated_ids)[:, -1, :]  # 获取最后一个 token 的 logits
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # 贪心选择
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # 计算当前生成的奖励
            with torch.no_grad():
                reward = reward_model(generated_ids).item()
            total_reward += reward
            
            # 简化 PPO 损失计算
            # 1. 计算旧策略的概率
            with torch.no_grad():
                old_logits = model(input_ids)
                old_probs = torch.softmax(old_logits, dim=-1)
            
            # 2. 计算新策略的概率
            new_logits = model(input_ids)
            new_probs = torch.softmax(new_logits, dim=-1)
            
            # 3. 计算策略比率 r(θ) = new_probs / old_probs
            ratio = (new_probs / (old_probs + 1e-10)).mean(dim=-1)  # 简化：取平均值
            
            # 4. 计算裁剪和未裁剪的优势
            surr1 = ratio * reward
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * reward
            
            # 5. PPO 损失：取两者的最小值并取负（因为我们要最大化奖励）
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 优化步骤
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"奖励": reward})
        
        avg_reward = total_reward / len(instructions)
        print(f"PPO Epoch {epoch+1} 完成。平均奖励: {avg_reward:.4f}")

# 开始 PPO 训练
print("开始 PPO 训练阶段...")
ppo_train(model, reward_model, tokenizer, epochs=2)
```

PPO 训练的核心是通过奖励模型提供的反馈来优化语言模型。我们首先冻结奖励模型参数，确保奖励信号稳定。然后，对于每个指令，模型生成回答并获得奖励。PPO 的关键是使用裁剪机制限制策略更新的幅度，防止更新过大导致训练不稳定。随着训练进行，模型生成的回答应该能获得越来越高的奖励。

## 5. 实验结果与分析

通过上述三个阶段的训练，我们完成了简化版的 InstructGPT 训练流程。虽然实现经过了大幅简化，但核心思想得以保留：

1. **SFT 阶段**：模型学会了基本的指令跟随能力
2. **RM 阶段**：奖励模型学会了区分回答质量
3. **PPO 阶段**：语言模型根据奖励信号进一步优化生成策略

### 5.1 训练过程可视化

```python
# 绘制训练损失曲线
plt.figure(figsize=(10, 5))

# SFT 损失曲线
plt.plot(range(1, len(sft_losses)+1), sft_losses, marker='o', color='blue')
plt.title('SFT 训练损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

**预期结果**：SFT 阶段的损失应该呈现稳步下降趋势，表明模型正在学习指令与回答之间的关系。RM 阶段的损失也应该下降，表明奖励模型学会了区分好回答和差回答。PPO 阶段的平均奖励应该上升，表明模型生成的回答越来越符合奖励模型定义的"好"标准。

### 5.2 模型效果对比

我们可以对比 SFT 阶段和 PPO 阶段模型的生成效果：

```python
# 测试函数：生成回答并显示
def generate_response(model, tokenizer, instruction, max_tokens=15):
    model.eval()
    input_text = f"指令：{instruction} 回答："
    input_ids = tokenizer.encode(input_text)[:-1].unsqueeze(0).to(device)
    
    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(generated_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids.squeeze().cpu().numpy())
    # 提取回答部分
    answer = generated_text.split("回答：")[-1].strip()
    return answer

# 测试几个指令
test_instructions = [
    "什么是 AI？",
    "1+1 等于几？",
    "写一句问候语"
]

print("模型生成结果：")
for instr in test_instructions:
    response = generate_response(model, tokenizer, instr)
    print(f"指令：{instr}")
    print(f"回答：{response}\n")
```

经过 PPO 优化的模型生成的回答应该比仅经过 SFT 的模型更符合人类偏好。

## 6. 总结与思考

本实验通过简化实现复现了 InstructGPT 的 RLHF 三阶段流程。虽然实际应用中的实现更加复杂，但核心思想是一致的：

1. **监督微调**为模型提供基础的指令跟随能力
2. **奖励模型**学习人类偏好并提供量化反馈
3. **PPO 强化学习**利用奖励信号进一步优化模型

这种方法的优势在于能够将人类的主观偏好有效地传递给模型，使模型生成更加符合人类期望的内容。
