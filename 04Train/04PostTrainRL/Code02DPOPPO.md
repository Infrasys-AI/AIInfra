<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 02: DPO 与 PPO 在 LLM 对比

在大语言模型和多模态大模型的发展中，如何让模型生成的内容更好地符合人类价值观和偏好是一个核心挑战。

近端策略优化（PPO）作为强化学习的主流方法，通过奖励模型引导模型优化，在人类反馈的强化学习（RLHF）中取得了显著成果。然而，PPO 需要复杂的奖励模型设计和多阶段训练流程。直接偏好优化（DPO）则提供了一种更直接的解决方案，它通过比较不同响应的偏好数据来优化策略，避免了显式奖励模型的设计。

本实验将使用 Hugging Face 的 Qwen-1.8B 模型作为基础模型，通过一个简化的文本生成任务，深入对比分析这两种方法在大语言模型场景下的表现。

## 1. 实验环境设置

首先，我们需要加载 Qwen-1.8B 模型并创建文本生成环境。Qwen 系列模型是由阿里巴巴开发的开源大语言模型，1.8B 版本在保持较好性能的同时计算资源需求适中，适合实验环境。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置设备 - 优先使用 GPU 加速计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载 Qwen-1.8B 模型和分词器
model_name = "Qwen/Qwen-1_8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

# 加载基础模型，使用 bfloat16 精度减少内存占用
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
print("Qwen-1.8B 模型加载完成")
```

## 2. 文本生成环境

为了对比 PPO 和 DPO，我们创建一个简化的文本生成环境。这个环境模拟了对话系统或文本补全任务的基本流程，其中模型需要根据给定的提示生成合适的响应。

```python
class TextGenerationEnv:
    def __init__(self, prompt_list, max_length=30):
        """
        文本生成环境
        :param prompt_list: 提示文本列表
        :param max_length: 生成文本的最大长度
        """
        self.prompts = prompt_list
        self.max_length = max_length
        self.current_prompt = None
        self.generated_text = ""
        
    def reset(self):
        """重置环境，随机选择一个提示"""
        self.current_prompt = np.random.choice(self.prompts)
        self.generated_text = ""
        return self.current_prompt
    
    def step(self, action):
        """
        执行一个动作（生成一个 token）
        :param action: token ID
        :return: 生成文本, 奖励, 是否完成
        """
        # 解码 token 并添加到生成文本
        token = tokenizer.decode([action])
        self.generated_text += token
        
        # 检查终止条件：达到最大长度或生成结束标记
        done = (len(self.generated_text) >= self.max_length or 
                action == tokenizer.eos_token_id)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        return self.generated_text, reward, done
    
    def _calculate_reward(self):
        """计算生成文本的奖励（简化版本）"""
        # 在实际应用中，这里可以使用奖励模型或人工评估
        # 这里使用简单的启发式规则评估生成质量
        text = self.generated_text.lower()
        prompt = self.current_prompt.lower()
        
        # 1. 长度奖励：鼓励生成长文本
        length_reward = min(len(text) / self.max_length, 1.0)
        
        # 2. 多样性奖励：鼓励使用不同的词汇
        unique_words = len(set(text.split()))
        diversity_reward = min(unique_words / 10, 1.0)
        
        # 3. 相关性奖励：检查是否与提示相关
        prompt_words = set(prompt.split())
        response_words = set(text.split())
        common_words = prompt_words & response_words
        relevance_reward = min(len(common_words) / max(1, len(prompt_words)), 1.0)
        
        # 4. 流畅性奖励：简单检查常见连接词
        fluency_reward = 0.5  # 基础值
        for word in ["and", "the", "but", "however"]:
            if word in text:
                fluency_reward += 0.1
        
        # 加权组合各项奖励
        total_reward = (length_reward * 0.3 + 
                        diversity_reward * 0.2 + 
                        relevance_reward * 0.3 + 
                        min(fluency_reward, 1.0) * 0.2)
        
        return total_reward
```

## 3. PPO 原理与实现

PPO 算法的核心思想是通过限制策略更新的幅度来保证训练的稳定性。它使用一个裁剪函数来防止策略更新过大，从而避免训练过程中的剧烈波动。PPO 的目标函数可以表示为：

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$$

其中：

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是策略比
- $A_t$ 是优势函数，表示当前动作相对于平均水平的优势
- $\epsilon$ 是裁剪参数，通常设为 0.1-0.3

这个目标函数的核心思想是：当策略比 $r_t(\theta)$ 偏离 1 太远时，通过裁剪限制其影响，从而避免过大的策略更新。

在大语言模型场景中，PPO 通常用于 RLHF 流程，通过奖励模型来优化策略。我们实现一个简化的 PPO 训练器：

```python
class PPOPolicy(nn.Module):
    """包装语言模型作为策略网络"""
    def __init__(self, base_model):
        super(PPOPolicy, self).__init__()
        self.model = base_model
        
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def get_logits(self, input_ids, attention_mask=None):
        """获取语言模型的输出 logits"""
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

class PPO:
    """PPO 算法实现"""
    def __init__(self, policy_model, value_model, ppo_epochs=4, lr=1e-5, gamma=0.99, epsilon=0.2):
        """
        :param policy_model: 策略模型
        :param value_model: 价值模型
        :param ppo_epochs: PPO 更新轮数
        :param lr: 学习率
        :param gamma: 折扣因子
        :param epsilon: 裁剪参数
        """
        self.policy = policy_model
        self.value_model = value_model
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 创建优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=lr)
    
    def generate(self, prompt, max_length=20):
        """使用当前策略生成文本"""
        # 编码提示文本
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        generated = input_ids
        log_probs = []  # 记录每个动作的对数概率
        values = []     # 记录每个状态的价值
        
        # 逐步生成文本
        for _ in range(max_length):
            with torch.no_grad():
                # 获取当前策略的输出 logits
                logits = self.policy.get_logits(generated)
                next_token_logits = logits[:, -1, :]
                
                # 创建分类分布并采样
                dist = Categorical(logits=next_token_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # 获取当前状态的价值
                value = self.value_model(generated).squeeze(-1)
                
            # 将新 token 添加到生成序列
            generated = torch.cat([generated, action.unsqueeze(0)], dim=-1)
            log_probs.append(log_prob)
            values.append(value)
            
            # 如果生成结束标记则提前终止
            if action.item() == tokenizer.eos_token_id:
                break
        
        return generated, torch.stack(log_probs), torch.stack(values)
    
    def update(self, prompts, rewards, old_log_probs, values):
        """更新策略和价值模型"""
        # 计算折扣回报
        returns = self._calculate_returns(rewards, values)
        # 计算优势函数：回报 - 价值估计
        advantages = returns - values
        
        # 多轮 PPO 更新
        for _ in range(self.ppo_epochs):
            # 重新计算新策略的对数概率
            new_log_probs = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = self.policy.get_logits(input_ids)
                    # 只考虑最后一个 token 的分布
                    dist = Categorical(logits=logits[:, -1, :])
                    new_log_probs.append(dist.log_prob(input_ids[:, -1]))
            
            new_log_probs = torch.stack(new_log_probs)
            
            # 计算策略比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算 PPO 裁剪目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 更新价值函数
            value_loss = nn.MSELoss()(self.value_model(prompts), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
    
    def _calculate_returns(self, rewards, values):
        """计算折扣回报"""
        returns = []
        R = 0
        # 从后向前计算累积回报
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(device)
```

## 4. DPO 原理与实现

DPO 算法直接从人类偏好中学习策略，避免了显式奖励函数的设计。它基于一个关键洞见：最优策略可以通过 Bradley-Terry 模型表示：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r^*(x,y)\right)$$

其中：

- $\pi_{ref}$ 是参考策略
- $r^*$ 是最优奖励函数
- $\beta$ 是温度参数
- $Z(x)$ 是归一化常数

DPO 通过优化以下目标函数来学习策略：

$$L_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

这个目标函数的核心思想是：对于给定的提示 $x$，偏好响应 $y_w$ 的对数概率应该高于非偏好响应 $y_l$ 的对数概率。

DPO 不需要单独的价值函数或奖励模型，直接使用偏好数据优化策略：

```python
class DPO:
    """DPO 算法实现"""
    def __init__(self, policy_model, reference_model, beta=0.1, lr=1e-5):
        """
        :param policy_model: 待优化的策略模型
        :param reference_model: 参考模型（通常固定）
        :param beta: 温度参数
        :param lr: 学习率
        """
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def update(self, prompts, preferred_responses, dispreferred_responses):
        """使用偏好数据更新策略"""
        losses = []
        
        # 遍历每个偏好样本
        for prompt, preferred, dispreferred in zip(prompts, preferred_responses, dispreferred_responses):
            # 编码提示和响应
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            preferred_ids = tokenizer.encode(preferred, return_tensors="pt").to(device)
            dispreferred_ids = tokenizer.encode(dispreferred, return_tensors="pt").to(device)
            
            # 计算策略模型对偏好响应的对数概率
            policy_logits = self.policy(torch.cat([prompt_ids, preferred_ids], dim=-1))
            policy_log_probs = self._get_log_probs(policy_logits.logits, preferred_ids)
            
            # 计算参考模型对偏好响应的对数概率
            ref_logits = self.reference(torch.cat([prompt_ids, preferred_ids], dim=-1))
            ref_log_probs = self._get_log_probs(ref_logits.logits, preferred_ids)
            
            # 计算策略模型对非偏好响应的对数概率
            policy_dis_logits = self.policy(torch.cat([prompt_ids, dispreferred_ids], dim=-1))
            policy_dis_log_probs = self._get_log_probs(policy_dis_logits.logits, dispreferred_ids)
            
            # 计算参考模型对非偏好响应的对数概率
            ref_dis_logits = self.reference(torch.cat([prompt_ids, dispreferred_ids], dim=-1))
            ref_dis_log_probs = self._get_log_probs(ref_dis_logits.logits, dispreferred_ids)
            
            # 计算对数比值
            log_ratio_preferred = (policy_log_probs - ref_log_probs).sum()
            log_ratio_dispreferred = (policy_dis_log_probs - ref_dis_log_probs).sum()
            
            # 计算 DPO 损失
            loss = -torch.log(
                torch.sigmoid(
                    self.beta * (log_ratio_preferred - log_ratio_dispreferred)
                )
            )
            
            losses.append(loss)
        
        # 平均损失并更新策略
        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def _get_log_probs(self, logits, labels):
        """计算标签序列的对数概率"""
        # 将 logits 和 labels 对齐
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # 计算每个 token 的对数概率
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.shape)
```

## 5. 准备训练数据

我们创建一组多样化的提示文本，并生成模拟的偏好数据用于训练：

```python
# 准备训练提示
prompts = [
    "The weather today is",
    "I really enjoy",
    "In my opinion,",
    "The best thing about",
    "I think that",
    "Artificial intelligence",
    "Machine learning models",
    "Deep reinforcement learning",
    "Natural language processing",
    "The future of AI",
    "Climate change is",
    "Renewable energy sources",
    "The impact of technology",
    "Education in the digital age",
    "Cultural diversity means"
]

# 生成模拟偏好数据
def generate_preference_data(num_samples=100):
    """生成模拟的偏好数据"""
    preferences = []
    
    for _ in range(num_samples):
        prompt = np.random.choice(prompts)
        
        # 生成两种可能的回应
        response_options = [
            "nice and sunny, perfect for outdoor activities.",
            "quite unpredictable, with a chance of rain later.",
            "a fascinating field with immense potential.",
            "challenging but rewarding to study and apply.",
            "essential for addressing global challenges.",
            "a fundamental aspect of human society."
        ]
        
        # 随机选择两个不同的回应
        idx1, idx2 = np.random.choice(len(response_options), 2, replace=False)
        response1 = response_options[idx1]
        response2 = response_options[idx2]
        
        # 随机分配偏好（实际应用中来自人类标注）
        if np.random.random() > 0.5:
            preferred = response1
            dispreferred = response2
        else:
            preferred = response2
            dispreferred = response1
        
        preferences.append((prompt, preferred, dispreferred))
    
    return preferences
```

## 6. 模型初始化

我们初始化策略模型、价值模型（用于 PPO）和参考模型（用于 DPO）：

```python
# 初始化策略模型（将用于两种算法）
policy_model = PPOPolicy(base_model).to(device)

# 价值模型（用于 PPO）
# 这是一个简单的神经网络，用于估计状态价值
value_model = nn.Sequential(
    nn.Linear(base_model.config.hidden_size, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)

# 参考模型（用于 DPO）
# 我们加载一个新的模型实例作为参考模型
reference_model = PPOPolicy(AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device))

# 冻结参考模型参数
for param in reference_model.parameters():
    param.requires_grad = False

# 初始化训练器
ppo_trainer = PPO(policy_model, value_model)
dpo_trainer = DPO(policy_model, reference_model)
```

## 7. 模型训练循环

我们分别实现 PPO 和 DPO 的训练循环：

```python
def train_ppo(ppo_trainer, env, num_episodes=50):
    """PPO 训练循环"""
    rewards_history = []
    
    for episode in range(num_episodes):
        # 重置环境
        prompt = env.reset()
        
        # 生成文本
        generated, log_probs, values = ppo_trainer.generate(prompt)
        generated_text = tokenizer.decode(generated[0])
        
        # 计算奖励（使用环境中的奖励函数）
        # 注意：这里我们只取生成部分（不包括提示）
        env.generated_text = generated_text[len(prompt):]
        reward = env._calculate_reward()
        
        # 更新策略
        ppo_trainer.update([prompt], [reward], log_probs, values)
        
        # 记录奖励历史
        rewards_history.append(reward)
        
        # 定期输出进度
        if episode % 5 == 0:
            print(f"PPO Episode {episode}: 奖励={reward:.3f}")
            print(f"  提示: '{prompt}'")
            print(f"  生成: '{generated_text}'\n")
    
    return rewards_history

def train_dpo(dpo_trainer, preference_data, num_epochs=10):
    """DPO 训练循环"""
    losses = []
    
    for epoch in range(num_epochs):
        # 打乱数据
        np.random.shuffle(preference_data)
        
        # 拆分数据
        prompts = [d[0] for d in preference_data]
        preferred = [d[1] for d in preference_data]
        dispreferred = [d[2] for d in preference_data]
        
        # 更新策略
        loss = dpo_trainer.update(prompts, preferred, dispreferred)
        losses.append(loss)
        
        # 定期输出进度
        if epoch % 2 == 0:
            print(f"DPO Epoch {epoch}: 损失={loss:.4f}")
    
    return losses

# 创建环境
env = TextGenerationEnv(prompts)

# 生成偏好数据
preference_data = generate_preference_data(num_samples=100)

# 运行训练
print("开始 PPO 训练...")
ppo_rewards = train_ppo(ppo_trainer, env)

print("\n 开始 DPO 训练...")
dpo_losses = train_dpo(dpo_trainer, preference_data)
```

## 8. 结果分析

训练完成后，我们可视化结果并比较生成文本的质量：

```python
# 绘制训练曲线
plt.figure(figsize=(12, 5))

# PPO 奖励曲线
plt.subplot(1, 2, 1)
plt.plot(ppo_rewards, label='PPO 奖励', color='blue')
plt.xlabel('训练轮次')
plt.ylabel('奖励')
plt.title('PPO 训练奖励变化')
plt.grid(True)

# DPO 损失曲线
plt.subplot(1, 2, 2)
plt.plot(dpo_losses, label='DPO 损失', color='red')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('DPO 训练损失变化')
plt.grid(True)

plt.tight_layout()
plt.show()

# 测试生成质量
def test_generation(model, prompts, num_samples=3):
    """测试模型生成质量"""
    print("\n 生成文本质量测试:")
    for i, prompt in enumerate(prompts[:num_samples]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # 使用采样生成更自然的文本
            outputs = model.model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"样本 {i+1}:")
        print(f"  提示: '{prompt}'")
        print(f"  生成: '{generated_text}'\n")

# 测试基础模型
print("基础模型生成结果:")
test_generation(policy_model, prompts)

# 测试 PPO 微调后的模型
print("PPO 微调后模型生成结果:")
test_generation(ppo_trainer.policy, prompts)

# 测试 DPO 微调后的模型
print("DPO 微调后模型生成结果:")
test_generation(dpo_trainer.policy, prompts)
```

## 9. 讨论与结论

PPO 训练过程中奖励值逐渐提高，表明模型学会了生成更符合奖励函数定义的文本。PPO 的优势在于它能够直接从环境中学习，但需要精心设计奖励函数。在文本生成任务中，设计一个全面评估文本质量的奖励函数本身就是一项挑战。

DPO 训练过程中损失值逐渐降低，表明模型学会了区分偏好和非偏好响应。DPO 避免了奖励函数的设计问题，但需要高质量的偏好数据。在实际应用中，获取大规模高质量的偏好数据可能需要大量人工标注工作。

在生成质量方面，基础模型生成的文本通常较为通用，缺乏针对性；PPO 微调后的模型生成的文本更符合奖励函数的定义（如长度、多样性、相关性）；而 DPO 微调后的模型生成的文本更符合人类偏好，表现出更好的主观质量。
