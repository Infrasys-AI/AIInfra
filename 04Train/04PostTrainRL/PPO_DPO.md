##  PPO（Proximal Policy Optimization）

PPO（Proximal Policy Optimization，近端策略优化）是一种强化学习算法，它通过优化策略来最大化奖励，适用于连续和离散动作空间。


在强化学习中策略优化是核心问题之一。传统的策略优化方法可能会导致策略更新过大，从而破坏学习过程的稳定性。PPO 通过限制策略更新的幅度，确保学习过程更加平稳，同时保持较高的样本效率。


PPO 的核心思想是通过限制策略更新的幅度来实现稳定的优化。它引入了一个重要的概念——**剪辑（Clipping）**，用于约束策略变化范围。具体来说，PPO 优化目标函数如下：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t) \right]
$$

其中：
- r(θ)是新旧策略的概率比值。
- A_t 是优势函数，用于衡量当前动作的好坏。
- ϵ 是剪辑范围的超参数。


在 RLHF-PPO 阶段，一共有四个主要模型

- Actor Model：演员模型，这就是我们想要训练的目标语言模型

- Critic Model：评论家模型，它的作用是预估总收益 

- Reward Model：奖励模型，它的作用是计算即时收益 

- Reference Model：参考模型，它的作用是在 RLHF 阶段给语言模型增加一些“约束”，防止语言模型训歪（朝不受控制的方向更新，效果可能越来越差）

在上面的四个模型中 Actor/Critic Model 在 RLHF 阶段是需要训练的，而 Reward/Reference Model 是参数冻结的。


## DPO (Direct Preference Optimization)

**核心思想**：  
DPO 是一种**无需显式奖励模型**的强化学习对齐方法，直接利用人类偏好数据优化策略模型（如语言模型）。它通过数学变换将强化学习目标转化为简单的分类损失函数，避免了 PPO 的复杂流程。

#### 关键动机
- 传统 RLHF 依赖两阶段流程：先训练奖励模型（RM），再用 PPO 优化策略模型，过程复杂且不稳定
- DPO 发现：最优策略与奖励模型存在**解析关系**，可直接用偏好数据训练策略模型，跳过 RM 训练和 RL 优化

### DPO 工作原理
#### 1. 理论基础
- **偏好建模**：基于 Bradley-Terry 模型，偏好概率表示为：
  $$
  P(y_w \succ y_l | x) = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}
  $$
  其中 $y_w$ 是优选响应，$y_l$ 是劣选响应
  
- **策略与奖励的关联**：推导出最优策略 $\pi^*$ 与奖励函数 $r^*$ 的关系：
  $$
  r^*(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \text{const}
  $$
  $\beta$ 控制策略偏离参考模型 $\pi_{\text{ref}}$ 的程度

#### 2. 损失函数
将奖励函数代入偏好概率，得到 DPO 损失函数：
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ 
\log \sigma \left( 
\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} 
- \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} 
\right) \right]
$$
其中：
- $\sigma$ 是 sigmoid 函数
- $\mathcal{D}$ 是偏好数据集（三元组 $(x, y_w, y_l)$）
- $\pi_\theta$ 是待优化的策略模型

#### 3. 训练流程
1. **数据准备**：收集人类偏好三元组（提示 $x$，优选响应 $y_w$，劣选响应 $y_l$）
2. **参考模型固定**：使用初始 SFT 模型作为 $\pi_{\text{ref}}$（不更新参数）
3. **直接优化策略**：通过最小化 $\mathcal{L}_{\text{DPO}}$ 更新 $\pi_\theta$

## DPO vs PPO 关键对比
| **特性**         | **PPO**                          | **DPO**                          |
|------------------|----------------------------------|----------------------------------|
| **训练流程**     | 两阶段：先训练 RM，再 RL 微调    | 单阶段：直接用偏好数据优化策略   |
| **模型复杂度**   | 需维护 4 个模型                 | 仅需 2 个模型（策略 + 参考模型） |
| **稳定性**       | 依赖 RL 稳定性，易出现发散       | 类似监督学习，训练更稳定         |
| **计算开销**     | 高（需在线采样和多次策略更新）   | 低（仅离线批量训练）             |
| **奖励建模**     | 需显式奖励模型                   | 隐式通过策略与参考模型差异表示   |
| **超参数调优**   | 复杂（clip range, KL 系数等）    | 简单（主要调 $\beta$）           |




