# 高效 Transformer 研究：Linear Attention、Attention Free Transformer 与 Longformer 的结构原理及应用

Author：张嘉瑶

## 1. 引言：追求 Transformer 的效率

### 1.1. 标准 Transformer：回顾其强大功能与固有缺陷

Transformer 模型作为一种先进的人工智能模型，通过分析大量文本数据中的模式来学习理解和生成类人文本，已成为自然语言处理（NLP）领域的最新技术水平，并被认为是编码器-解码器架构的演进。它最初是为序列转导或神经机器翻译而设计的，擅长将输入序列转换为输出序列，并且是第一个完全依赖自注意力机制来计算其输入和输出表示，而不使用序列对齐的循环神经网络（RNN）或卷积的模型。

其核心特点是保留了编码器-解码器结构，并通过分析不同元素之间的关系来理解上下文和意义，这几乎完全依赖于一种称为"注意力"的数学技术。2017 年，Vaswani 等人发表的论文《Attention Is All You Need》引入了 Transformer 架构，该架构凭借其并行处理能力，迅速取代了循环神经网络（RNN）等早期模型，成为现代人工智能，尤其是大型语言模型（LLM）的基础。

然而，Transformer 的核心组件——自注意力机制（Self-Attention）——虽然赋予了模型强大的上下文理解能力，但也带来了计算瓶颈。具体而言，自注意力机制的计算复杂度和内存占用均与输入序列长度 N 的平方成正比，即 O(N²)。这意味着当处理长序列时，计算成本会急剧上升，成为一个主要的性能瓶颈。

### 1.2. 高效替代方案的需求

标准 Transformer 的二次复杂度带来了显著的局限性。在处理长序列（例如长文档、高分辨率图像或基因组数据）时，其计算和内存需求变得令人望而却步。例如，当输入序列长度增加时，内存使用量会迅速增长。这不仅限制了模型能够处理的上下文长度，还导致了高昂的训练和推理成本，并因巨大的能源消耗而对环境产生影响。

这些挑战催生了对"高效 Transformer"（Efficient Transformers）的研究，旨在通过修改流行的但计算要求高的基于 Transformer 的语言建模技术，使其在保持强大性能的同时降低计算和内存开销。高效 Transformer 的研究领域致力于通过模型压缩、高效架构设计等方法，减少 Transformer 模型的内存占用和计算成本，使其能够在实际设备上部署，并降低对计算资源的依赖。

### 1.3. 报告焦点：深入研究 Linear Attention、AFT 与 Longformer

本报告旨在深入分析三种具有代表性的高效 Transformer 架构：Linear Attention、Attention Free Transformer (AFT) 和 Longformer。它们分别代表了实现效率的三种不同思路：

- **Linear Attention** 通过改进注意力计算的数学公式，将复杂度降低至线性级别
- **AFT** 完全抛弃注意力机制，采用简化的加权平均方法
- **Longformer** 则采用稀疏注意力机制，在保持对长距离依赖建模能力的同时，有效减少了计算量

通过对这三种模型的结构原理、应用场景和性能表现进行详细探讨，本报告期望为理解和应用高效 Transformer 提供有价值的参考。

## 2. 理解标准 Transformer 的效率瓶颈

### 2.1. 自注意力机制：深入探究其运作方式

要理解 Transformer 的效率瓶颈，首先需要深入了解其核心组件——自注意力机制。在标准 Transformer 中，最常用的是缩放点积注意力（Scaled Dot-Product Attention）。其计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中，$Q$（Query）、$K$（Key）和 $V$（Value）是输入序列经过不同线性变换后得到的矩阵，$d_k$ 是键向量的维度，用于缩放点积结果以稳定梯度。$Q, K, V$ 矩阵均派生自相同的输入序列，这使得模型能够并行计算，无需像 RNN 那样顺序处理。

Transformer 通常还采用多头注意力（Multi-Head Attention）机制。该机制并非执行单一的注意力函数，而是将 $Q, K, V$ 分别线性投影 $h$ 次（$h$ 为注意力头的数量），得到不同子空间的表示。然后，对每个投影版本的 $Q, K, V$ 并行执行注意力计算。最后，将所有头的输出拼接起来并再次进行线性变换，得到最终的输出。

多头注意力允许模型在不同位置共同关注来自不同表示子空间的信息，从而增强了模型的表达能力。然而，这也意味着注意力计算被复制了多次，进一步增加了计算负担。


### 2.2. 随序列长度 ($N$) 增加的计算与内存开销

自注意力机制的计算瓶颈主要源于 $QK^T$ 这一步。假设输入序列的长度为 $N$，每个词的表示维度为 $d_k$（对于 $Q$ 和 $K$）或 $d_v$（对于 $V$）。那么，$Q$ 和 $K$ 的维度均为 $N \times d_k$。计算 $QK^T$ 会产生一个 $N \times N$ 大小的注意力分数矩阵，其中每个元素表示序列中一个词对另一个词的关注程度。这个矩阵的计算和存储都需要 $O(N^2)$ 的时间和空间复杂度。

例如，当输入序列长度达到 2048 时，注意力矩阵的尺寸便为 $2048 \times 2048$，包含超过 400 万个元素。这种二次方的增长使得模型在处理长序列时变得异常缓慢且消耗大量内存。

内存瓶颈不仅仅在于存储这个 $N \times N$ 的注意力矩阵本身，在反向传播过程中，为了计算梯度，这个大矩阵通常也需要被保留，进一步加剧了内存压力。这使得 Transformer 在处理超过典型长度（如 BERT 通常限制的 512 个词元）的序列时效率低下，甚至不可行。

这种计算和内存瓶颈不仅仅是理论上的，它在实际硬件（如 GPU）上表现得更为突出。GPU 的计算速度提升速度超过了内存带宽的提升速度，导致许多操作越来越受到高带宽内存（HBM）访问速度的限制。即使计算单元足够强大，频繁地从 HBM 读写巨大的注意力矩阵也会成为性能的瓶颈。

因此，高效的解决方案不仅需要减少浮点运算次数（FLOPs），还需要优化内存访问模式。这为后续讨论的 Linear Attention（通过改变数学公式避免生成 $N \times N$ 矩阵）、AFT（完全抛弃注意力机制）和 Longformer（通过稀疏化减少计算）等不同方法的价值奠定了基础。


## 3. Linear Attention：以线性复杂度重塑注意力机制

### 3.1. 核心概念与动机：打破二次方壁垒

Katharopoulos 等人在 2020 年的论文《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》中提出了 Linear Attention，这是一种革命性的方法，旨在将自注意力机制的计算复杂度从 O(N²) 降低到 O(N)。该工作的核心洞察是：通过巧妙的数学重构，可以避免显式计算完整的注意力矩阵，同时保持 Transformer 的并行训练能力和 RNN 的高效推理特性。

### 3.2. 理论基础：将 Softmax 注意力线性化

#### 3.2.1. 标准注意力的分解

标准的缩放点积注意力可以写成：

$$
\text{Attention}(Q, K, V) = D^{-1}AV
$$

其中：
- $A = \exp\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 是未归一化的注意力权重矩阵
- $D$ 是对角矩阵，$D_{ii} = \sum_j A_{ij}$，用于行归一化

#### 3.2.2. 核技巧与特征映射

Linear Attention 的关键创新是使用核函数来近似 softmax 操作。具体来说，他们将注意力权重表示为：

$$
\text{sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j)
$$

其中 $\phi$ 是一个特征映射函数。Katharopoulos 等人提出使用：

$$
\phi(x) = \text{elu}(x) + 1
$$

这里 `elu` 是指数线性单元激活函数。选择 $\text{elu}+1$ 的原因是：
1. 保证输出为正（模拟 $\exp$ 函数的效果）
2. 计算简单高效
3. 在实践中表现良好

#### 3.2.3. 利用结合律实现线性复杂度

一旦有了特征映射，第 $i$ 个位置的输出 $y_i$ 可以重写为：

$$
y_i = \frac{\sum_j \phi(q_i)^T \phi(k_j) v_j}{\sum_j \phi(q_i)^T \phi(k_j)} = \frac{\phi(q_i)^T \sum_j (\phi(k_j) v_j^T)}{\phi(q_i)^T \sum_j \phi(k_j)}
$$

利用矩阵乘法的结合律，我们可以先计算 $\sum_j \phi(k_j) v_j^T$ 和 $\sum_j \phi(k_j)$：

$$
S = \sum_j \phi(k_j) v_j^T \quad \text{and} \quad Z = \sum_j \phi(k_j)
$$

$S$ 是一个累积的键值对矩阵，$Z$ 是一个归一化向量。关键洞察是：$S$ 和 $Z$ 可以在所有查询之间共享，因此整个序列的计算复杂度降为 $O(N)$。


### 3.3. 因果注意力与 RNN 形式

#### 3.3.1. 因果掩码下的线性注意力

对于自回归任务（如语言建模），需要因果注意力，即位置 i 只能关注位置 j ≤ i。在这种情况下，Linear Attention 可以表示为：

$$
y_i = \frac{\phi(q_i)^T S_i}{\phi(q_i)^T Z_i}
$$

其中：
- $S_i = \sum_{j \leq i} \phi(k_j) v_j^T$
- $Z_i = \sum_{j \leq i} \phi(k_j)$

#### 3.3.2. RNN 递归形式

Katharopoulos 等人的重要发现是，因果 Linear Attention 可以完全等价地表示为 RNN：

$$
S_i = S_{i-1} + \phi(k_i) v_i^T
$$
$$
Z_i = Z_{i-1} + \phi(k_i)
$$
$$
y_i = \frac{\phi(q_i)^T S_i}{\phi(q_i)^T Z_i}
$$

这种 RNN 形式带来了巨大的优势：
- **训练时**：可以像标准 Transformer 一样并行计算
- **推理时**：可以像 RNN 一样以 O(1) 复杂度生成每个新词元
- **内存效率**：只需维护固定大小的隐状态 S_i 和 z_i

### 3.4. 实现细节与优化

#### 3.4.1. 数值稳定性

Linear Attention 在实现时需要注意数值稳定性：
- 使用 `float32` 或更高精度避免数值误差累积
- 在归一化时添加小的 epsilon 值避免除零
- 对特征向量进行适当的缩放

#### 3.4.2. 多头线性注意力

Linear Attention 自然支持多头机制：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLinearAttention(nn.Module):
    """
    一个完整的多头线性注意力模块实现。
    它包含输入投影、多头拆分、线性注意力核心计算和输出投影。
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 1. 初始化输入/输出投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def elu_feature_map(self, x):
        """ 特征映射函数 φ(x) = elu(x) + 1 """
        return F.elu(x) + 1.0

    def forward(self, query, key, value, mask=None):
        # query, key, value 的输入维度: [Batch, SeqLen, d_model]
        batch_size = query.size(0)

        # 2. 对 Q, K, V 进行线性投影
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 3. 拆分成多个头：[B, SeqLen, d_model] -> [B, num_heads, SeqLen, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. 应用特征映射函数
        Q_mapped = self.elu_feature_map(Q)
        K_mapped = self.elu_feature_map(K)
        
        # 5. 核心线性注意力计算（在每个头内部独立进行）
        # K_mapped, V 的维度: [B, h, N, d_k], [B, h, N, d_v] (这里 d_k=d_v=head_dim)
        # S 的维度: [B, h, d_k, d_v]
        # Z 的维度: [B, h, d_k]
        # 'bhjd,bhje->bhde' 表示在 batch(b) 和 head(h) 维度上独立计算
        S = torch.einsum('bhjd,bhje->bhde', K_mapped, V)
        Z = K_mapped.sum(dim=2)
        
        # 计算分子和分母
        numerator = torch.einsum('bhid,bhde->bhie', Q_mapped, S)
        denominator = torch.einsum('bhid,bhd->bhi', Q_mapped, Z).unsqueeze(-1) + 1e-8
        
        # 计算最终输出 Y (在每个头内部)
        Y = numerator / denominator # Y 的维度: [B, h, N, d_v]

        # 6. 合并多个头的输出并进行最终的线性投影
        # [B, h, N, d_v] -> [B, N, h, d_v] -> [B, N, d_model]
        Y = Y.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(Y)
```

### 3.5. 性能评估

#### 3.5.1. 速度提升

Katharopoulos 等人报告了显著的速度提升：

**表 2：Linear Attention 速度性能对比**

| 序列长度 | 标准注意力 | Linear Attention | 速度提升 |
|---------|-----------|------------------|---------|
| 512 | 1.0x | 3.5x | 3.5 倍 |
| 1024 | 1.0x | 8.9x | 8.9 倍 |
| 2048 | 1.0x | 23.8x | 23.8 倍 |
| 8192 | OOM | 可运行 | >100 倍 |

*注：测试环境为单个 V100 GPU，批大小为 8*

#### 3.5.2. 任务性能

Linear Attention 在多个基准任务上的表现：

**表 3：Linear Attention 任务性能对比**

| 任务 | 指标 | Linear Attention | 标准 Transformer | 性能差距 |
|------|------|-----------------|-----------------|---------|
| 图像生成 (CIFAR-10) | Bits/dim | 3.60 | 3.44 | -4.7% |
| 语音识别 (TIMIT) | PER | 18.0% | 17.2% | -4.7% |
| 机器翻译 (WMT14) | BLEU | 26.8 | 27.3 | -1.8% |
| 文本分类 (IMDB) | 准确率 | 88.2% | 88.5% | -0.3% |
| 复制任务 | 准确率 | 100% | 100% | 0% |

### 3.6. 理论分析与局限性

#### 3.6.1. 表达能力分析

Linear Attention 的表达能力受到特征映射 φ 的限制：
- 无法精确表示所有可能的注意力分布
- 对于需要"硬"注意力（sharp attention）的任务可能表现不佳
- 特征维度 d' 影响近似质量

#### 3.6.2. 主要局限性

1. **近似误差**：elu+1 特征映射不能完美近似 softmax
2. **任务敏感性**：在某些需要精确注意力模式的任务上性能下降
3. **特征维度权衡**：增加特征维度提高近似质量但增加计算成本
4. **训练不稳定性**：某些情况下可能需要特殊的初始化或学习率调整

### 3.7. 扩展与后续工作

Linear Attention 启发了许多后续研究：

1. **Performer**：使用随机傅里叶特征（FAVOR+）改进近似质量
2. **Cosformer**：基于余弦相似度的线性注意力
3. **Luna**：将线性注意力与固定长度的记忆结合
4. **FNet**：更激进地用傅里叶变换完全替代注意力

这些工作共同推动了线性复杂度序列建模的发展，使处理超长序列成为可能。

## 4. Attention Free Transformer (AFT)：无注意力机制的高效序列建模

### 4.1. 核心概念与动机：重新思考注意力的必要性

Attention Free Transformer (AFT) 代表了高效 Transformer 研究中的一个激进方向——完全抛弃注意力机制。AFT 的核心动机源于一个大胆的假设：也许我们并不需要复杂的注意力计算来实现有效的序列建模。相反，通过精心设计的位置偏置和可学习的特征交互机制，可能就足以捕获序列中的依赖关系。

这种思路的出发点是观察到标准注意力机制虽然强大，但其 O(N²)的复杂度成为了严重的计算瓶颈。与其试图近似或稀疏化注意力机制，AFT 选择了一条完全不同的道路。

### 4.2. 结构原理：简化的特征交互机制

#### 4.2.1. AFT 的核心计算

AFT 用一种极其简单的加权平均机制替代了复杂的注意力计算。其核心操作可以表示为：

$$
y_t = \sum_i w_{t,i} \odot x_i
$$

其中：
- $x_i$ 是位置 $i$ 的输入特征
- $w_{t,i}$ 是位置 $t$ 对位置 $i$ 的可学习权重
- $\odot$ 表示逐元素乘法
- $y_t$ 是位置 $t$ 的输出

关键创新在于权重 $w_{t,i}$ 的设计，它由两部分组成：
1. **全局可学习权重**：为每对位置学习一个固定的权重
2. **位置偏置**：基于相对位置的预定义或可学习的偏置

```python
class AFTFull(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        # 可学习的全局位置偏置矩阵 B [MaxSeqLen, MaxSeqLen, 1]
        self.bias = nn.Parameter(torch.Tensor(max_seq_len, max_seq_len))
        nn.init.xavier_uniform_(self.bias)

    def forward(self, Q, K, V):
        # Q, K, V: [Batch, SeqLen, Dim]
        batch_size, seq_len, dim = Q.size()

        # 计算 Q_t + K_i 并扩展维度以便广播加
        # [B, L, D] -> [B, L, 1, D] + [B, 1, L, D] = [B, L, L, D]
        qk = Q.unsqueeze(2) + K.unsqueeze(1)

        # 加上截取的位置偏置 B[:seq_len, :seq_len].view(1, L, L, 1)
        # 计算权重 w_{t,i} = σ(Q_t + K_i + B_{t-i})
        weights = torch.sigmoid(qk + self.bias[:seq_len, :seq_len].view(1, seq_len, seq_len, 1))

        # 计算加权和: y_t = Σ_i (w_{t,i} * v_i) / (Σ_i w_{t,i} + eps)
        # [B, L, L, D] * [B, 1, L, D] -> [B, L, L, D]
        weighted_v = weights * V.unsqueeze(1)
        output = weighted_v.sum(dim=2) / (weights.sum(dim=2) + 1e-8)

        return output
```

#### 4.2.2. 位置感知的特征融合

AFT 通过以下方式实现位置感知：

$$
w_{t,i} = \sigma(q_t \odot k_i + b_{t-i})
$$

其中：
- $q_t$ 和 $k_i$ 类似于注意力机制中的查询和键，但使用方式更简单
- $b_{t-i}$ 是基于相对位置 $(t-i)$ 的偏置项
- $\sigma$ 是 sigmoid 激活函数，确保权重在 $[0, 1]$ 范围内

#### 4.2.3. 计算复杂度分析

AFT 的核心思想是通过将查询（Query）和键（Key）的交互与值（Value）的加权分离开来，从而避免了 N x N 矩阵的计算。在其最简化的形式 AFT-Simple 中，输出 Y_t 是通过对所有 V_i 进行全局加权平均得到的，而这个加权平均的过程可以被重新组合，使得计算可以线性进行，类似于 Linear Attention 中利用结合律的方式。它首先计算一个上下文感知的全局“summary”，然后将这个“summary”与每个位置的查询进行交互，从而避免了逐对计算注意力分数。


### 4.3. AFT 的变体与优化

研究者提出了几种 AFT 的变体以适应不同场景：

#### 4.3.1. AFT-local
限制每个位置只与局部窗口内的位置交互，类似于卷积操作：
- 复杂度进一步降低
- 适合局部依赖较强的任务
- 可以通过堆叠层来扩大感受野

#### 4.3.2. AFT-conv
将 AFT 与深度可分离卷积结合：
- 保留了卷积的归纳偏置
- 适合处理具有空间结构的数据
- 在视觉任务上表现良好

#### 4.3.3. AFT-simple
最简化的版本，移除了所有非必要组件：
- 极低的计算开销
- 适合资源受限的场景
- 性能与复杂度的极致权衡

### 4.4. 应用与性能

AFT 在多种任务上展现了其实用性：

**表 4：AFT 关键性能指标摘要**

| 基准/任务 | 指标 | AFT 性能 | 标准 Transformer | 效率提升 |
|-----------|------|---------|----------------|----------|
| ImageNet 分类 | Top-1 准确率 | 82.1% (AFT-Full) | 83.1% (ViT) | 3 倍速度提升 |
| CIFAR-100 | 准确率 | 89.2% (AFT-Conv) | 90.1% (DeiT) | 2.5 倍速度提升 |
| Long Range Arena | 平均分数 | 74.8% | 76.3% | 5 倍内存节省 |
| 语言建模（WikiText-103） | 困惑度 | 27.1 | 24.8 | 4 倍速度提升 |

### 4.5. 优势与局限性

#### 4.5.1. 主要优势
- **极致的效率**：计算和内存开销都远低于标准注意力
- **训练稳定性**：没有注意力机制的数值不稳定问题
- **易于实现**：代码实现简单，易于优化和部署
- **硬件友好**：更适合在边缘设备上运行

#### 4.5.2. 主要局限性
- **表达能力受限**：在需要复杂推理的任务上性能明显下降
- **长距离依赖建模**：难以捕获非常长距离的依赖关系
- **灵活性不足**：固定的交互模式限制了模型的适应性
- **理论基础薄弱**：缺乏像注意力机制那样的强理论支撑

### 4.6. AFT 的设计哲学与启示

AFT 的出现提出了一个根本性的问题：注意力机制是否真的是序列建模的必需品？虽然 AFT 在某些复杂任务上的性能不如标准 Transformer，但它在许多实际应用中展现了足够好的性能，同时带来了显著的效率提升。

这种"够用即可"的设计哲学对高效 AI 的发展具有重要启示：
1. 并非所有任务都需要最复杂的模型
2. 在资源受限的场景下，简单的解决方案可能更实用
3. 效率和性能的权衡应根据具体应用场景来决定

AFT 的成功也激励了更多研究者探索注意力机制的替代方案，推动了整个领域对序列建模本质的重新思考。

## 5. Longformer：通过稀疏注意力高效处理长文档

### 5.1. 核心概念与动机：赋能 Transformer 处理文档级输入

标准 Transformer 模型，如 BERT，通常将其输入序列长度限制在较短的范围内（例如 512 个词元）。当面对需要处理数千甚至更长词元的长文档时，这些模型不得不采用截断输入或设计复杂的分块处理策略，这往往会丢失重要的上下文信息或增加模型设计的复杂度。

Longformer 的提出正是为了解决这一痛点，其核心动机是使 Transformer 能够直接、高效地处理长文档，同时将注意力机制的计算复杂度随序列长度线性扩展。

### 5.2. 架构创新：稀疏注意力机制

Longformer 的核心创新在于其稀疏注意力机制，它可以作为标准自注意力机制的直接替代品（drop-in replacement）。这种稀疏注意力结合了局部注意力和全局注意力，以在控制计算成本的同时捕获长距离依赖。

#### 5.2.1. 滑动窗口（局部）注意力

在滑动窗口注意力中，每个词元只关注其自身周围一个固定大小（例如 w）的窗口内的词元。具体来说，一个词元会关注其左右各 w/2 个词元。这种局部注意力模式的计算复杂度为 O(N×w)，其中 N 是序列长度。由于窗口大小 w 是一个固定的常数，因此这种模式实现了随序列长度 N 的线性扩展。

通过堆叠多层这样的局部注意力，模型的感受野会逐渐扩大，高层网络能够整合来自更广泛上下文的信息。这种机制主要用于构建上下文表示。

```python
def sliding_window_attention(Q, K, V, window_size, attention_mask=None):
    # Q, K, V: [Batch, Heads, SeqLen, Dim]
    batch, heads, seq_len, dim = Q.shape
    # 1. 计算原始注意力分数 [B, H, L, L]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)

    # 2. 创建滑动窗口掩码 [L, L]
    # 例如，对于序列中心位置，掩码是一个带宽为 window_size 的带状矩阵
    row_indices = torch.arange(seq_len).unsqueeze(1) # [L, 1]
    col_indices = torch.arange(seq_len).unsqueeze(0) # [1, L]
    mask = (torch.abs(row_indices - col_indices) <= window_size // 2)
    mask = mask.to(attn_scores.device).unsqueeze(0).unsqueeze(0) # [1, 1, L, L]

    # 3. 将窗口外的注意力分数置为负无穷
    attn_scores = torch.where(mask, attn_scores, torch.tensor(float('-inf')).to(attn_scores.device))

    # 4. 可选：加上任务相关的全局注意力掩码（如[CLS] token 关注所有）
    # if global_mask is not None:
    #    attn_scores = attn_scores.masked_fill(global_mask, new_value)

    # 5. 计算注意力权重和输出
    attn_weights = F.softmax(attn_scores, dim=-1)
    # 可选：应用 dropout
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

注意：以上代码仅为 概念演示，旨在清晰地展示滑动窗口注意力的逻辑。一个高效的实现（如 Longformer 的官方实现）不会 生成完整的 N x N 注意力矩阵，而是会采用分块计算（chunking）或专门的 CUDA 核函数等技术，仅计算和存储窗口内的注意力分数，从而真正实现 O(N x w) 的线性和内存复杂度。

#### 5.2.2. 空洞滑动窗口注意力

为了在不增加计算量的前提下进一步扩大感受野，Longformer 引入了空洞滑动窗口（Dilated Sliding Window）注意力。类似于卷积神经网络（CNN）中的空洞卷积，该机制在滑动窗口内引入"空洞"或间隙（dilation size d）。

在多头注意力机制中，不同的注意力头可以配置不同的空洞大小，使得一些头关注紧邻的局部上下文，而另一些头则可以跳过一些词元，关注更远距离的上下文信息。这种设计有效地借鉴了 CNN 中扩大感受野的思想，使其适用于 Transformer 架构。

#### 5.2.3. 任务驱动的全局注意力

除了局部注意力，Longformer 还包含一种任务驱动的全局注意力机制。在这种模式下，一小部分预先选定的词元（全局词元）可以关注序列中的所有其他词元，并且序列中的所有词元也可以关注这些全局词元。

这种全局注意力对于那些需要从整个序列中聚合信息的任务（例如，分类任务中的 `[CLS]` 词元）或需要编码特定归纳偏置的任务（例如，问答任务中问题词元需要关注整个文档）至关重要。全局注意力的存在使得 Longformer 能够构建完整的序列表示以用于最终预测。

由于全局词元的数量通常较小且与序列长度无关，因此全局注意力的引入并不会改变整体的线性复杂度。

Longformer 的设计非常务实，它通过结合计算成本较低的局部注意力和策略性放置的、计算成本较高的全局注意力，直接解决了"长文档"处理问题。这是一种工程上的解决方案，它承认并非所有信息都需要在每一层进行全局整合。

### 5.3. 实现随序列长度的线性扩展

通过巧妙地结合滑动窗口注意力（复杂度 O(N×w)）和全局注意力（复杂度 O(N×num_global_tokens)），Longformer 成功地将整体注意力计算复杂度降低至 O(N)。这是因为窗口大小 w 和全局词元的数量 num_global_tokens 通常是远小于序列长度 N 的常数。

### 5.4. 应用与性能

Longformer 在多种需要处理长上下文的自然语言处理任务中展现了强大的性能。

**表 5：Longformer 关键性能指标摘要**

| 基准/任务 | 指标 | Longformer 性能（变体） | 基线性能（例如 RoBERTa） | 来源 |
|-----------|------|-------------------|----------------------|------|
| WikiHop | F1 | 81.9 (Longformer-large) | 73.8 (RoBERTa-large) | Beltagy et al., 2020 |
| TriviaQA | F1 | 77.3 (Longformer-large) | 70.6 (RoBERTa-large) | Beltagy et al., 2020 |
| HotpotQA | F1 | 73.2 (Longformer-large) | 68.3 (RoBERTa-large) | Beltagy et al., 2020 |
| arXiv Summarization | ROUGE-L | 40.43 (LED-large) | （不同基线） | Beltagy et al., 2020 |
| text8（字符级语言建模） | BPC | 1.10 (Longformer) | 1.13 (Transformer-XL) | Beltagy et al., 2020 |
| enwik8（字符级语言建模） | BPC | 1.00 (Longformer) | 1.06 (Transformer-XL) | Beltagy et al., 2020 |
| IMDB 分类 | 准确率 | 95.7 (Longformer-base) | 94.9 (RoBERTa-base) | Beltagy et al., 2020 |
| Hyperpartisan News Detect | 准确率 | 92.3 (Longformer-base) | 88.2 (RoBERTa-base) | Beltagy et al., 2020 |

*注：表中的具体数值来源于相关研究，用于说明性能趋势。不同实验设置和模型配置可能导致结果差异。BPC 指 Bits Per Character。*

实验结果表明，预训练的 Longformer 在长文档任务上持续优于 RoBERTa 等强基线模型，并在 WikiHop、TriviaQA 等基准上取得了当时的最佳性能。在字符级语言建模任务（如 text8 和 enwik8）上也达到了顶尖水平。

### 5.5. 用于生成任务的 Longformer-Encoder-Decoder (LED)

为了支持长文档的序列到序列（seq2seq）生成任务，如摘要生成，研究者们提出了 Longformer-Encoder-Decoder (LED) 模型。LED 模型在其编码器部分采用了 Longformer 的高效稀疏注意力机制，使其能够处理长篇输入文档，并已在 arXiv 文档摘要等任务上证明了其有效性。例如，一个 LED-Large 模型可能包含约 4.6 亿参数。

### 5.6. 潜在的局限性

尽管 Longformer 取得了显著成功，但它也存在一些潜在的局限性：

- 全局注意力的配置（即哪些词元应被赋予全局注意力）往往是任务相关的，可能需要根据具体任务进行调整和实验。

- 对于那些并非严格需要非常长上下文的任务，或者信息主要集中在局部区域的任务（例如 TriviaQA 的部分情况，或 OntoNotes 指代消解中多数指称距离较近），Longformer 相对于上下文窗口有限的模型的性能提升可能较为温和。

- 即使采用了稀疏注意力，对于极长的文档，捕获文档开头和结尾之间非常遥远的主题关联等超长距离依赖关系仍然可能是一个挑战。这意味着稀疏性本身虽然提高了效率，但也可能在某些极端情况下限制了信息的流动范围。

Longformer 的成功表明，对于许多长序列任务，精心设计的稀疏注意力模式可以与尝试用线性复杂度近似完全注意力的方法一样有效，甚至更具实用性。选择哪些词元接收全局注意力成为一个关键的建模决策。未来，如果这些稀疏模式能够根据输入序列动态学习或自适应调整，可能会带来更高效和上下文感知能力更强的模型，这是对当前"任务驱动"全局注意力的一种自然扩展。

## 6. 比较性见解与更广阔的背景

### 6.1. 区分 Linear Attention、AFT 与 Longformer：通往效率的不同路径

Linear Attention、AFT 和 Longformer 代表了解决标准 Transformer 效率瓶颈的三种截然不同的策略。理解它们的根本区别对于选择合适的模型至关重要。

- **Linear Attention**：其核心在于**修改注意力计算的数学公式本身**。通过使用核函数近似 Softmax 或利用矩阵乘法的结合律，Linear Attention 避免了显式构造和存储 N×N 的注意力矩阵，从而将计算复杂度降低到 O(N)。从原理上讲，注意力仍然是"稠密"的（即每个查询理论上可以与所有键交互），但这种交互是通过一种计算上更高效的方式实现的。

- **AFT**：采用了最激进的方法，**完全抛弃注意力机制**。它使用简化的加权平均和位置偏置来实现序列建模，虽然牺牲了一定的表达能力，但获得了极致的计算效率。

- **Longformer**：它**保留了标准自注意力的计算方式**（即缩放点积注意力），但将其**稀疏地应用于输入序列**。通过结合滑动窗口（局部）注意力、空洞滑动窗口注意力和任务驱动的全局注意力，Longformer 只计算一部分最重要的注意力分数，从而将整体复杂度降低到 O(N)。

**表 1：Transformer 架构比较概览**

| 特征 | 标准 Transformer | Linear Attention | AFT | Longformer |
|------|------------------|------------------|-----|------------|
| **注意力复杂度** | O(N²) | O(N) | O(N)（无注意力） | O(N) |
| **内存占用（随序列长度）** | O(N²) | O(N) | O(N) | O(N) |
| **核心效率机制** | 无（完全自注意力） | 核函数近似 Softmax，矩阵乘法结合律 | 位置偏置+加权平均 | 稀疏注意力模式 |
| **主要优势** | 强大的上下文建模能力 | 极快的自回归推理速度，理论基础扎实 | 极致简单高效，硬件友好 | 有效处理极长文档，性能损失小 |
| **主要局限性** | 无法处理长序列 | Softmax 近似可能导致性能下降 | 表达能力受限，长距离依赖建模弱 | 全局注意力配置依赖任务 |
| **典型应用场景** | 机器翻译，文本摘要 | 极长序列的自回归语言建模 | 资源受限场景，实时推理 | 长文档分类、问答、摘要 |

此表清晰地展示了这四种架构在关键特性上的差异。标准 Transformer 作为基准，其强大但昂贵的特性催生了对效率的追求。Linear Attention 通过数学上的革新实现了根本性的复杂度降低，尤其在生成任务上表现突出。AFT 则选择了完全不同的道路，牺牲部分性能换取极致效率。而 Longformer 通过一种更工程化的稀疏化方法，成功地将 Transformer 的能力扩展到了以往难以企及的长文档领域。

### 6.2. 置于高效 Transformer 的更广阔图景中

Linear Attention、AFT 和 Longformer 只是高效 Transformer 研究领域中的三个突出代表。为了更全面地理解它们的地位，有必要将其置于更广阔的技术图景中。

- **其他稀疏注意力模型**：除了 Longformer，还有其他采用稀疏注意力思想的模型。
  - **BigBird**：结合了随机注意力、窗口注意力和全局注意力，也实现了线性复杂度，并在理论上保持了完整 Transformer 的一些特性。
  - **Sparse Transformers (Child et al., 2019)**：采用了跨步（strided）和固定（fixed）的稀疏注意力模式，也是早期探索稀疏化的重要工作之一。

- **基于哈希的方法**：
  - **Reformer**：使用局部敏感哈希（LSH）来近似注意力计算，将复杂度降低到 O(NlogN)。它还引入了可逆层等技术来进一步减少内存占用。

- **其他线性注意力变体**：
  - **Performer**：使用随机傅里叶特征（FAVOR+）来近似注意力核，理论基础更强
  - **Cosformer**：基于余弦相似度重新设计线性注意力
  - **FNet**：用傅里叶变换完全替代注意力机制

- **硬件感知优化（侧重实现）**：
  - **FlashAttention**：这是一种 IO 感知的注意力算法，它并不改变注意力计算的数学本质（仍然是精确注意力），而是通过优化 GPU 内存访问（如分块计算、重计算、核函数融合）来显著加速注意力的计算速度并减少内存占用。FlashAttention 的优化可以与各种 Transformer 架构（包括线性或稀疏变体）相结合，提供额外的效率提升。

- **模型压缩技术**：这些技术旨在减小现有模型的尺寸和计算量，而不一定改变其核心架构。
  - **知识蒸馏**：训练一个小型的"学生"模型来模仿一个大型"教师"模型的行为。
  - **剪枝**：移除模型中冗余的权重、注意力头或层。
  - **量化**：降低模型权重和激活值的数值精度（例如从 32 位浮点数量化到 8 位整数）。

- **类 RNN / 状态空间模型（SSM）**：近年来，出现了一些新的架构，它们借鉴了 RNN 的思想或采用状态空间模型，以线性复杂度处理序列，并在长序列任务上表现出色。
  - **Mamba**：一种基于选择性状态空间模型的架构，实现了线性时间序列建模，并在多种模态上取得了与 Transformer 相当甚至更好的性能。
  - **RetNet (Retention Network)** 和 **RWKV**：也是此类架构的代表，它们结合了 RNN 的高效推理和 Transformer 的并行训练能力。

高效 Transformer 领域不存在"一刀切"的解决方案。最佳方法取决于具体的任务需求、序列长度、可用的硬件资源以及在速度、内存和准确性之间的权衡。例如，Linear Attention 可能因其在自回归任务上的极高推理速度和理论优雅性而受到青睐；AFT 在资源极度受限的场景下是理想选择；Longformer 则因其在长文档理解方面的强大能力和较小的性能损失而表现突出；而 FlashAttention 则为任何希望加速精确注意力计算的场景提供了通用的实现层优化。

这种多样性表明，未来的突破很可能来自于结合不同方法的优势，例如将 FlashAttention 应用于 Linear Attention 或 Longformer，或者对稀疏模型进行量化。这种模块化的效率提升方式预示着一个灵活且不断演进的技术生态系统。

## 7. 挑战、局限性与未来展望

尽管高效 Transformer 的研究取得了显著进展，但该领域仍面临诸多挑战，并且未来的发展方向也充满机遇。

### 7.1. 高效 Transformer 设计中的持续挑战

- **效率与性能的平衡**：在提高计算效率（如降低 FLOPs、减少内存占用）的同时，保持甚至提升模型性能（如准确率、困惑度）始终是核心挑战。许多近似方法或稀疏化策略都可能以牺牲一定的模型表达能力为代价。例如，量化到极低比特（如 4 位）时，Transformer 模型的性能仍可能出现显著下降；结构化剪枝后，如何有效恢复模型性能也是一个难题。

- **位置信息编码**：一些线性注意力模型在有效编码序列中的位置信息方面可能存在困难，这可能影响其在某些需要精确位置感知的任务上的表现。

- **泛化能力**：将特定的效率提升技术推广到不同的任务、数据集和模态（如从文本到视觉、语音）仍然是一个挑战。一种在一个领域表现良好的高效架构，在另一领域可能不尽如人意。

### 7.2. Linear Attention、AFT 与 Longformer 的特定局限性

- **Linear Attention**：
  - **近似误差**：基于核函数的 Softmax 近似不可避免地会引入误差，这可能导致其在某些对注意力分布精度要求较高的任务上性能不如标准 Transformer。
  - **核函数选择**：核函数的选择对模型性能有直接影响，但目前缺乏通用的、理论指导下的最佳核函数选择方法，往往需要通过实验调整。
  - **训练稳定性**：在某些情况下可能需要特殊的初始化策略或学习率调度。

- **AFT**：
  - **表达能力严重受限**：完全抛弃注意力机制导致在复杂推理任务上性能显著下降。
  - **缺乏理论保证**：没有像注意力机制那样的强理论基础支撑其有效性。
  - **任务适应性差**：固定的特征交互模式难以适应多样化的任务需求。

- **Longformer**：
  - **全局注意力配置**：全局注意力的位置和数量通常是启发式或基于任务经验设定的，缺乏自适应性，可能不是最优配置。
  - **极长距离依赖**：尽管 Longformer 能够处理数千词元的序列，但对于横跨文档极大范围（例如文档开头和结尾）的非常细微或复杂的依赖关系，仅靠固定的稀疏模式可能仍难以完美捕获。

### 7.3. 未来研究方向与潜在进展

高效 Transformer 的未来发展将可能沿着以下几个方向演进：

- **混合架构**：结合不同方法的优势，例如将稀疏注意力与线性注意力机制相结合，或者将注意力机制与类 RNN 或状态空间模型（如 Mamba）的组件融合，有望取长补短，实现更优的效率-性能平衡。

- **可学习的稀疏性/线性化**：开发能够让模型在训练过程中自动学习最优稀疏注意力模式或自适应调整核函数参数的方法，而不是依赖固定的、预先设计的模式。

- **硬件-软件协同设计**：进一步加强算法与硬件的协同设计。类似于 FlashAttention 和 Mamba 中针对 GPU 特性进行的优化，未来的高效模型将更加关注底层硬件的并行能力、内存层级和通信带宽，以实现极致的性能。

- **理论理解的深化**：对高效 Transformer 的表达能力、收敛特性以及效率与性能之间的理论边界进行更深入的探索，为设计更优模型提供理论指导。特别是对 Linear Attention 的核函数选择和 AFT 的理论基础需要更多研究。

- **超越注意力的新架构**：继续探索如 Mamba 等完全不依赖或显著改变传统注意力机制的新型序列建模架构。这些模型通过引入选择性状态空间等新机制，为处理超长序列提供了全新的视角和可能性。

当前的研究趋势表明，该领域正朝着更"智能化"的效率提升方向发展——不仅仅是简单地减少运算量，而是追求自适应的、可学习的、并与硬件特性深度融合的方法。例如，固定稀疏模式（如 Longformer）或固定核函数（Linear Attention 的潜在瓶颈）或完全抛弃注意力（AFT 的激进选择）的局限性，都指向了对更动态、更灵活方法的潜在需求。

神经架构搜索（NAS）在高效 Transformer 设计中的应用以及 Mamba 中选择性状态空间模型的成功，都印证了这一趋势。

最终，"Transformer"本身的定义可能会变得更加灵活和宽泛，它可能会吸收来自 RNN、CNN 和状态空间模型等不同架构的元素，从而催生出新一代高效且功能强大的序列模型。这种架构的融合与创新，预示着一个充满活力的研究前景。

## 8. 结论：高效序列建模的演进格局

### 8.1. Linear Attention、AFT 与 Longformer 的贡献总结

Linear Attention、AFT 和 Longformer 作为高效 Transformer 领域的杰出代表，为解决标准 Transformer 在处理长序列时面临的二次复杂度瓶颈问题做出了重要贡献。

- **Linear Attention** 通过对自注意力机制进行根本性的数学重构，采用核函数近似和利用矩阵乘法的结合律，成功地将计算复杂度降低至线性级别。其核心贡献在于展示了在不显式构建 N×N 注意力矩阵的情况下，依然可以实现有效的序列建模，并且其 RNN 等价形式为自回归任务带来了显著的推理加速。Katharopoulos 等人的工作特别展示了 Transformer 和 RNN 之间的深层联系。

- **AFT** 采用了最激进的方法，完全抛弃注意力机制，使用简化的加权平均和位置偏置实现序列建模。虽然牺牲了部分表达能力，但为资源极度受限的场景提供了可行方案，并引发了对注意力机制必要性的根本性思考。

- **Longformer** 则通过引入精心设计的稀疏注意力模式——结合滑动窗口（局部）注意力、空洞滑动窗口注意力和任务驱动的全局注意力——在保持对长距离依赖捕获能力的同时，实现了计算复杂度的线性扩展。它为处理数千词元级别的长文档提供了切实可行的解决方案，并在多个长文本 NLP 基准上取得了优异性能。

这三种模型代表了通往效率的三条不同但均有价值的路径：Linear Attention 侧重于数学公式的革新，AFT 追求极致的简化，Longformer 则侧重于结构化的稀疏计算。

### 8.2. 对其影响及高效 NLP 未来的总结性思考

Linear Attention、AFT 和 Longformer 的出现，极大地推动了自然语言处理及相关领域处理长序列的能力，使得以往因计算资源限制而难以处理的任务（如整本书籍的分析、长篇对话的理解、基因组序列的建模、资源受限设备上的实时推理）成为可能。它们不仅提升了模型的效率，也为更广泛、更实际的 AI 应用铺平了道路。

展望未来，对效率的追求仍将是驱动模型架构创新和能力边界拓展的核心动力。高效 Transformer 的研究不会止步于现有的方法。正如本报告所探讨的，混合架构、可学习的稀疏/线性化策略、更深度的硬件-软件协同设计以及超越传统注意力机制的新范式，都预示着一个持续演进、充满活力的领域。

虽然 Linear Attention、AFT 和 Longformer 提供了显著的进步，但它们是更大、快速发展的生态系统的一部分。最终的"高效 Transformer"可能并非单一的某种架构，而是一个包含多种技术和策略的工具箱，研究人员和工程师可以根据具体任务的需求、数据特性和硬件条件，灵活地选择和组合这些工具，以构建出在特定场景下最优的解决方案。

未来趋势不再是“一种架构统治天下”，而是​​“专模专用”​​。就像计算机有 CPU、GPU、TPU 一样，未来也会有擅长长文本理解的 Longformer、擅长高效生成的 Linear Attention 模型、擅长边缘部署的 AFT 模型等。高效 Transformer 的研究正在为我们提供这个丰富的​​模型工具箱​​。例如，当我们需要处理法律文书或医学报告等长文档时，Longformer 是首选；当我们需要极高吞吐量的自回归生成任务时，Linear Attention 展现出巨大优势；而在资源受限的边缘设备上部署模型时，AFT 的设计哲学则提供了宝贵的参考。

## 9. 参考文献

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30. (提出了 Transformer 架构)

2. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning(pp. 5156-5165). PMLR.(将 Transformer 重新表述为循环神经网络 RNN，并提出了线性注意力机制)

3. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150. (提出了 Longformer 模型，通过结合局部滑动窗口注意力和任务驱动的全局注意力，以线性复杂度高效处理长文档)

4. Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., ... & Ahmed, A. (2020). Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 17283-17297. (提出 BigBird 模型，通过使用稀疏注意力机制（包括全局、局部和随机注意力）来有效地处理更长的序列)

5. Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., ... & Weller, A. (2020). Rethinking attention with performers. arXiv preprint arXiv:2009.14794.(提出了 Performer 模型，使用快速注意力 Via 正交随机特征（FAVOR+）方法，以线性复杂度近似标准注意力机制）)

6. Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451. (提出了 Reformer 模型，结合局部敏感哈希（LSH）注意力和可逆残差层，大幅降低了长序列处理的内存消耗)

7. Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.（提出了 Sparse Transformer 模型，通过因子化的稀疏注意力模式来生成更长的序列）

8. Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. Advances in Neural Information Processing Systems, 35, 16344-16359.(提出了 FlashAttention，一种具有 IO 感知能力的快速且内存高效的精确实现注意力算法，通过优化 GPU 内存读写来加速计算)

9. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752. (提出了 Mamba 模型，基于状态空间模型 SSM 并引入了选择性机制，实现了高效的线性时间序列建模)

10. Zhai, S., Talbott, W., Srivastava, N., Huang, C., Goh, H., Zhang, R., & Susskind, J. (2021). An attention free transformer. arXiv preprint arXiv:2105.14103.（提出了无注意力 Transformer AFT，用一种高效的、基于加权平均的特征交互机制完全替代了自注意力机制）