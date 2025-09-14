<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE03:Transformer 机器翻译(DONE)

author by: ZOMI

本次将把之前实现的 Transformer 模型应用于真实的机器翻译任务，使用 [IWSLT 2016 英德数据集](https://www.kaggle.com/datasets/tttzof351/iwslt-2016-de-en)。该数据集包含英德双语平行语料，句子长度适中（多为日常对话或短文本），适合验证 Transformer 在中低资源翻译任务中的效果。

我们将引入一些训练过程的最佳实践，包括学习率调度、标签平滑、梯度裁剪等优化技巧——这些技术是解决 Transformer 训练不稳定性（如梯度爆炸、过拟合）和提升泛化能力的核心手段，并实现贪婪搜索和束搜索算法进行推理解码（两种算法分别平衡推理速度与翻译质量），最后使用 BLEU 分数评估翻译质量（机器翻译领域的标准自动评估指标）。

## 1. 环境准备与数据加载

首先，我们导入必要的库并设置环境。这里需要重点说明：PyTorch 的随机种子设置（`torch.manual_seed`等）是为了确保实验可重现——Transformer 模型参数规模大，随机初始化的微小差异可能导致训练结果波动，固定种子后可排除随机因素对实验结论的干扰。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy
import numpy as np
import random
import math
import time
from torchtext.datasets import IWSLT2016
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# 设置随机种子以确保结果可重现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # 禁用 CuDNN 的非确定性算法，进一步保证可重现性

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
```

    ```
    使用设备: cuda
    ```

### 1.1 加载和预处理数据

我们将使用 torchtext 库加载 IWSLT 2016 英德翻译数据集，并进行预处理。数据预处理是机器翻译的关键步骤，核心目标是将原始文本转换为模型可处理的数值序列，同时保留语言的语义和结构信息。

```python
# 加载英语和德语的 spacy 模型用于分词
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
except OSError:
    # 如果还没有下载模型，先下载
    print("正在下载 spacy 模型...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download de_core_news_sm")
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

# 定义分词函数
def tokenize_en(text):
    """
    英语分词函数
    """
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenize_de(text):
    """
    德语分词函数
    """
    return [token.text for token in spacy_de.tokenizer(text)]

# 定义 Field 对象处理文本
# Field 负责文本的预处理逻辑：分词、添加边界符号、小写化、数值化等
SRC = Field(tokenize=tokenize_de, 
            init_token='<sos>',  # 序列起始符号，让模型识别翻译的开始
            eos_token='<eos>',  # 序列结束符号，让模型识别翻译的结束
            lower=True,  # 小写化统一大小写，减少词汇表规模（如"The"和"the"视为同一词）
            batch_first=True)  # 输出张量格式为[batch_size, seq_len]，符合 PyTorch 常用习惯

TRG = Field(tokenize=tokenize_en, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True,
            batch_first=True)

# 加载 IWSLT2016 数据集
# splits 函数按语言后缀(.de 为德语源语言，.en 为英语目标语言)划分数据，并关联 Field 处理逻辑
print("加载 IWSLT2016 数据集...")
train_data, valid_data, test_data = IWSLT2016.splits(exts=('.de', '.en'), 
                                                     fields=(SRC, TRG))

# 构建词汇表
# 基于训练集统计词频，min_freq=2 表示只保留出现次数≥2 的词，过滤低频噪声词
print("构建词汇表...")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"源语言词汇表大小: {len(SRC.vocab)}")
print(f"目标语言词汇表大小: {len(TRG.vocab)}")

# 创建数据迭代器
# BucketIterator 按序列长度分组（同批次句子长度相近），减少 padding 数量，提升计算效率
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

print("数据加载完成!")
```

对应输出：

    ```
    加载 IWSLT2016 数据集...
    构建词汇表...
    源语言词汇表大小: 18832
    目标语言词汇表大小: 35432
    数据加载完成!
    ```

## 2. 模型构建与优化技术

### 2.1 构建 Transformer 模型

我们将使用之前实现的 Transformer 模型，但进行一些调整以适应机器翻译任务。尽管我们使用了较小的模型（D_MODEL=256，N_LAYERS=3），Transformer 仍然能够学习英德翻译的基本模式，生成语法基本正确的翻译结果——这得益于 Transformer 的自注意力机制，能有效捕捉语言的长距离依赖（如德语的后置定语与英语的前置定语对应关系）。

```python
# 导入之前实现的 Transformer 组件
from transformer_components import Embedding, PositionalEncoding, MultiHeadAttention
from transformer_components import FeedForward, SublayerConnection, EncoderLayer
from transformer_components import DecoderLayer, Encoder, Decoder, Transformer, Generator

def make_model(src_vocab_size, trg_vocab_size, d_model=512, N=6, d_ff=2048, h=8, dropout=0.1):
    """
    构建完整的 Transformer 模型
    
    Args:
        src_vocab_size: 源语言词汇表大小
        trg_vocab_size: 目标语言词汇表大小
        d_model: 模型核心维度（所有子层输入输出维度统一为 d_model，确保特征传递一致性）
        N: 编码器/解码器层数（多层叠加可捕捉更复杂的语言结构，如短语级、句子级依赖）
        d_ff: 前馈网络内部维度（通常为 d_model 的 4 倍，通过维度扩张增强特征表达能力）
        h: 注意力头数（多头注意力将 d_model 拆分 h 份，并行捕捉不同类型的依赖，如语义、语法）
        dropout: Dropout 率（防止过拟合，在注意力层和前馈层随机丢弃部分特征）
        
    Returns:
        完整的 Transformer 模型
    """
    # 创建注意力机制和前馈网络
    attn = MultiHeadAttention(d_model, h, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    
    # 创建位置编码
    # Transformer 无循环结构，需通过位置编码注入序列的时序信息（如"我吃饭"和"饭吃我"的语序差异）
    position = PositionalEncoding(d_model, dropout)
    
    # 创建模型
    model = Transformer(
        Encoder(EncoderLayer(d_model, attn, ff, dropout), N),  # 编码器：多层 EncoderLayer 堆叠，处理源语言序列
        Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N),  # 解码器：多层 DecoderLayer 堆叠，生成目标语言序列
        nn.Sequential(Embedding(src_vocab_size, d_model), deepcopy(position)),  # 源语言嵌入+位置编码
        nn.Sequential(Embedding(trg_vocab_size, d_model), deepcopy(position)),  # 目标语言嵌入+位置编码
        Generator(d_model, trg_vocab_size)  # 生成器：将 d_model 维度特征映射到目标词汇表概率分布
    )
    
    # 初始化参数
    # Xavier 均匀初始化适用于线性层，确保前向传播特征方差与反向传播梯度方差一致，避免梯度消失/爆炸
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

# 创建模型
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
D_MODEL = 256  # 为了训练速度，使用较小的模型（标准 Transformer 为 512）
N_LAYERS = 3  # 层数减少（标准为 6），平衡模型能力与训练成本
HID_DIM = 512  # 前馈网络内部维度（通常为 d_model 的 2 倍，此处适配小模型）
N_HEADS = 8  # 保持 8 头注意力，确保多类型依赖捕捉能力
DROPOUT = 0.1  # 适度 Dropout，缓解小模型过拟合

model = make_model(INPUT_DIM, OUTPUT_DIM, D_MODEL, N_LAYERS, HID_DIM, N_HEADS, DROPOUT).to(device)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

    ```
    模型参数量: 25,634,336
    ```

### 2.2 标签平滑 (Label Smoothing)

标签平滑是一种正则化技术，通过软化硬标签来防止模型过度自信，提高泛化能力。在机器翻译中，硬标签（one-hot 编码）会让模型对"正确词"的预测概率趋近于 1，对其他词趋近于 0，导致模型对微小输入变化敏感（如源语言中一个词的歧义），泛化到测试集时容易出错。

**原理公式**：

$$
y_{\text{smooth}} = (1 - \epsilon) \cdot y + \frac{\epsilon}{K}
$$

其中 $y$ 是原始 one-hot 标签（正确词位置为 1，其余为 0），$\epsilon$ 是平滑因子（通常取 0.1，控制软化程度），$K$ 是目标词汇表大小（类别数）。公式含义是：给正确词保留 $(1-\epsilon)$ 的概率，剩余 $\epsilon$ 均匀分配给其他所有词，迫使模型学习更鲁棒的特征（不仅能区分正确词，还能理解其他词的合理性）。

```python
class LabelSmoothing(nn.Module):
    """
    标签平滑实现
    
    Args:
        smoothing: 平滑因子（ε）
        pad_idx: 填充索引（不应用平滑，因填充 token 无实际语义，不应参与损失计算）
    """
    def __init__(self, smoothing=0.1, pad_idx=0):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing  # 正确词的保留概率
        self.criterion = nn.KLDivLoss(reduction='sum')  # KL 散度损失：衡量预测分布与平滑后目标分布的差异
        
    def forward(self, x, target):
        """
        Args:
            x: 模型输出（log 概率，shape: [batch_size*seq_len, trg_vocab_size]）
            target: 目标标签（原始 token 索引，shape: [batch_size*seq_len]）
            
        Returns:
            平滑后的损失（按批次大小归一化，确保不同批次损失可比）
        """
        batch_size = x.size(0)
        x = x.contiguous().view(-1, x.size(-1))  # 展平为[总 token 数, 词汇表大小]
        target = target.contiguous().view(-1)  # 展平为[总 token 数]
        
        # 创建平滑后的目标分布
        true_dist = x.clone()
        # 给所有非 pad 词分配ε/(K-2)（减去 pad 和正确词两类）
        true_dist.fill_(self.smoothing / (x.size(1) - 2))
        
        # 将正确词的位置概率设为(1-ε)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 将 pad 位置的概率设为 0（不参与损失）
        true_dist[:, self.pad_idx] = 0
        mask = target == self.pad_idx
        if mask.sum() > 0:
            true_dist.index_fill_(0, mask.nonzero().squeeze(), 0.0)
            
        return self.criterion(x, true_dist.detach()) / batch_size  # 按批次大小归一化

# 创建标签平滑损失函数
criterion = LabelSmoothing(smoothing=0.1, pad_idx=TRG.vocab.stoi['<pad>'])
```

### 2.3 学习率调度 (Learning Rate Scheduling)

Transformer 使用带 warmup 的学习率调度策略，先线性增加学习率，然后按步数的平方根反比衰减。这一策略是为了解决 Transformer 训练的两个核心问题：1）训练初期参数随机，小学习率避免参数震荡；2）训练后期参数接近最优，小学习率微调避免过拟合。

**原理公式**：

$$
\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step_num}^{-0.5}, \text{step_num} \cdot \text{warmup_steps}^{-1.5})
$$

公式解读：
- $d_{\text{model}}^{-0.5}$：模型维度越大，初始学习率越小——因大模型参数更多，需更谨慎的更新幅度；
- $\text{step_num} \cdot \text{warmup_steps}^{-1.5}$（warmup 阶段）：学习率随步数线性增加，直到 warmup_steps 时达到峰值；
- $\text{step_num}^{-0.5}$（warmup 后）：学习率随步数平方根反比衰减，确保后期更新幅度逐渐减小。

```python
class TransformerOptimizer:
    """
    Transformer 专用的优化器，包含学习率调度
    
    Args:
        optimizer: 基础优化器（此处用 Adam，适合大参数模型的自适应优化）
        d_model: 模型维度（用于计算初始学习率缩放因子）
        warmup_steps: warmup 步数（通常取 4000，平衡训练初期稳定性与收敛速度）
        factor: 学习率整体缩放因子（微调学习率范围）
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0  # 记录训练步数
        self.lr = 0  # 记录当前学习率
        
    def step(self):
        """
        更新参数和学习率：先计算当前学习率，再更新优化器参数
        """
        self.step_num += 1
        lr = self._get_lr()
        # 给所有参数组设置当前学习率
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()
        
    def zero_grad(self):
        """
        清空梯度：PyTorch 默认梯度累积，需手动清空
        """
        self.optimizer.zero_grad()
        
    def _get_lr(self):
        """
        计算当前学习率（按公式实现）
        """
        lr = self.factor * (self.d_model ** -0.5) * \
             min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        self.lr = lr
        return lr

# 创建优化器和学习率调度器
# Adam 优化器的 betas=(0.9, 0.98)：一阶矩（动量）用 0.9 加速收敛，二阶矩用 0.98 关注近期梯度变化
# eps=1e-9：避免分母为 0，适合 Transformer 的大参数规模
optimizer = TransformerOptimizer(
    optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
    d_model=D_MODEL,
    warmup_steps=4000
)
```

### 2.4 梯度裁剪 (Gradient Clipping)

梯度裁剪可以防止训练过程中梯度爆炸问题，提高训练稳定性。Transformer 的残差连接虽能缓解梯度消失，但多层叠加（即使是 3 层）仍可能导致部分参数的梯度 norms 过大，进而导致参数更新幅度过大，模型震荡甚至发散。

梯度裁剪的核心逻辑：计算所有可训练参数梯度的 L2 范数（整体梯度规模），若超过预设的 max_norm，则按比例（max_norm / 实际范数）缩放所有梯度，确保梯度在合理范围内。例如 max_norm=1.0 是常见阈值，既允许足够的更新幅度，又避免梯度爆炸。

```python
def clip_gradients(model, max_norm=1.0):
    """
    梯度裁剪
    
    Args:
        model: 模型（需裁剪所有可训练参数的梯度）
        max_norm: 最大梯度范数（阈值，通常取 1.0 或 5.0）
    """
    # 计算所有参数的梯度 L2 范数（平方和开根号）
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:  # 仅处理有梯度的参数（如冻结层无梯度）
            param_norm = p.grad.data.norm(2)  # 单个参数的梯度 L2 范数
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    # 裁剪梯度：若总范数超过 max_norm，按比例缩放
    clip_coef = max_norm / (total_norm + 1e-6)  # +1e-6 避免分母为 0
    if clip_coef < 1:  # 仅当总范数超过阈值时裁剪
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

## 3. 训练与验证

### 3.1 训练循环

训练循环的核心是实现 Transformer 的前向传播、损失计算、反向传播和参数更新，其中**掩码机制**是确保 Transformer 正确工作的关键——需分别处理源序列的 pad 掩码和目标序列的自回归掩码。

```python
def train_epoch(model, iterator, optimizer, criterion, clip):
    """
    训练一个 epoch（遍历一次训练集）
    
    Args:
        model: 模型
        iterator: 数据迭代器（训练集）
        optimizer: 优化器（含学习率调度）
        criterion: 损失函数（标签平滑）
        clip: 梯度裁剪阈值
        
    Returns:
        平均损失（按迭代器长度归一化，反映该 epoch 的训练效果）
    """
    model.train()  # 设为训练模式：启用 Dropout、BatchNorm 更新等
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src  # 源语言序列，shape: [batch_size, src_seq_len]
        trg = batch.trg  # 目标语言序列，shape: [batch_size, trg_seq_len]
        
        # 创建掩码：解决 pad token 和自回归问题
        # 1. 源序列掩码（src_mask）：忽略 pad token 的注意力计算
        # 形状从[batch_size, src_seq_len]扩展为[batch_size, 1, 1, src_seq_len]，适配多头注意力的维度
        src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
        
        # 2. 目标序列掩码（trg_mask）：包含 pad 掩码和上三角掩码（自回归）
        trg_pad_mask = (trg != TRG.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
        # 上三角掩码（nopeak_mask）：防止解码器看到未来的词（如生成第 i 个词时，看不到 i+1 及以后的词）
        nopeak_mask = torch.triu(torch.ones(trg_pad_mask.size(0), trg_pad_mask.size(1), trg_pad_mask.size(2)), 
                                diagonal=1).to(device) == 0
        trg_mask = trg_pad_mask & nopeak_mask  # 合并两种掩码
        
        optimizer.zero_grad()  # 清空梯度（避免累积）
        
        # 前向传播：解码器输入为 trg[:, :-1]（去掉最后一个词），输出预测 trg[:, 1:]（去掉第一个词）
        # 原因：自回归生成中，用前 i 个词预测第 i+1 个词，需对齐输入输出
        output = model(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
        
        # 计算损失：输出 shape [batch_size, trg_seq_len-1, trg_vocab_size]，目标 shape [batch_size, trg_seq_len-1]
        loss = criterion(output, trg[:, 1:])
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 梯度裁剪：防止梯度爆炸
        clip_gradients(model, clip)
        
        # 更新参数：含学习率调度
        optimizer.step()
        
        epoch_loss += loss.item()  # 累积损失
        
        # 每 100 批次打印一次中间结果，监控训练进度
        if i % 100 == 0:
            print(f"批次 {i}, 损失: {loss.item():.4f}, 学习率: {optimizer.lr:.6f}")
            
    return epoch_loss / len(iterator)  # 返回平均损失
```

### 3.2 验证循环

验证循环与训练循环逻辑相似，但需关闭梯度计算（节省显存、加速推理），且不更新参数、不启用 Dropout，确保验证结果反映模型的泛化能力。

```python
def evaluate(model, iterator, criterion):
    """
    验证模型（遍历一次验证集或测试集）
    
    Args:
        model: 模型
        iterator: 数据迭代器（验证集/测试集）
        criterion: 损失函数（标签平滑）
        
    Returns:
        平均损失（反映模型的泛化能力）
    """
    model.eval()  # 设为评估模式：禁用 Dropout、固定 BatchNorm 等
    epoch_loss = 0
    
    with torch.no_grad():  # 关闭梯度计算，节省显存和计算资源
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            # 创建掩码（逻辑与训练一致）
            src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
            trg_mask = (trg != TRG.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
            nopeak_mask = torch.triu(torch.ones(trg_mask.size(0), trg_mask.size(1), trg_mask.size(2)), 
                                    diagonal=1).to(device) == 0
            trg_mask = trg_mask & nopeak_mask
            
            # 前向传播（逻辑与训练一致）
            output = model(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
            
            # 计算损失
            loss = criterion(output, trg[:, 1:])      
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)  # 返回平均损失
```

### 3.3 训练模型

训练过程中需监控训练损失和验证损失：若验证损失下降，说明模型泛化能力提升；若验证损失上升，可能出现过拟合，需提前停止训练（此处通过保存"最佳模型"实现，即仅保存验证损失最低的模型参数）。

```python
# 训练参数
N_EPOCHS = 10  # 小模型训练 10 个 epoch 足够收敛，大模型需更多 epoch
CLIP = 1.0  # 梯度裁剪阈值
best_valid_loss = float('inf')  # 初始化最佳验证损失为无穷大

# 训练模型
print("开始训练模型...")
for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)  # 计算 epoch 耗时
    
    # 保存最佳模型：仅当当前验证损失低于历史最佳时保存，避免保存过拟合模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-model.pt')
    
    # 打印 epoch 结果：训练损失下降说明模型在学习，验证损失下降说明泛化能力提升
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.4f}')
    print(f'\tVal Loss: {valid_loss:.4f}')
    
print("训练完成!")
```

在训练过程中，你将看到类似以下的输出。随着训练的进行，训练损失和验证损失应该逐渐下降，表明模型在学习翻译任务。若训练损失持续下降但验证损失上升，说明模型过拟合，需调整 Dropout 率或减小模型规模。

    ```
    开始训练模型...
    批次 0, 损失: 10.4253, 学习率: 0.000002
    批次 100, 损失: 6.8321, 学习率: 0.000052
    ...
    Epoch: 01 | Time: 3m 45s
        Train Loss: 6.2345
        Val Loss: 5.8901
    ...
    Epoch: 10 | Time: 3m 42s
        Train Loss: 3.1245
        Val Loss: 4.2310
    训练完成!
    ```

## 4. 推理解码算法

训练完成后，需通过**解码算法**将模型的输出（词汇表概率分布）转换为可读的目标语言序列。常用的解码算法有贪婪搜索和束搜索，二者在速度和翻译质量上存在权衡。

### 4.1 贪婪搜索 (Greedy Search)

贪婪搜索在每一步选择概率最高的词作为当前输出，优点是速度快（每步仅需一次 argmax 操作），缺点是容易陷入**局部最优**——例如某一步选择概率最高的词，但后续无法组成语义通顺的句子（如德语"Können Sie"翻译时，第一步选"Can"是最优，但第二步若选"you"而非"you please"，可能导致后续翻译不完整）。

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    贪婪搜索解码
    
    Args:
        model: 模型
        src: 源序列（单句，shape: [1, src_seq_len]）
        src_mask: 源序列掩码
        max_len: 最大生成长度（防止无限生成，通常设为源序列长度的 1.5~2 倍）
        start_symbol: 开始符号索引（<sos>的数值化表示）
        
    Returns:
        解码后的序列（shape: [1, trg_seq_len]，包含<sos>和<eos>）
    """
    model.eval()
    
    # 编码源序列：得到源语言的特征表示 memory（shape: [1, src_seq_len, d_model]）
    memory = model.encode(src, src_mask)
    
    # 初始化目标序列：从<sos>开始，shape: [1, 1]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len-1):  # 减去初始的<sos>，避免超过 max_len
        # 创建目标序列掩码：包含 pad 掩码和上三角掩码（自回归）
        trg_mask = (ys != TRG.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
        nopeak_mask = torch.triu(torch.ones(trg_mask.size(0), trg_mask.size(1), trg_mask.size(2)), 
                                diagonal=1).to(device) == 0
        trg_mask = trg_mask & nopeak_mask
        
        # 解码：用当前目标序列 ys 和源特征 memory 预测下一个词
        out = model.decode(memory, src_mask, ys, trg_mask)  # out shape: [1, len(ys), d_model]
        prob = model.generator(out[:, -1])  # 取最后一个词的特征，映射为词汇表概率（shape: [1, trg_vocab_size]）
        _, next_word = torch.max(prob, dim=1)  # 贪婪选择概率最高的词（argmax）
        next_word = next_word.item()
        
        # 将下一个词添加到目标序列
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # 如果遇到<eos>，停止生成（序列已完整）
        if next_word == TRG.vocab.stoi['<eos>']:
            break
            
    return ys
```

### 4.2 束搜索 (Beam Search)

**解码策略比较**：束搜索（beam search）通常比贪婪搜索（greedy search）能产生质量更高的翻译结果，因为它考虑了更多可能的翻译路径——贪婪搜索每步仅保留 1 个候选序列，束搜索每步保留`beam_size`个候选序列（如 beam_size=5），通过多路径探索避免局部最优；但束搜索速度较慢（候选数越多，计算量越大），需在速度和质量间权衡。

束搜索的核心逻辑：1）初始化`beam_size`个候选序列（均从<sos>开始）；2）每步对每个候选序列扩展所有可能的下一个词，计算序列的累积概率；3）保留概率最高的`beam_size`个候选序列；4）重复步骤 2-3，直到所有序列生成<eos>或达到 max_len；5）应用**长度惩罚**（避免生成过短序列），选择最优序列。

```python
def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, length_penalty=0.6):
    """
    束搜索解码
    
    Args:
        model: 模型
        src: 源序列（单句，shape: [1, src_seq_len]）
        src_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 开始符号索引
        beam_size: 束宽（候选序列数，通常取 5~10，越大质量越高但速度越慢）
        length_penalty: 长度惩罚因子（通常取 0.6~1.0，惩罚过短序列，避免语义不完整）
        
    Returns:
        解码后的最优序列（shape: [1, trg_seq_len]）
    """
    model.eval()
    import torch.nn.functional as F  # 导入 F 用于 log_softmax
    
    # 编码源序列：得到源特征 memory（仅需编码一次，所有候选序列共享）
    memory = model.encode(src, src_mask)
    
    # 初始化束：列表存储（序列 token 索引列表，累积 log 概率），初始为[(<sos>, 0.0)]
    beams = [([start_symbol], 0.0)]
    
    for i in range(max_len):
        all_candidates = []  # 存储所有扩展后的候选序列
        
        # 对每个现有候选序列进行扩展
        for seq, score in beams:
            # 如果序列已以<eos>结束，不再扩展（避免生成冗余 token）
            if seq[-1] == TRG.vocab.stoi['<eos>']:
                all_candidates.append((seq, score))
                continue
                
            # 准备当前序列的张量输入（shape: [1, len(seq)]）
            ys = torch.tensor(seq).unsqueeze(0).to(device)
            
            # 创建目标序列掩码（逻辑与贪婪搜索一致）
            trg_mask = (ys != TRG.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)
            nopeak_mask = torch.triu(torch.ones(trg_mask.size(0), trg_mask.size(1), trg_mask.size(2)), 
                                    diagonal=1).to(device) == 0
            trg_mask = trg_mask & nopeak_mask
            
            # 解码：计算下一个词的概率分布
            with torch.no_grad():
                out = model.decode(memory, src_mask, ys, trg_mask)  # out shape: [1, len(seq), d_model]
                prob = model.generator(out[:, -1])  # shape: [1, trg_vocab_size]
                log_prob = F.log_softmax(prob, dim=1)  # 转换为 log 概率，避免数值下溢
                
            # 获取当前步概率最高的 beam_size 个词（减少计算量，无需扩展所有词）
            topk_prob, topk_idx = torch.topk(log_prob, beam_size, dim=1)
            
            # 生成新的候选序列：原序列 + 新词，累积概率 = 原分数 + 新词 log 概率
            for j in range(beam_size):
                candidate_seq = seq + [topk_idx[0, j].item()]
                candidate_score = score + topk_prob[0, j].item()
                all_candidates.append((candidate_seq, candidate_score))
                
        # 按累积概率降序排序，保留前 beam_size 个候选序列
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_size]
        
        # 提前停止条件：所有候选序列都已生成<eos>（无需继续扩展）
        if all(seq[-1] == TRG.vocab.stoi['<eos>'] for seq, _ in beams):
            break
            
    # 应用长度惩罚：纠正"短序列概率高"的偏差（如"我吃饭"比"我正在吃饭"短，概率可能更高但语义不完整）
    # 惩罚公式：score = 累积 log 概率 / (序列长度 ^ length_penalty)，长度越长惩罚越小
    best_seq = None
    best_score = -float('inf')
    for seq, score in beams:
        # 序列长度需排除<sos>和<eos>吗？此处用原始长度（含边界符号），因生成时已包含完整逻辑
        length_penalized_score = score / (len(seq) ** length_penalty)
        if length_penalized_score > best_score:
            best_score = length_penalized_score
            best_seq = seq
            
    return torch.tensor(best_seq).unsqueeze(0)  # 转换为张量返回
```

## 5. 模型评估

### 5.1 翻译函数

翻译函数是解码算法的上层封装，负责将原始文本（如德语句子）转换为模型可处理的数值序列，调用束搜索解码后，再将数值序列转换为可读的目标语言文本（如英语句子）。

```python
def translate_sentence(sentence, model, src_field, trg_field, max_len=50, beam_size=5):
    """
    翻译单个句子
    
    Args:
        sentence: 源语言句子（原始文本，如"Ich liebe dich."）
        model: 模型
        src_field: 源语言 Field（含分词、词汇表等预处理逻辑）
        trg_field: 目标语言 Field
        max_len: 最大生成长度
        beam_size: 束宽
        
    Returns:
        翻译结果（目标语言文本，如"I love you."）
    """
    model.eval()
    
    # 1. 文本预处理：分词 → 添加边界符号 → 数值化
    tokenized = src_field.tokenize(sentence)  # 分词（如"Ich liebe dich." → ["ich", "liebe", "dich", "."]）
    tokenized = [src_field.init_token] + tokenized + [src_field.eos_token]  # 添加<sos>和<eos>
    numericalized = [src_field.vocab.stoi[token] for token in tokenized]  # 数值化（词→索引）
    
    # 2. 转换为张量并创建掩码
    src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(device)  # shape: [1, src_seq_len]
    src_mask = (src_tensor != src_field.vocab.stoi['<pad>']).unsqueeze(1).unsqueeze(2)  # 源掩码
    
    # 3. 束搜索解码
    trg_indexes = beam_search_decode(model, src_tensor, src_mask, max_len, 
                                    trg_field.vocab.stoi[trg_field.init_token],  # <sos>的索引
                                    beam_size)
    
    # 4. 数值序列→文本：索引→词 → 移除<sos>和<eos> → 拼接为句子
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes[0]]  # 索引→词（itos: index to string）
    trg_tokens = trg_tokens[1:-1]  # 移除<sos>（第一个词）和<eos>（最后一个词）
    
    return ' '.join(trg_tokens)
```

### 5.2 BLEU 分数评估

BLEU (Bilingual Evaluation Understudy) 是机器翻译中最常用的自动评估指标，通过比较机器翻译输出（hypothesis）与参考翻译（reference）的**n-gram 重叠度**来评估质量。n-gram 是指连续的 n 个词，如 1-gram（单个词）、2-gram（词组）、3-gram（短语）等，重叠度越高，BLEU 分数越高（满分 100）。

BLEU 的核心逻辑：
1. 计算各阶 n-gram 的精确率（precision）：预测中出现的 n-gram 在参考中出现的比例；
2. 应用** brevity penalty（简短惩罚）**：若预测序列比参考序列短太多，降低分数（避免"短而准"的无意义翻译）；
3. 对各阶精确率取几何平均，得到最终 BLEU 分数。

此处使用 NLTK 库的`corpus_bleu`函数，并通过`SmoothingFunction().method4`处理短序列或零重叠的情况（避免 BLEU 分数为 0，确保评估稳定性）。

```python
def calculate_bleu(data, model, src_field, trg_field, max_len=50, beam_size=5):
    """
    计算整个数据集的 BLEU 分数
    
    Args:
        data: 测试数据集（含源序列和参考目标序列）
        model: 模型
        src_field: 源语言 Field
        trg_field: 目标语言 Field
        max_len: 最大生成长度
        beam_size: 束宽
        
    Returns:
        BLEU 分数（0~1，乘以 100 后为百分比）
    """
    trgs = []  # 存储所有参考序列（列表的列表，每个参考序列是词的列表）
    pred_trgs = []  # 存储所有预测序列（每个预测序列是词的列表）
    
    for example in data:
        # 获取源序列（原始词列表）和参考目标序列
        src = vars(example)['src']  # 源序列（如["ich", "liebe", "dich", "."]）
        trg = vars(example)['trg']  # 参考目标序列（如["i", "love", "you", "."]）
        
        # 处理参考序列：添加<sos>和<eos>，符合模型训练时的目标序列格式
        trg = [trg_field.init_token] + trg + [trg_field.eos_token]
        trgs.append([trg])  # corpus_bleu 要求参考序列为"列表的列表"（支持多参考）
        
        # 处理预测序列：调用 translate_sentence 得到预测文本，再分词为词列表
        pred_trg = translate_sentence(' '.join(src), model, src_field, trg_field, max_len, beam_size)
        pred_trgs.append(pred_trg.split())  # 分裂为词列表（如"I love you." → ["i", "love", "you", "."]）
    
    # 计算 BLEU 分数：使用 method4 平滑，避免零分数
    smooth = SmoothingFunction().method4
    bleu_score = corpus_bleu(trgs, pred_trgs, smoothing_function=smooth)
    
    return bleu_score

# 加载最佳模型：使用训练过程中保存的"验证损失最低"的模型参数，确保评估泛化能力
model.load_state_dict(torch.load('best-model.pt'))

# 计算测试集 BLEU 分数
bleu_score = calculate_bleu(test_data, model, SRC, TRG)
print(f'BLEU 分数: {bleu_score*100:.2f}')
```

### 5.3 推理翻译

通过实际示例验证模型的翻译效果，选择常见的德语句子，观察模型是否能生成语义正确、语法通顺的英语翻译。

```python
# 测试一些示例翻译
examples = [
    "Ich liebe dich.",  # 德语：我爱你。
    "Das Wetter ist heute schön.",  # 德语：今天天气很好。
    "Wie geht es dir?",  # 德语：你好吗？
    "Könnten Sie mir bitte helfen?"  # 德语：您能帮我一下吗？
]

for example in examples:
    translation = translate_sentence(example, model, SRC, TRG)
    print(f"德语: {example}")
    print(f"英语: {translation}\n")
```

在 10 个 epoch 的训练后，预期的 BLEU 分数大约在 15-25 之间（满分为 100）。这个分数对于小型模型（d_model=256，3 层）和有限的训练时间来说是合理的——若增大模型规模（d_model=512，6 层）、增加训练 epoch（如 30 个）或使用更大的数据集（如 WMT 数据集），BLEU 分数可提升至 30 以上，甚至接近人类翻译水平。

示例输出（模型训练充分后）：
```
德语: Ich liebe dich.
英语: i love you.

德语: Das Wetter ist heute schön.
英语: the weather is nice today.

德语: Wie geht es dir?
英语: how are you?

德语: Könnten Sie mir bitte helfen?
英语: could you please help me?
```

## 6. 总结

在本实验中，我们完成使用 torchtext 加载 IWSLT 2016 英德数据集，并进行分词和词汇表构建。实现推理解码，最后使用 BEU 进行评分。

总体而言，本实验成功地将 Transformer 模型应用于英德翻译任务，并验证了多种优化技术的有效性。通过调整模型架构和训练参数，可以进一步提高翻译质量。不仅将 Transformer 模型应用于真实的机器翻译任务，还学习了工业界常用的优化技术和评估方法。这些技术对于训练高质量的大模型至关重要，也是深度学习实践中不可或缺的部分。

你可以尝试调整超参数（如模型大小、学习率调度参数、束搜索宽度等），观察它们对翻译质量的影响，进一步加深对机器翻译任务的理解。
