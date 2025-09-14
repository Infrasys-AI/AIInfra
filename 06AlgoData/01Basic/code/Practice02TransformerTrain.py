import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


class Embedding(nn.Module):
    """
    标准的嵌入层，将 token 索引映射为 d_model 维的向量
    
    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度/嵌入维度
    """
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        # 初始化嵌入矩阵，使用标准正态分布初始化
        self.embed = nn.Parameter(torch.randn(vocab_size, d_model))
        # 缩放因子，用于控制嵌入数值范围
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: 输入 token 索引，形状为 (batch_size, seq_len)
            
        Returns:
            嵌入后的张量，形状为 (batch_size, seq_len, d_model)
        """
        # 根据索引从嵌入矩阵中查找对应的向量
        # 并乘以 sqrt(d_model)进行缩放，这是 Transformer 的标准做法
        return self.embed[x] * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码
    
    Args:
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout 率
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 计算位置信息
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 应用正弦函数到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用余弦函数到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 将 pe 注册为缓冲区（不参与梯度更新）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量，形状与输入相同
        """
        # 将位置编码添加到输入中（只取前 seq_len 个位置）
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力
    
    Args:
        query: 查询张量，形状为 (..., seq_len_q, d_k)
        key: 键张量，形状为 (..., seq_len_k, d_k)
        value: 值张量，形状为 (..., seq_len_v, d_v)
        mask: 可选的掩码张量
        dropout: 可选的 dropout 层
        
    Returns:
        输出张量和注意力权重
    """
    d_k = query.size(-1)
    
    # 计算 Q 和 K 的点积并缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    p_attn = F.softmax(scores, dim=-1)
    
    # 应用 dropout（如果提供）
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 返回加权和的值和注意力权重
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    Args:
        d_model: 模型维度
        h: 注意力头的数量
        dropout: Dropout 率
    """
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # 定义线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn = None  # 用于存储注意力权重（可视化用）
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: 查询张量，形状为 (batch_size, seq_len, d_model)
            key: 键张量，形状为 (batch_size, seq_len, d_model)
            value: 值张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的掩码张量
            
        Returns:
            多头注意力的输出，形状为 (batch_size, seq_len, d_model)
        """
        if mask is not None:
            # 同样的掩码应用于所有头
            mask = mask.unsqueeze(1)
        
        batch_size = query.size(0)
        
        # 1. 线性投影并分头
        query = self.w_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2. 应用注意力机制
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3. 拼接头并应用最终线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)


class FeedForward(nn.Module):
    """
    前馈网络
    
    Args:
        d_model: 模型维度
        d_ff: 前馈网络内部维度
        dropout: Dropout 率
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            前馈网络的输出，形状与输入相同
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    残差连接后的层归一化（修正版）
    
    Args:
        size: 输入维度
        dropout: Dropout 率
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        应用残差连接和层归一化（修正为标准 Transformer 顺序）
        
        Args:
            x: 输入张量
            sublayer: 子层函数（如注意力或前馈网络）
            
        Returns:
            归一化后的输出
        """
        # 标准 Transformer 实现: norm(x + dropout(sublayer(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    编码器层
    
    Args:
        d_model: 模型维度
        self_attn: 自注意力机制
        feed_forward: 前馈网络
        dropout: Dropout 率
    """
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) 
                                      for _ in range(2)])
        self.d_model = d_model
        
    def forward(self, x, mask):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量
            
        Returns:
            编码器层的输出，形状与输入相同
        """
        # 第一子层: 自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二子层: 前馈网络
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    解码器层
    
    Args:
        d_model: 模型维度
        self_attn: 自注意力机制
        src_attn: 编码器-解码器注意力机制
        feed_forward: 前馈网络
        dropout: Dropout 率
    """
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) 
                                      for _ in range(3)])
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: 解码器输入
            memory: 编码器输出（记忆）
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            解码器层的输出
        """
        m = memory
        
        # 第一子层: 自注意力（带目标掩码）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二子层: 编码器-解码器注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三子层: 前馈网络
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """
    编码器
    
    Args:
        layer: 编码器层
        N: 层数
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)
        
    def forward(self, x, mask):
        """
        依次通过所有编码器层
        
        Args:
            x: 输入张量
            mask: 掩码张量
            
        Returns:
            编码器的输出
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    解码器
    
    Args:
        layer: 解码器层
        N: 层数
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        依次通过所有解码器层
        
        Args:
            x: 输入张量
            memory: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            解码器的输出
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    Args:
        encoder: 编码器
        decoder: 解码器
        src_embed: 源序列嵌入
        tgt_embed: 目标序列嵌入
        generator: 生成器（输出层）
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播
        
        Args:
            src: 源序列
            tgt: 目标序列
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        Returns:
            模型输出
        """
        # 编码源序列
        memory = self.encode(src, src_mask)
        # 解码目标序列
        return self.decode(memory, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        """
        编码源序列
        
        Args:
            src: 源序列
            src_mask: 源序列掩码
            
        Returns:
            编码器输出（记忆）
        """
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码目标序列
        
        Args:
            memory: 编码器输出
            src_mask: 源序列掩码
            tgt: 目标序列
            tgt_mask: 目标序列掩码
            
        Returns:
            解码器输出
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    生成器（输出层）
    
    Args:
        d_model: 模型维度
        vocab: 词汇表大小
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        """
        Args:
            x: 解码器输出，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            投影到词汇表空间的结果，形状为 (batch_size, seq_len, vocab)
        """
        return F.log_softmax(self.proj(x), dim=-1)


def subsequent_mask(size):
    """
    生成后续位置掩码（用于解码器自注意力）
    
    Args:
        size: 序列长度
        
    Returns:
        下三角掩码矩阵，形状为 (1, size, size)
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建完整的 Transformer 模型
    
    Args:
        src_vocab: 源词汇表大小
        tgt_vocab: 目标词汇表大小
        N: 编码器/解码器层数
        d_model: 模型维度
        d_ff: 前馈网络内部维度
        h: 注意力头数
        dropout: Dropout 率
        
    Returns:
        完整的 Transformer 模型
    """
    # 创建注意力机制和前馈网络
    attn = MultiHeadAttention(d_model, h, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    
    # 创建位置编码
    position = PositionalEncoding(d_model, dropout)
    
    # 创建模型
    model = Transformer(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N),
        nn.Sequential(Embedding(src_vocab, d_model), deepcopy(position)),
        nn.Sequential(Embedding(tgt_vocab, d_model), deepcopy(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model


def data_gen(batch_size, n_batches, seq_len, vocab_size):
    """
    生成复制任务数据
    
    Args:
        batch_size: 批次大小
        n_batches: 批次数量
        seq_len: 序列长度
        vocab_size: 词汇表大小
        
    Returns:
        生成器，每次产生一个批次的(src, tgt)数据
    """
    for i in range(n_batches):
        # 随机生成源序列（排除 0，因为 0 通常用于填充）
        src = torch.randint(1, vocab_size, (batch_size, seq_len))
        # 目标序列与源序列相同（复制任务）
        tgt = src.clone()
        # 设置开始标记(1)和结束标记(vocab_size-1)
        src = torch.cat([torch.ones(batch_size, 1, dtype=torch.long), src], dim=1)
        tgt = torch.cat([torch.ones(batch_size, 1, dtype=torch.long), tgt, 
                        (vocab_size-1)*torch.ones(batch_size, 1, dtype=torch.long)], dim=1)
        yield src, tgt


class NoamOpt:
    """
    带 warmup 的学习率调度器
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def run_epoch(model, data_iter, loss_fn, optimizer):
    """
    运行一个训练周期
    
    Args:
        model: Transformer 模型
        data_iter: 数据迭代器
        loss_fn: 损失函数
        optimizer: 优化器
        
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for src, tgt in data_iter:
        batch_size, src_len = src.size()
        tgt_len = tgt.size(1)
        
        # 创建掩码
        src_mask = torch.ones(batch_size, 1, src_len)
        tgt_mask = subsequent_mask(tgt_len-1).expand(batch_size, -1, -1)
        
        # 前向传播
        out = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # 计算损失
        loss = loss_fn(out.contiguous().view(-1, out.size(-1)), 
                      tgt[:, 1:].contiguous().view(-1))
        
        # 反向传播
        optimizer.optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    return total_loss / n_batches


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    """
    自回归贪婪解码
    
    Args:
        model: Transformer 模型
        src: 源序列
        src_mask: 源序列掩码
        max_len: 最大生成长度
        start_symbol: 开始标记
        end_symbol: 结束标记
        
    Returns:
        生成的序列
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for _ in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys


# 测试模型
def test_model(model, vocab_size, seq_len):
    model.eval()
    with torch.no_grad():
        # 创建一个测试样本
        test_src = torch.randint(1, vocab_size-1, (1, seq_len))
        # 添加开始标记
        test_src = torch.cat([torch.ones(1, 1, dtype=torch.long), test_src], dim=1)
        # 创建目标（应该与源相同）
        test_tgt = torch.cat([test_src, (vocab_size-1)*torch.ones(1, 1, dtype=torch.long)], dim=1)
        
        # 创建掩码
        src_mask = torch.ones(1, 1, test_src.size(1))
        
        # 进行预测
        predicted_ids = greedy_decode(model, test_src, src_mask, 
                                     max_len=seq_len+2, 
                                     start_symbol=1, 
                                     end_symbol=vocab_size-1)
        
        print("输入序列:", test_src[0, 1:].numpy())  # 排除开始标记
        print("目标序列:", test_tgt[0, 1:-1].numpy())  # 排除开始和结束标记
        print("预测序列:", predicted_ids[0, 1:-1].numpy())  # 排除开始和结束标记
        print("匹配程度:", (predicted_ids[0, 1:-1].numpy() == test_tgt[0, 1:-1].numpy()).mean())


# 主函数
def main():
    # 设置超参数
    vocab_size = 11  # 小词汇表，包含 0-10，其中 1 是开始标记，10 是结束标记
    base_seq_len = 10  # 短序列
    d_model = 32     # 小模型维度（为了快速训练）
    N = 2            # 2 层编码器和解码器
    h = 4            # 4 个注意力头
    d_ff = 64        # 前馈网络内部维度
    dropout = 0.1    # Dropout 率

    # 创建模型
    model = make_model(vocab_size, vocab_size, N, d_model, d_ff, h, dropout)

    # 定义优化器和损失函数
    optimizer = NoamOpt(d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    loss_fn = nn.NLLLoss(ignore_index=0)  # 忽略填充位置的损失

    # 生成训练数据
    print("开始训练...")
    for epoch in range(10):
        train_data = data_gen(30, 20, base_seq_len, vocab_size)
        loss = run_epoch(model, train_data, loss_fn, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("训练完成!")

    # 测试模型
    print("\n 测试模型...")
    test_model(model, vocab_size, base_seq_len)


if __name__ == "__main__":
    main()
