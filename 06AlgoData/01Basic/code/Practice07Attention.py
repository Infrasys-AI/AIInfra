import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# 1. 基础缩放点积注意力
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力。
    
    Args:
        query: 查询张量，形状 (..., seq_len_q, d_k)
        key: 键张量，形状 (..., seq_len_k, d_k)
        value: 值张量，形状 (..., seq_len_v, d_v)
        mask: 可选的掩码张量，形状 (..., seq_len_q, seq_len_k)
    
    Returns:
        输出张量，形状 (..., seq_len_q, d_v)
        注意力权重张量，形状 (..., seq_len_q, seq_len_k)
    """
    # 1. 计算 Q 和 K 的转置的点积
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    
    # 2. 缩放：除以 sqrt(d_k)
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)
    
    # 3. 可选：应用掩码（在解码器中用于掩盖未来位置）
    if mask is not None:
        # 将掩码中为 0 的位置置为一个非常大的负数，softmax 后概率为 0
        scaled_attention_logits += (mask * -1e9)
    
    # 4. 计算注意力权重 (softmax on the last axis, seq_len_k)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1) # (..., seq_len_q, seq_len_k)
    
    # 5. 用注意力权重对 V 进行加权求和
    output = torch.matmul(attention_weights, value) # (..., seq_len_q, d_v)
    
    return output, attention_weights


# 2. 多头注意力 (MHA)
class MultiHeadAttention(nn.Module):
    """标准的多头注意力机制 (MHA)"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads # 每个头的维度
        
        # 定义线性投影层
        self.wq = nn.Linear(d_model, d_model) # W^Q
        self.wk = nn.Linear(d_model, d_model) # W^K
        self.wv = nn.Linear(d_model, d_model) # W^V
        self.dense = nn.Linear(d_model, d_model) # W^O
        
    def split_heads(self, x, batch_size):
        """将最后的维度 (d_model) 分割为 (num_heads, depth).
        并转置为 (batch_size, num_heads, seq_len, depth) 的形状
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影
        q = self.wq(q) # (batch_size, seq_len_q, d_model)
        k = self.wk(k) # (batch_size, seq_len_k, d_model)
        v = self.wv(v) # (batch_size, seq_len_v, d_model)
        
        # 2. 分割头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
        # 3. 缩放点积注意力 (在每个头上并行计算)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        
        # 4. 拼接头 (Transpose and reshape)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3) # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len_q, d_model)
        
        # 5. 最终线性投影
        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


# 3. 多查询注意力 (MQA)
class MultiQueryAttention(nn.Module):
    """多查询注意力 (MQA)"""
    
    def __init__(self, d_model, num_heads):
        super(MultiQueryAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        # Q 的投影和 MHA 一样，有 num_heads 个
        self.wq = nn.Linear(d_model, d_model)
        # K 和 V 的投影输出维度仅为 depth，意味着只有一个头
        self.wk = nn.Linear(d_model, self.depth) # 注意：输出是 depth，不是 d_model
        self.wv = nn.Linear(d_model, self.depth) # 注意：输出是 depth，不是 d_model
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads_q(self, x, batch_size):
        """仅对 Q 进行分头"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影
        q = self.wq(q) # -> (batch_size, seq_len_q, d_model)
        k = self.wk(k) # -> (batch_size, seq_len_k, depth)
        v = self.wv(v) # -> (batch_size, seq_len_v, depth)
        
        # 2. 仅对 Q 进行分头
        q = self.split_heads_q(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        # K 和 V 不分头，但为了广播计算，增加一个维度 (num_heads=1 的维度)
        k = k.unsqueeze(1) # (batch_size, 1, seq_len_k, depth)
        v = v.unsqueeze(1) # (batch_size, 1, seq_len_v, depth)
        
        # 3. 缩放点积注意力
        # 由于 k, v 的形状是 (batch_size, 1, seq_len, depth)，它们会自动广播到与 q 的 num_heads 维度匹配
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        
        # 4. 拼接头
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终线性投影
        output = self.dense(concat_attention)
        
        return output, attention_weights


# 4. 分组查询注意力 (GQA)
class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (GQA)"""
    
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_groups == 0, "num_heads 必须能被 num_groups 整除"
        
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.group_size = num_heads // num_groups # 每组包含的头数
        
        self.wq = nn.Linear(d_model, d_model)
        # K 和 V 的投影输出维度为: num_groups * depth
        self.wk = nn.Linear(d_model, num_groups * self.depth)
        self.wv = nn.Linear(d_model, num_groups * self.depth)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads_q(self, x, batch_size):
        """对 Q 进行分头"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def split_heads_kv(self, x, batch_size):
        """对 K, V 进行分组"""
        x = x.view(batch_size, -1, self.num_groups, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影
        q = self.wq(q) # -> (batch_size, seq_len_q, d_model)
        k = self.wk(k) # -> (batch_size, seq_len_k, num_groups * depth)
        v = self.wv(v) # -> (batch_size, seq_len_v, num_groups * depth)
        
        # 2. 分割头/组
        q = self.split_heads_q(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads_kv(k, batch_size) # (batch_size, num_groups, seq_len_k, depth)
        v = self.split_heads_kv(v, batch_size) # (batch_size, num_groups, seq_len_v, depth)
        
        # 3. 关键步骤：将 K, V 的组维度广播到与 Q 的头数匹配
        # 例如: k (bs, num_groups, ...) -> (bs, num_groups, 1, ...) -> (bs, num_groups, group_size, ...)
        k = k.unsqueeze(2) # 插入一个维度
        k = k.expand(-1, -1, self.group_size, -1, -1) # 扩展 group_size 次
        k = k.contiguous().view(batch_size, self.num_heads, *k.size()[3:]) # 重塑为 (bs, num_heads, seq_len_k, depth)
        
        v = v.unsqueeze(2)
        v = v.expand(-1, -1, self.group_size, -1, -1)
        v = v.contiguous().view(batch_size, self.num_heads, *v.size()[3:]) # (bs, num_heads, seq_len_v, depth)
        
        # 4. 缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 5. 拼接头
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)
        
        # 6. 最终线性投影
        output = self.dense(concat_attention)
        
        return output, attention_weights


# 5. 多潜在注意力 (MLA)
class MultiLatentAttention(nn.Module):
    """多潜在注意力 (MLA)"""
    
    def __init__(self, d_model, num_heads, num_latents):
        super(MultiLatentAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.num_latents = num_latents # 潜在向量的数量
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 可学习的潜在向量 (Keys and Values for latents)
        self.latent_k = nn.Parameter(torch.randn(1, num_latents, d_model)) # (1, num_latents, d_model)
        self.latent_v = nn.Parameter(torch.randn(1, num_latents, d_model)) # (1, num_latents, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 对原始输入进行投影
        q = self.wq(q)
        # 注意：这里我们不再使用输入的 k, v，而是使用可学习的潜在向量
        
        # 2. 获取潜在向量并扩展到 batch size
        k_latent = self.latent_k.expand(batch_size, -1, -1) # (batch_size, num_latents, d_model)
        v_latent = self.latent_v.expand(batch_size, -1, -1) # (batch_size, num_latents, d_model)
        
        # 3. 分割头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k_latent = self.split_heads(k_latent, batch_size) # (batch_size, num_heads, num_latents, depth)
        v_latent = self.split_heads(v_latent, batch_size) # (batch_size, num_heads, num_latents, depth)
        
        # 4. 计算 Q 和潜在 K 之间的注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k_latent, v_latent, mask)
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, depth)
        
        # 5. 拼接头
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)
        
        # 6. 最终线性投影
        output = self.dense(concat_attention)
        
        return output, attention_weights


# 6. 性能对比实验
def benchmark_attention(attention_class, config, seq_len, batch_size=2, device='cuda'):
    """基准测试函数"""
    d_model, num_heads = config['d_model'], config['num_heads']
    # 根据类需要传递额外的参数
    if attention_class == GroupedQueryAttention:
        model = attention_class(d_model, num_heads, num_groups=config.get('num_groups', 2)).to(device)
    elif attention_class == MultiLatentAttention:
        model = attention_class(d_model, num_heads, num_latents=config.get('num_latents', 64)).to(device)
    else:
        model = attention_class(d_model, num_heads).to(device)
        
    model.eval()
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 清空 GPU 缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 预热
    with torch.no_grad():
        _ = model(x, x, x)
    
    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(50): # 多次迭代取平均
            output, _ = model(x, x, x)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / 50
    
    # 估算内存占用 (参数数量)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"{model.__class__.__name__:>25}: Time = {avg_time*1000:>5.2f} ms, Params = {num_params:>6}")
    return avg_time, num_params


# 执行测试
if __name__ == "__main__":
    # 测试配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    seq_len = 1024
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_groups': 2,   # For GQA
        'num_latents': 64, # For MLA
    }

    print(f"\nBenchmarking with seq_len={seq_len}, d_model={config['d_model']}, num_heads={config['num_heads']}")
    print("-" * 60)

    results = {}
    for attn_class in [MultiHeadAttention, MultiQueryAttention, GroupedQueryAttention, MultiLatentAttention]:
        results[attn_class.__name__] = benchmark_attention(attn_class, config, seq_len, device=device)
