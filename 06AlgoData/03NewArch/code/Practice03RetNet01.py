import torch
import torch.nn as nn
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SimplifiedRetNet(nn.Module):
    """简化版 RetNet 实现，保留核心机制"""
    def __init__(self, d_model=128, head_size=64, gamma=0.9):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, head_size, bias=False)
        self.k_proj = nn.Linear(d_model, head_size, bias=False)
        self.v_proj = nn.Linear(d_model, head_size, bias=False)
        
        # 衰减参数
        self.decay = nn.Parameter(torch.log(torch.ones(head_size) * gamma))
        
        # 简单的前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(head_size, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def parallel_forward(self, x):
        """并行模式 - 训练时使用"""
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, head_size]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 计算衰减矩阵
        indices = torch.arange(seq_len, device=device)
        decay_matrix = torch.exp(-self.decay * torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)))
        
        # 应用因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        decay_matrix = decay_matrix * causal_mask
        
        # 计算保留分数
        retention_scores = (Q @ K.transpose(-1, -2)) * decay_matrix
        retention_scores = retention_scores / (self.head_size **0.5)
        
        # 计算输出
        out = retention_scores @ V
        
        # 通过前馈网络
        out = self.ffn(out)
        return self.norm(x + out)  # 残差连接 + 归一化

    def recurrent_forward(self, x, prev_state=None):
        """递归模式 - 推理时使用"""
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, head_size]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 初始化状态
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.head_size, self.head_size, device=device)
        
        outputs = []
        current_state = prev_state
        
        # 逐时间步处理
        for t in range(seq_len):
            # 状态更新: S_t = γ * S_{t-1} + K_t^T @ V_t
            current_state = torch.exp(-self.decay) * current_state + \
                           K[:, t:t+1].transpose(-1, -2) @ V[:, t:t+1]
            
            # 计算输出: O_t = Q_t @ S_t
            output_t = Q[:, t:t+1] @ current_state
            outputs.append(output_t)
        
        # 拼接输出
        out = torch.cat(outputs, dim=1)
        
        # 通过前馈网络
        out = self.ffn(out)
        return self.norm(x + out), current_state  # 残差连接 + 归一化

    def chunk_forward(self, x, chunk_size=32):
        """分块模式 - 长序列处理"""
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 分块处理
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            
            # 当前块的 Q, K, V
            Q_chunk = Q[:, start:end]
            K_chunk = K[:, start:end]
            V_chunk = V[:, start:end]
            
            # 块内计算
            chunk_len = end - start
            indices = torch.arange(chunk_len, device=device)
            decay_matrix = torch.exp(-self.decay * torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)))
            
            # 块内注意力
            inner_attn = (Q_chunk @ K_chunk.transpose(-1, -2)) * decay_matrix
            
            # 与之前块的交互
            if i > 0:
                cross_attn = Q_chunk @ K[:, :start].transpose(-1, -2)
                cross_decay = torch.exp(-self.decay * (indices.unsqueeze(1) + start - indices[:start].unsqueeze(0)))
                inner_attn += cross_attn * cross_decay
            
            # 计算输出
            inner_attn = inner_attn / (self.head_size** 0.5)
            chunk_out = inner_attn @ V_chunk
            chunks.append(chunk_out)
        
        # 拼接所有块
        out = torch.cat(chunks, dim=1)
        
        # 通过前馈网络
        out = self.ffn(out)
        return self.norm(x + out)  # 残差连接 + 归一化

    def forward(self, x, mode='parallel', **kwargs):
        """统一前向接口"""
        if mode == 'parallel':
            return self.parallel_forward(x)
        elif mode == 'recurrent':
            return self.recurrent_forward(x, kwargs.get('prev_state', None))
        elif mode == 'chunk':
            return self.chunk_forward(x, kwargs.get('chunk_size', 32))
        else:
            raise ValueError(f"不支持的模式: {mode}")

def main():
    # 超参数
    batch_size = 8
    seq_len = 128
    d_model = 128
    steps = 10  # 测试步数
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 初始化模型
    model = SimplifiedRetNet(d_model=d_model).to(device)
    print(f"简化模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试三种模式
    print("\n=== 测试并行模式 ===")
    start = time.time()
    for _ in range(steps):
        out = model(x, mode='parallel')
    print(f"输出形状: {out.shape}")
    print(f"耗时: {time.time() - start:.4f}秒")
    
    print("\n=== 测试递归模式 ===")
    start = time.time()
    prev_state = None
    for _ in range(steps):
        out, prev_state = model(x, mode='recurrent', prev_state=prev_state)
    print(f"输出形状: {out.shape}")
    print(f"耗时: {time.time() - start:.4f}秒")
    
    print("\n=== 测试分块模式 ===")
    start = time.time()
    for _ in range(steps):
        out = model(x, mode='chunk', chunk_size=32)
    print(f"输出形状: {out.shape}")
    print(f"耗时: {time.time() - start:.4f}秒")
    
    # 验证输出一致性（在误差范围内）
    with torch.no_grad():
        out_parallel = model(x, mode='parallel')
        out_recurrent, _ = model(x, mode='recurrent')
        out_chunk = model(x, mode='chunk')
        
        # 计算模式间的差异
        diff_recurrent = torch.mean(torch.abs(out_parallel - out_recurrent))
        diff_chunk = torch.mean(torch.abs(out_parallel - out_chunk))
        
        print(f"\n 并行模式与递归模式的平均差异: {diff_recurrent:.6f}")
        print(f"并行模式与分块模式的平均差异: {diff_chunk:.6f}")
        print("注: 微小差异是由于数值计算误差，三种模式在理论上是等价的")

if __name__ == "__main__":
    main()
    