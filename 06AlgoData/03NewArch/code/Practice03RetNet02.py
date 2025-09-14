import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 设置随机种子，确保实验可复现
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 基础保留机制实现
class Retention(nn.Module):
    """
    Retention 机制核心实现
    论文: https://arxiv.org/abs/2307.08621
    """
    def __init__(self, d_model, head_size, gamma):
        super().__init__()
        self.gamma = gamma  # 衰减因子
        self.d_model = d_model
        self.head_size = head_size
        
        # 初始化 Q, K, V 投影矩阵
        self.q_proj = nn.Linear(d_model, head_size, bias=False)
        self.k_proj = nn.Linear(d_model, head_size, bias=False)
        self.v_proj = nn.Linear(d_model, head_size, bias=False)
        
        # 可学习的衰减矩阵参数
        self.decay = nn.Parameter(torch.log(torch.ones(head_size) * gamma))
        
    def parallel_forward(self, X):
        """
        并行模式 - 用于训练
        输入: [batch_size, seq_len, d_model]
        输出: [batch_size, seq_len, head_size]
        """
        batch_size, seq_len, _ = X.shape
        
        # 计算 Q, K, V
        Q = self.q_proj(X)  # [batch_size, seq_len, head_size]
        K = self.k_proj(X)  # [batch_size, seq_len, head_size]
        V = self.v_proj(X)  # [batch_size, seq_len, head_size]
        
        # 计算衰减矩阵 D
        indices = torch.arange(seq_len).to(X.device)
        decay_matrix = torch.exp(-self.decay * torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)))
        
        # 应用因果掩码 - 只允许查看之前的位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(X.device)
        decay_matrix = decay_matrix * causal_mask
        
        # 计算注意力权重并应用衰减
        attention_weights = Q @ K.transpose(-1, -2)  # [batch_size, seq_len, seq_len]
        retention_scores = attention_weights * decay_matrix
        
        # 应用缩放
        retention_scores = retention_scores / (self.head_size **0.5)
        retention_output = retention_scores @ V  # [batch_size, seq_len, head_size]
        
        return retention_output
    
    def recurrent_forward(self, X, prev_state=None):
        """
        递归模式 - 用于推理
        输入: [batch_size, seq_len, d_model]
        输出: [batch_size, seq_len, head_size], 最终状态
        """
        batch_size, seq_len, _ = X.shape
        
        Q = self.q_proj(X)  # [batch_size, seq_len, head_size]
        K = self.k_proj(X)  # [batch_size, seq_len, head_size]
        V = self.v_proj(X)  # [batch_size, seq_len, head_size]
        
        # 初始化状态（如果没有提供）
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.head_size, self.head_size).to(X.device)
        
        outputs = []
        current_state = prev_state
        
        # 逐步处理序列
        for t in range(seq_len):
            # 计算当前时间步的衰减
            decay = torch.exp(-self.decay * torch.ones(batch_size, 1).to(X.device))
            
            # 更新状态: S_t = γ * S_{t-1} + K_t^T @ V_t
            current_state = decay * current_state + K[:, t:t+1].transpose(-1, -2) @ V[:, t:t+1]
            
            # 计算输出: O_t = Q_t @ S_t
            output_t = Q[:, t:t+1] @ current_state
            outputs.append(output_t)
        
        # 拼接所有时间步的输出
        output = torch.cat(outputs, dim=1)
        return output, current_state
    
    def chunk_forward(self, X, chunk_size=64):
        """
        分块递归模式 - 用于长序列处理
        输入: [batch_size, seq_len, d_model]
        输出: [batch_size, seq_len, head_size]
        """
        batch_size, seq_len, _ = X.shape
        
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # 将序列分成块
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunks = []
        
        # 处理每个块
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            
            # 当前块的 Q, K, V
            Q_chunk = Q[:, start:end]
            K_chunk = K[:, start:end]
            V_chunk = V[:, start:end]
            
            # 计算块内并行部分
            chunk_inner = (Q_chunk @ K_chunk.transpose(-1, -2)) * torch.exp(
                -self.decay * torch.abs(torch.arange(end-start).unsqueeze(0) - 
                                       torch.arange(end-start).unsqueeze(1)).to(X.device)
            )
            
            # 计算块间递归部分（如果需要）
            if i > 0:
                # 计算与前一个块的交叉注意力
                cross_attention = Q_chunk @ K[:, :start].transpose(-1, -2)
                cross_decay = torch.exp(-self.decay * (torch.arange(end-start).unsqueeze(1) + 
                                                      torch.arange(start).unsqueeze(0) + 1)).to(X.device)
                chunk_inner += cross_attention * cross_decay
            
            # 应用缩放
            chunk_inner = chunk_inner / (self.head_size** 0.5)
            chunk_output = chunk_inner @ V_chunk
            chunks.append(chunk_output)
        
        return torch.cat(chunks, dim=1)

# 2. 完整 RetNet 层实现
class RetNetLayer(nn.Module):
    """
    完整的 RetNet 层实现，支持三种计算模式
    """
    def __init__(self, d_model, head_size, gamma=0.9):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.retention = Retention(d_model, head_size, gamma)
        
        # 分组归一化（GroupNorm）更适合保留机制
        self.norm = nn.GroupNorm(1, head_size)
        
        # FFN 部分
        self.ffn = nn.Sequential(
            nn.Linear(head_size, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, X, mode='parallel', **kwargs):
        """
        前向传播，支持三种模式:
        - parallel: 并行模式，用于训练
        - recurrent: 递归模式，用于推理
        - chunk: 分块模式，用于长序列
        """
        # 保留机制部分
        if mode == 'parallel':
            retention_out = self.retention.parallel_forward(X)
        elif mode == 'recurrent':
            retention_out, state = self.retention.recurrent_forward(X, kwargs.get('prev_state', None))
        elif mode == 'chunk':
            retention_out = self.retention.chunk_forward(X, kwargs.get('chunk_size', 64))
        
        # 应用归一化和残差连接
        retention_out = self.norm(retention_out)
        X = X + retention_out  # 残差连接
        
        # FFN 部分
        ffn_out = self.ffn(self.ffn_norm(X))
        X = X + ffn_out  # 残差连接
        
        if mode == 'recurrent':
            return X, state
        return X

# 3. FlashAttention 集成
try:
    from flash_attn import flash_attn_func
    
    class FlashRetention(nn.Module):
        """
        使用 FlashAttention 加速的 Retention 机制
        """
        def __init__(self, d_model, head_size, gamma):
            super().__init__()
            self.gamma = gamma
            self.d_model = d_model
            self.head_size = head_size
            
            self.q_proj = nn.Linear(d_model, head_size, bias=False)
            self.k_proj = nn.Linear(d_model, head_size, bias=False)
            self.v_proj = nn.Linear(d_model, head_size, bias=False)
            self.decay = nn.Parameter(torch.log(torch.ones(head_size) * gamma))
        
        def forward(self, X):
            batch_size, seq_len, _ = X.shape
            
            Q = self.q_proj(X)
            K = self.k_proj(X)
            V = self.v_proj(X)
            
            # 生成衰减掩码
            indices = torch.arange(seq_len).to(X.device)
            decay_mask = torch.exp(-self.decay * torch.abs(
                indices.unsqueeze(0) - indices.unsqueeze(1)
            ))
            
            # 应用因果掩码
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(X.device)
            decay_mask = decay_mask * causal_mask
            
            # 使用 FlashAttention
            output = flash_attn_func(
                Q, K, V,
                softmax_scale=1.0 / (self.head_size **0.5),
                causal=True
            )
            
            # 应用衰减（后处理）
            output = output * decay_mask.unsqueeze(0)
            
            return output

except ImportError:
    print("FlashAttention 未安装，使用标准实现")
    FlashRetention = Retention

# 4. 构建完整的 RetNet 模型
def setup_retnet_model(vocab_size, d_model=512, n_layers=6, head_size=64, gamma=0.9):
    """
    构建完整的 RetNet 模型
    """
    class RetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                RetNetLayer(d_model, head_size, gamma) for _ in range(n_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids, mode='parallel',** kwargs):
            x = self.embedding(input_ids)
            
            states = []
            for layer in self.layers:
                if mode == 'recurrent':
                    x, state = layer(x, mode=mode, **kwargs)
                    states.append(state)
                else:
                    x = layer(x, mode=mode,** kwargs)
            
            logits = self.output(x)
            
            if mode == 'recurrent':
                return logits, states
            return logits
    
    return RetNet()

# 5. 创建示例数据集和数据加载器
class DummyTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机序列作为示例数据
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # 目标是输入的偏移（简单语言建模任务）
        targets = torch.roll(input_ids, shifts=-1, dims=0)
        targets[-1] = 0  # 最后一个位置的目标设为 0
        return input_ids, targets

class RetrievalDataset(Dataset):
    def __init__(self, vocab_size, query_len=32, doc_len=256, num_samples=1000):
        self.vocab_size = vocab_size
        self.query_len = query_len
        self.doc_len = doc_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        query = torch.randint(0, self.vocab_size, (self.query_len,))
        # 生成 3 个文档，其中一个与查询相关（前半部分相似）
        doc1 = torch.cat([query[:self.query_len//2], 
                         torch.randint(0, self.vocab_size, (self.doc_len - self.query_len//2,))])
        doc2 = torch.randint(0, self.vocab_size, (self.doc_len,))
        doc3 = torch.randint(0, self.vocab_size, (self.doc_len,))
        
        documents = torch.stack([doc1, doc2, doc3])
        # 正确答案是第一个文档
        label = torch.tensor(0, dtype=torch.long)
        return query, documents, label

# 6. 训练函数
def train_model(model, dataloader, epochs=3, mode='parallel'):
    """
    训练函数，支持不同模式
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    model.to(device)
    total_tokens = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if mode == 'recurrent':
                outputs, _ = model(input_ids, mode=mode)
            else:
                outputs = model(input_ids, mode=mode)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算吞吐量
            total_tokens += input_ids.numel()
            elapsed_time = time.time() - start_time
            
            if batch_idx % 10 == 0:  # 为了演示，每 10 个 batch 打印一次
                tokens_per_sec = total_tokens / elapsed_time
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Tokens/sec: {tokens_per_sec:.2f}")
        
        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    return tokens_per_sec

# 7. 检索任务评估函数
def evaluate_retrieval_accuracy(model, retrieval_dataloader, context_length=4096):
    """
    评估长上下文检索准确率
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (queries, documents, labels) in enumerate(retrieval_dataloader):
            batch_size = queries.size(0)
            queries = queries.to(device)
            documents = documents.to(device)
            labels = labels.to(device)
            
            # 处理每个文档
            doc_outputs = []
            for i in range(documents.size(1)):
                # 拼接查询和文档
                inputs = torch.cat([queries, documents[:, i]], dim=1)
                # 使用分块模式处理
                outputs = model(inputs, mode='chunk', chunk_size=64)
                # 取查询部分的输出作为特征
                doc_outputs.append(outputs[:, :queries.size(1)].mean(dim=1))  # [batch_size, d_model]
            
            # 计算查询与每个文档的相似度
            query_feat = model(queries, mode='parallel').mean(dim=1)  # [batch_size, d_model]
            doc_feats = torch.stack(doc_outputs, dim=1)  # [batch_size, 3, d_model]
            
            # 计算余弦相似度
            scores = F.cosine_similarity(
                query_feat.unsqueeze(1),  # [batch_size, 1, d_model]
                doc_feats,  # [batch_size, 3, d_model]
                dim=2
            )  # [batch_size, 3]
            
            predictions = scores.argmax(dim=1)
            
            # 计算准确率
            correct += (predictions == labels).sum().item()
            total += batch_size
            
            if batch_idx % 10 == 0:
                print(f"评估 Batch {batch_idx}, 累计准确率: {correct/total:.4f}")
    
    accuracy = correct / total
    print(f"上下文长度 {context_length}, 检索准确率: {accuracy:.4f}")
    return accuracy

# 8. 梯度传播可视化
def visualize_gradients(model, input_sample):
    """
    可视化不同模式下的梯度传播
    """
    model.train()
    model.to(device)
    input_sample = input_sample.to(device)
    
    # 测试并行模式
    model.zero_grad()
    output_parallel = model(input_sample, mode='parallel')
    loss_parallel = output_parallel.sum()
    loss_parallel.backward()
    grads_parallel = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    
    # 测试分块模式
    model.zero_grad()
    output_chunk = model(input_sample, mode='chunk', chunk_size=64)
    loss_chunk = output_chunk.sum()
    loss_chunk.backward()
    grads_chunk = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    
    # 绘制梯度分布
    plt.figure(figsize=(10, 6))
    plt.plot(grads_parallel, label='Parallel Mode', alpha=0.7)
    plt.plot(grads_chunk, label='Chunk Mode', alpha=0.7)
    plt.xlabel('Parameter Index')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Flow Comparison Between Modes')
    plt.legend()
    plt.yscale('log')
    plt.savefig('gradient_comparison.png')
    plt.close()
    print("梯度对比图已保存为 'gradient_comparison.png'")

# 主函数：执行实验
def main():
    # 实验参数设置（为了演示，使用较小的模型规模）
    vocab_size = 10000
    d_model = 256
    n_layers = 3
    head_size = 64
    seq_len = 128
    batch_size = 32
    
    # 创建数据加载器
    print("创建数据集...")
    train_dataset = DummyTextDataset(vocab_size, seq_len, num_samples=1000)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    retrieval_dataset = RetrievalDataset(vocab_size, query_len=32, doc_len=seq_len-32)
    retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    print("初始化模型...")
    model = setup_retnet_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试不同模式的性能
    modes = ['parallel', 'chunk']
    throughputs = {}
    
    for mode in modes:
        print(f"\n 测试 {mode} 模式...")
        # 为每个模式创建新模型以确保公平比较
        model = setup_retnet_model(vocab_size, d_model, n_layers, head_size)
        throughput = train_model(model, train_dataloader, epochs=1, mode=mode)
        throughputs[mode] = throughput
    
    # 测试递归模式（推理）
    print(f"\n 测试递归模式（推理）...")
    model = setup_retnet_model(vocab_size, d_model, n_layers, head_size)
    model.to(device)
    model.eval()
    
    # 推理吞吐量测试
    input_sample = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 多次运行以获得稳定测量
            _, _ = model(input_sample, mode='recurrent')
    elapsed_time = time.time() - start_time
    infer_throughput = (batch_size * seq_len * 100) / elapsed_time
    throughputs['recurrent'] = infer_throughput
    print(f"递归模式推理吞吐量: {infer_throughput:.2f} tokens/sec")
    
    # 评估检索准确率（不同上下文长度）
    print("\n 评估检索准确率...")
    accuracies = {}
    context_lengths = [512, 1024, 2048]
    
    # 使用较大的模型进行评估
    eval_model = setup_retnet_model(vocab_size, d_model=256, n_layers=4)
    # 简单训练一下评估模型
    train_model(eval_model, train_dataloader, epochs=2)
    
    for cl in context_lengths:
        # 调整文档长度以达到目标上下文长度
        doc_len = cl - 32  # 减去查询长度
        retrieval_dataset = RetrievalDataset(vocab_size, query_len=32, doc_len=doc_len, num_samples=200)
        retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=16, shuffle=False)
        acc = evaluate_retrieval_accuracy(eval_model, retrieval_dataloader, context_length=cl)
        accuracies[f'context_{cl}'] = acc
    
    # 梯度可视化
    print("\n 生成梯度可视化...")
    visualize_gradients(model, input_sample)
    
    # 绘制实验结果
    print("\n 绘制实验结果...")
    
    # 绘制吞吐量对比
    plt.figure(figsize=(8, 5))
    modes = ['Parallel', 'Chunk', 'Recurrent']
    throughput_values = [throughputs['parallel'], throughputs['chunk'], throughputs['recurrent']]
    plt.bar(modes, throughput_values)
    plt.title('Training/Inference Throughput Comparison')
    plt.ylabel('Tokens/Second')
    plt.savefig('throughput_comparison.png')
    plt.close()
    print("吞吐量对比图已保存为 'throughput_comparison.png'")
    
    # 绘制准确率随上下文长度变化
    plt.figure(figsize=(8, 5))
    ctx_lengths = list(context_lengths)
    acc_values = [accuracies[f'context_{cl}'] for cl in context_lengths]
    plt.plot(ctx_lengths, acc_values, marker='o')
    plt.title('Retrieval Accuracy vs Context Length')
    plt.xlabel('Context Length')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_context.png')
    plt.close()
    print("准确率对比图已保存为 'accuracy_vs_context.png'")
    
    # 输出最终结果
    print("\n===== 实验结果 =====")
    print("吞吐量 (tokens/sec):")
    for mode, tp in throughputs.items():
        print(f"  {mode}: {tp:.2f}")
    
    print("\n 检索准确率:")
    for ctx, acc in accuracies.items():
        print(f"  {ctx}: {acc:.4f}")

if __name__ == "__main__":
    main()
