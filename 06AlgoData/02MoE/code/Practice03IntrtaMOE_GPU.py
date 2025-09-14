import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.multiprocessing as mp
import os
import time
import json
from datetime import datetime
from collections import defaultdict

# 性能分析器：监控专家负载和计算性能
class ExpertPerformanceAnalyzer:
    def __init__(self, num_experts, world_size):
        self.num_experts = num_experts
        self.world_size = world_size
        self.reset()
        
        # 创建性能日志目录
        self.log_dir = f"moe_performance_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
    
    def reset(self):
        # 重置所有统计
        self.expert_samples = [0] * self.num_experts  # 每个专家处理的样本数
        self.expert_compute_time = [0.0] * self.num_experts  # 每个专家的计算时间
        self.router_time = 0.0  # 路由计算时间
        self.aggregation_time = 0.0  # 结果聚合时间
        self.communication_time = 0.0  # 设备间通信时间
        self.batch_count = 0  # 批处理计数
    
    def record_router_time(self, duration):
        self.router_time += duration
    
    def record_expert_compute(self, expert_id, sample_count, duration):
        self.expert_samples[expert_id] += sample_count
        self.expert_compute_time[expert_id] += duration
    
    def record_aggregation_time(self, duration):
        self.aggregation_time += duration
    
    def record_communication_time(self, duration):
        self.communication_time += duration
    
    def increment_batch_count(self):
        self.batch_count += 1
    
    def get_stats(self):
        """计算并返回当前性能统计"""
        avg_samples_per_expert = [s / self.batch_count if self.batch_count > 0 else 0 
                                 for s in self.expert_samples]
        
        avg_time_per_expert = [t / self.batch_count if self.batch_count > 0 else 0 
                              for t in self.expert_compute_time]
        
        return {
            "total_batches": self.batch_count,
            "total_samples_per_expert": self.expert_samples,
            "avg_samples_per_expert": avg_samples_per_expert,
            "total_compute_time_per_expert": self.expert_compute_time,
            "avg_compute_time_per_expert": avg_time_per_expert,
            "total_router_time": self.router_time,
            "avg_router_time": self.router_time / self.batch_count if self.batch_count > 0 else 0,
            "total_aggregation_time": self.aggregation_time,
            "avg_aggregation_time": self.aggregation_time / self.batch_count if self.batch_count > 0 else 0,
            "total_communication_time": self.communication_time,
            "avg_communication_time": self.communication_time / self.batch_count if self.batch_count > 0 else 0,
            "load_imbalance": self._calculate_load_imbalance()
        }
    
    def _calculate_load_imbalance(self):
        """计算负载不平衡度，0 表示完全平衡，值越大表示越不平衡"""
        if self.batch_count == 0:
            return 0.0
            
        avg_samples = sum(self.expert_samples) / self.num_experts
        if avg_samples == 0:
            return 0.0
            
        # 计算每个专家与平均值的偏差平方和
        variance = sum(((s / self.batch_count) - avg_samples) ** 2 for s in self.expert_samples) / self.num_experts
        return variance / (avg_samples **2)  # 归一化方差
    
    def log_epoch_stats(self, epoch, rank):
        """记录并打印 epoch 性能统计"""
        stats = self.get_stats()
        
        # 仅在主进程打印统计信息
        if rank == 0:
            print("\n" + "="*50)
            print(f"Epoch Performance Analysis")
            print("="*50)
            print(f"Load Imbalance Score: {stats['load_imbalance']:.4f} (0 = perfect balance)")
            print(f"Average Router Time: {stats['avg_router_time']:.6f}s")
            print(f"Average Communication Time: {stats['avg_communication_time']:.6f}s")
            print(f"Average Aggregation Time: {stats['avg_aggregation_time']:.6f}s")
            print("\nExpert Load Distribution:")
            print("Expert ID | Avg Samples | Avg Compute Time | Total Samples")
            print("-"*60)
            for i in range(self.num_experts):
                print(f"{i:9d} | {stats['avg_samples_per_expert'][i]:11.1f} | {stats['avg_compute_time_per_expert'][i]:18.6f}s | {stats['total_samples_per_expert'][i]:13d}")
            print("="*50 + "\n")
        
        # 保存统计到文件
        with open(f"{self.log_dir}/epoch_{epoch}_rank_{rank}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

# 专家网络实现
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# MoE 模型架构
class MoEEP(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim, top_k=2,
                 capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 将不同专家分配到不同 GPU 设备
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # 路由器：决定样本分配给哪些专家
        self.router = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化性能分析器（会在训练函数中配置）
        self.perf_analyzer = None

    def set_perf_analyzer(self, analyzer):
        """设置性能分析器"""
        self.perf_analyzer = analyzer

    def forward(self, x):
        batch_size = x.size(0)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])  # 展平输入
        
        # 1. 路由计算（带计时）
        router_start = time.time()
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        router_duration = time.time() - router_start
        self.perf_analyzer.record_router_time(router_duration)
        
        # 2. 分布式负载均衡
        world_size = dist.get_world_size()
        capacity = int(self.capacity_factor * batch_size / (self.top_k * world_size))
        capacity = max(capacity, 1)  # 确保容量至少为 1
        
        # 3. 专家使用统计（带通信计时）
        comm_start = time.time()
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        expert_counts = expert_mask.sum(dim=0)  # 各专家被选中的次数
        dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)  # 跨设备聚合统计
        self.perf_analyzer.record_communication_time(time.time() - comm_start)
        
        # 4. 负载均衡损失
        density = probs.mean(dim=0)
        usage = expert_counts / (batch_size * world_size)
        balance_loss = (density * usage).sum() * self.num_experts
        
        # 5. 专家重要性损失（带通信计时）
        comm_start = time.time()
        importance = probs.sum(dim=0)
        dist.all_reduce(importance, op=dist.ReduceOp.SUM)
        self.perf_analyzer.record_communication_time(time.time() - comm_start)
        importance_loss = (importance ** 2).mean()
        
        aux_loss = balance_loss + importance_loss  # 总辅助损失
        
        # 6. 分布式专家计算（带每个专家的计时）
        outputs = []
        for expert_id in range(self.num_experts):
            # 确定专家所在设备
            device = f'cuda:{expert_id % torch.cuda.device_count()}'
            
            # 选择分配给当前专家的样本
            idx_mask = (expert_indices == expert_id).any(dim=-1)
            if idx_mask.sum() == 0:  # 无样本则跳过
                continue
                
            selected = torch.nonzero(idx_mask).flatten()
            selected = selected[:capacity]  # 容量截断
            sample_count = selected.numel()
            
            if sample_count == 0:  # 截断后为空则跳过
                continue

            # 跨设备计算（带计时）
            comm_start = time.time()
            expert_input = x[selected].to(device)
            compute_start = time.time()
            expert_output = self.experts[expert_id](expert_input)
            compute_duration = time.time() - compute_start
            
            # 记录专家计算统计
            self.perf_analyzer.record_expert_compute(expert_id, sample_count, compute_duration)
            
            # 加权并传回原设备（带通信计时）
            weights = expert_weights[selected, (expert_indices[selected] == expert_id).nonzero()[:,1]]
            weighted_output = (expert_output * weights.unsqueeze(-1)).to(x.device)
            self.perf_analyzer.record_communication_time(time.time() - comm_start)
            
            outputs.append((selected, weighted_output))
        
        # 7. 合并所有专家的输出（带计时）
        agg_start = time.time()
        final_output = torch.zeros_like(x)
        for selected, out in outputs:
            final_output[selected] += out  # 累加专家输出
        agg_duration = time.time() - agg_start
        self.perf_analyzer.record_aggregation_time(agg_duration)
        
        # 增加批处理计数
        self.perf_analyzer.increment_batch_count()
            
        return final_output.view(*orig_shape), aux_loss  # 恢复原始形状

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 生成模拟数据集
class SimulationDataset(Dataset):
    def __init__(self, size, input_dim):
        self.size = size
        self.input_dim = input_dim
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        data = torch.randn(self.input_dim)
        label = torch.randn(self.input_dim)  # 简单起见，使用随机标签
        return data, label

# 训练函数
def train(rank, world_size, args):
    setup(rank, world_size)
    
    # 创建性能分析器
    perf_analyzer = ExpertPerformanceAnalyzer(args.num_experts, world_size)
    
    # 模型参数配置
    model = MoEEP(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        capacity_factor=args.capacity_factor
    ).to(rank)
    
    # 关联性能分析器到模型
    model.set_perf_analyzer(perf_analyzer)
    
    # 将专家移动到指定设备
    for i, expert in enumerate(model.experts):
        model.experts[i] = expert.to(f'cuda:{i % world_size}')
    
    # 分布式数据并行包装
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 数据准备
    dataset = SimulationDataset(args.dataset_size, args.input_dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练循环
    start_time = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱
        model.train()
        total_loss = 0.0
        
        # 重置性能分析器
        perf_analyzer.reset()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(rank), y.to(rank)
            
            # 前向传播
            outputs, aux_loss = model(x)
            main_loss = F.mse_loss(outputs, y)
            loss = main_loss + 0.01 * aux_loss  # 组合损失
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 定期打印进度
            if batch_idx % args.log_interval == 0 and rank == 0:
                print(f'Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # 每个 epoch 结束时打印平均损失和性能统计
        avg_loss = total_loss / len(loader)
        if rank == 0:
            print(f'\nRank {rank}, Epoch {epoch+1} Average Loss: {avg_loss:.6f}')
            print(f'Epoch time: {time.time() - start_time:.2f} seconds\n')
        
        # 记录并打印性能统计
        perf_analyzer.log_epoch_stats(epoch, rank)
        
        start_time = time.time()
    
    cleanup()

# 主函数
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MoE Distributed Training with Performance Analysis')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                      help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                      help='learning rate (default: 1e-4)')
    parser.add_argument('--input-dim', type=int, default=512,
                      help='input dimension (default: 512)')
    parser.add_argument('--output-dim', type=int, default=512,
                      help='output dimension (default: 512)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                      help='hidden dimension (default: 1024)')
    parser.add_argument('--num-experts', type=int, default=8,
                      help='number of experts (default: 8)')
    parser.add_argument('--top-k', type=int, default=2,
                      help='number of experts to use per sample (default: 2)')
    parser.add_argument('--capacity-factor', type=float, default=1.2,
                      help='capacity factor (default: 1.2)')
    parser.add_argument('--dataset-size', type=int, default=10240,
                      help='size of simulation dataset (default: 10240)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    
    # 检查 GPU 数量
    if torch.cuda.device_count() < 8:
        print(f"Warning: Need at least 8 GPUs, but found {torch.cuda.device_count()}")
        return
    
    world_size = 8
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    # 确保 CUDA 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
    else:
        main()
