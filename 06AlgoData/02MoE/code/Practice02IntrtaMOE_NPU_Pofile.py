import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch_npu
from torch_npu.contrib import transfer_to_npu
from moe_stand import Expert
import torch.nn.functional as F

# export MASTER_ADDR="175.99.2.2"  # 主节点 IP
# export MASTER_PORT="29500"      # 任意未被占用的端口

def generate_simulation_data(batch_size, input_dim):
    # 生成高斯分布的输入数据
    data = torch.randn(batch_size, input_dim)
    # 生成随机标签
    labels = torch.randn(batch_size, input_dim)
    return data, labels

class ExpertParallel(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, hidden_dim, top_k=2,
                 capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 专家网络分布在不同设备
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim).to(f'cuda:{i}') 
            for i in range(num_experts)
        ])
        
        # 路由器
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Dropout(dropout))
        
    def forward(self, x):
        batch_size = x.size(0)
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        # 路由计算
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 分布式计算设置
        world_size = dist.get_world_size()
        capacity = int(self.capacity_factor * batch_size / (self.top_k * world_size))
        capacity = max(capacity, 1)
        
        # 跨设备通信
        all_expert_counts = torch.zeros(self.num_experts, device=x.device)
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=1)
        expert_counts = expert_mask.sum(dim=0)
        dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)
        
        # 分布式负载均衡损失
        density = probs.mean(dim=0)
        usage = expert_counts / (batch_size * world_size)
        balance_loss = (density * usage).sum() * self.num_experts
        
        # 分布式专家计算
        outputs = []
        for expert_id in range(self.num_experts):
            # 获取该专家对应的设备
            device = f'cuda:{expert_id % torch.cuda.device_count()}'

            # 获取该专家对应的样本
            idx_mask = (expert_indices == expert_id).any(dim=-1)
            if idx_mask.sum() == 0:
                continue
                
            # 容量截断
            selected = torch.nonzero(idx_mask).flatten()
            if selected.numel() == 0:
                continue

            selected = selected[:capacity]
            if selected.numel() == 0:
                continue  # 如果容量限制后仍然为空，跳过当前循环

            # 跨设备传输
            expert_input = x[selected].to(device)
            expert_output = self.experts[expert_id](expert_input)
            
            # 加权输出传回原设备
            weights = expert_weights[selected, (expert_indices[selected] == expert_id).nonzero()[:,1]]
            weighted_output = (expert_output * weights.unsqueeze(-1)).to(x.device)
            
            outputs.append((selected, weighted_output))
        
        # 合并结果
        final_output = torch.zeros_like(x)
        for selected, out in outputs:
            final_output[selected] += out
            
        # 重要性损失
        importance = probs.sum(dim=0)
        dist.all_reduce(importance, op=dist.ReduceOp.SUM)
        importance_loss = (importance ** 2).mean()
        
        aux_loss = balance_loss + importance_loss
        
        return final_output.view(*orig_shape), aux_loss

# 初始化分布式训练
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 训练循环
def train(rank, world_size):
    setup(rank, world_size)
    
    model = ExpertParallel(
        input_dim=512,
        output_dim=512,
        num_experts=8,
        hidden_dim=1024,
        top_k=2,
        capacity_factor=1.2
    ).to(rank)
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 数据加载器
    # dataset = YourDataset()  # 替换实际数据集
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    batch_size = 64
    input_dim = 512
    # 生成模拟数据
    data, labels = generate_simulation_data(batch_size, input_dim)
    dataset = list(zip(torch.tensor(data), torch.tensor(labels)))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )

    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=2, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./intrta_prof_result"),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config) as prof:
        for epoch in range(10):
            sampler.set_epoch(epoch)
            for x, y in loader:
                x = x.to(rank)
                y = y.to(rank)

                outputs, aux_loss = model(x)
                main_loss = F.mse_loss(outputs, y)
                total_loss = main_loss + 0.01 * aux_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                prof.step()
            
# 启动训练
if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)