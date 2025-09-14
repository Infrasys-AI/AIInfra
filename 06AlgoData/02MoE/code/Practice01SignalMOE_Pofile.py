import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        # 路由网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家集合
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device
        
        # 路由计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 辅助损失计算
        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)
            
            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
            
            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0.0

        # 专家分配逻辑
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()

        # 初始化输出
        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features, 
                            device=device)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            # 获取分配给当前专家的样本
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            # 容量控制
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            # 处理专家计算
            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # 累加输出
            outputs.index_add_(0, expert_samples, weighted_output)

        return outputs, aux_loss

# 测试示例
if __name__ == "__main__":
    input_dim = 128
    output_dim = 256
    num_experts = 8
    top_k = 2
    expert_capacity = 32
    hidden_dim = 512
    batch_size = 64

    # add
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)
    
    # 训练模式
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=[torch_npu.profiler.AiCMetrics.AiCoreNone],
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
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./stand_prof_result"),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config) as prof:
        for _ in range(5):
            moe.train()
            output, loss = moe(x)
            print(f"Output shape: {output.shape}")      # torch.Size([64, 256])
            print(f"Auxiliary loss: {loss.item():.4f}") # 示例值，如 0.1234
            prof.step()
    # 推理模式
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}") # torch.Size([64, 256])