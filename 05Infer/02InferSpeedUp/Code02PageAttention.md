<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE 02: PagedAttention 复现

这篇文章将带你深入理解 PagedAttention 的工作原理，并通过简化的 Python 实现来展示其核心机制，帮助你在有限资源下高效运行大模型。

## 大模型推理内存挑战

随着大语言模型在自然语言处理、图像识别等领域的广泛应用，其庞大的参数规模和高内存需求已成为实际部署的主要瓶颈。特别是在移动设备和资源受限环境中，有限的内存容量往往无法满足大模型运行的基本需求。Transformer 模型在自回归推理过程中需要存储历史计算的键值对（KV Cache），这部分内存占用随着序列长度增加而线性增长，成为显存使用的主要因素。

传统 KV Cache 管理方式存在两个突出问题：一是**内存碎片化**，由于不同序列长度变化不可预测，导致显存分配不连续；二是**过度保留**，系统为应对可能的长序列往往预先分配过多显存，实际利用率却很低。研究表明，现有系统中 60%-80% 的显存因此被浪费。

PagedAttention 技术借鉴了操作系统中的虚拟内存和分页思想，通过将 KV Cache 划分为固定大小的块（block）并动态管理，显著提高了内存利用效率，使大模型在资源受限环境中的部署成为可能。

## 2. PagedAttention 原理

PagedAttention 的核心创新在于将操作系统的**分页机制**引入到注意力计算中。与传统方法需要为每个序列分配连续内存空间不同，PagedAttention 将键值缓存分割成固定大小的页面（page），每个页面存储一定数量 token 的键和值。

这种设计使得内存分配从连续变为非连续，通过一个块表（block table）来维护逻辑页面到物理页面的映射关系，类似于操作系统中的页表机制。当序列增长需要更多空间时，系统只需分配新的物理页面并更新块表，无需重新分配整个连续内存空间。

从数学角度看，传统的注意力计算可以表示为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q$、$K$、$V$ 分别是查询、键和值矩阵，$d_k$ 是键向量的维度。

在 PagedAttention 中，由于键和值被分散存储在多个页面中，注意力计算需要按页面进行：

$$
\text{Attention}(Q,K,V) = \sum_{b=1}^{N/B} \text{softmax}\left(\frac{QK_b^T}{\sqrt{d_k}}\right)V_b
$$

这里 $K_b$ 和 $V_b$ 表示第 $b$ 个页面的键和值，$B$ 是每个页面存储的 token 数量。这种分页计算方式虽然增加了页面管理开销，但大大降低了内存碎片化。

PagedAttention 的另一个重要优势是支持**高效的内存共享**。在并行采样或波束搜索等场景中，多个输出序列通常由同一个提示（prompt）生成。传统方法需要为每个序列单独存储键值缓存，而 PagedAttention 允许不同序列共享相同的物理页面，只需在块表中设置不同的映射关系即可。

这种内存共享机制显著减少了重复内容的内存占用，提高了整体吞吐量，特别是在处理长提示文本时效果更加明显。

## 3 页面与块表结构

下面我们通过一个简化的 Python 实现来展示 PagedAttention 的核心机制。为了使代码易于理解，我们省略了生产环境中的一些优化措施。

首先定义页面（Page）和块表（PageTable）的数据结构，这是 PagedAttention 的基础：

```python
import torch
import math

class Page:
    """表示一个物理页面，存储固定数量 token 的键值对"""
    def __init__(self, page_size, num_heads, head_dim):
        self.page_size = page_size  # 每个页面存储的 token 数量
        self.num_heads = num_heads
        self.head_dim = head_dim
        # 初始化键值存储空间
        self.keys = torch.zeros(page_size, num_heads, head_dim)
        self.values = torch.zeros(page_size, num_heads, head_dim)
        self.ref_count = 0  # 引用计数，用于页面复用

    def update_access(self):
        """更新页面访问信息，用于实现缓存替换策略"""
        self.ref_count += 1
```

`Page` 类表示一个物理页面，类似于操作系统中的内存页。每个页面有固定容量（`page_size`），存储多个 token 的键和值。`ref_count` 字段记录页面被引用的次数，用于实现类似 LRU 的页面置换算法。

```python
class PageTable:
    """管理逻辑页面到物理页面的映射关系"""
    def __init__(self):
        self.logical_to_physical = {}  # 映射表：逻辑页面 ID → 物理页面 ID

    def map_page(self, logical_page_id, physical_page_id):
        """建立逻辑页面到物理页面的映射"""
        self.logical_to_physical[logical_page_id] = physical_page_id

    def get_physical_page(self, logical_page_id):
        """根据逻辑页面 ID 获取物理页面 ID"""
        return self.logical_to_physical.get(logical_page_id, -1)
```

`PageTable` 类管理逻辑页面与物理页面的映射关系，类似于操作系统的页表。当需要访问特定逻辑页面时，通过这个映射表找到对应的物理页面位置。

## 4. 块管理器与序列管理

接下来实现块管理器（BlockManager）和序列管理器（SequenceManager），负责页面的分配、回收和序列的页面映射：

```python
class BlockManager:
    """管理全局页面池，负责页面的分配和回收"""
    def __init__(self, num_pages, page_size, num_heads, head_dim):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # 初始化页面池
        self.pages = [Page(page_size, num_heads, head_dim) for _ in range(num_pages)]
        self.free_pages = list(range(num_pages))  # 空闲页面列表
        self.allocated_pages = set()  # 已分配页面集合

    def allocate_page(self):
        """从内存池中分配一个物理页面"""
        if not self.free_pages:
            raise RuntimeError("No free pages available")
        page_id = self.free_pages.pop()
        self.allocated_pages.add(page_id)
        self.pages[page_id].ref_count += 1
        return page_id

    def free_page(self, page_id):
        """释放页面回到内存池"""
        if page_id in self.allocated_pages:
            self.pages[page_id].ref_count -= 1
            if self.pages[page_id].ref_count == 0:
                self.allocated_pages.remove(page_id)
                self.free_pages.append(page_id)
```

`BlockManager` 管理物理页面池，处理页面分配和回收。它维护了一个空闲页面列表和已分配页面集合，采用**引用计数**机制确保页面安全复用。当页面引用计数降为零时，页面被回收至空闲池。

```python
class SequenceManager:
    """管理每个序列的页面映射和令牌存储"""
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.sequences = {}  # 存储序列 ID 到页表的映射

    def create_sequence(self, seq_id):
        """创建新序列并初始化页表"""
        page_table = PageTable()
        self.sequences[seq_id] = page_table
        return page_table

    def append_token(self, seq_id, token_pos, key, value):
        """为序列添加 token 的键值对"""
        page_table = self.sequences[seq_id]
        # 计算逻辑页面 ID 和页面内偏移量
        logical_page_id = token_pos // self.block_manager.page_size
        page_offset = token_pos % self.block_manager.page_size
        
        # 获取或分配物理页面
        physical_page_id = page_table.get_physical_page(logical_page_id)
        if physical_page_id == -1:
            physical_page_id = self.block_manager.allocate_page()
            page_table.map_page(logical_page_id, physical_page_id)
        
        # 将键值对写入页面
        page = self.block_manager.pages[physical_page_id]
        page.keys[page_offset] = key
        page.values[page_offset] = value
        page.update_access()  # 更新页面访问信息
```

`SequenceManager` 为每个序列维护独立的页表。当写入新 token 时，它计算 token 应该所在的逻辑页面和页面内偏移量。如果逻辑页面尚未映射到物理页面，则从块管理器分配新物理页面。这种设计使得不同序列的页面可以混合存储在物理内存中，减少了内存碎片。

## 5. PagedAttention 计算

现在实现分页注意力计算的核心逻辑，从分散的页面收集键值并执行注意力运算：

```python
class PagedAttention:
    """执行分页注意力计算"""
    def __init__(self, block_manager):
        self.block_manager = block_manager

    def compute_attention(self, query, page_table, seq_len):
        """
        计算分页注意力
        query: [batch_size, num_heads, head_dim]
        page_table: 序列的页表
        seq_len: 当前序列长度
        """
        batch_size, num_heads, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # 收集所有页面的键值
        all_keys, all_values = [], []
        num_pages = (seq_len + self.block_manager.page_size - 1) // self.block_manager.page_size
        
        for logical_page_id in range(num_pages):
            physical_page_id = page_table.get_physical_page(logical_page_id)
            if physical_page_id == -1:
                continue  # 跳过未分配的页面
            page = self.block_manager.pages[physical_page_id]
            # 计算当前页面有效的 token 数量（最后一页可能不满）
            start_idx = logical_page_id * self.block_manager.page_size
            valid_tokens = min(self.block_manager.page_size, seq_len - start_idx)
            # 提取有效键值
            page_keys = page.keys[:valid_tokens]
            page_values = page.values[:valid_tokens]
            all_keys.append(page_keys)
            all_values.append(page_values)
        
        if not all_keys:
            return torch.zeros(batch_size, num_heads, head_dim)
        
        # 拼接所有键值
        keys = torch.cat(all_keys, dim=0)  # [seq_len, num_heads, head_dim]
        values = torch.cat(all_values, dim=0)  # [seq_len, num_heads, head_dim]
        
        # 计算注意力分数
        query = query.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
        keys = keys.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        scores = torch.matmul(query, keys.transpose(-2, -1)) * scale  # [batch_size, num_heads, 1, seq_len]
        scores = scores.squeeze(2)  # [batch_size, num_heads, seq_len]
        
        # Softmax 和加权求和
        attention_weights = torch.softmax(scores, dim=-1)
        values = values.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        output = torch.sum(attention_weights.unsqueeze(-1) * values, dim=2)
        return output  # [batch_size, num_heads, head_dim]
```


`PagedAttention` 类负责实际的分页注意力计算。它首先根据页表收集所有相关的键值页面，考虑到最后一页可能不满的情况。然后将这些页面拼接成完整的键值矩阵，执行标准的注意力计算。这种实现虽然增加了页面收集的开销，但大大降低了内存碎片化，提高了内存利用率。

## 6. 实验验证

以下代码演示了 PagedAttention 的完整工作流程，模拟了实际推理场景：

```python
# 实验设置和参数初始化
num_pages = 100
page_size = 8  # 每个页面存储 8 个 token
num_heads = 4
head_dim = 16
batch_size = 1

# 初始化管理器和注意力模块
block_manager = BlockManager(num_pages, page_size, num_heads, head_dim)
seq_manager = SequenceManager(block_manager)
paged_attn = PagedAttention(block_manager)

# 创建序列
seq_id = 0
page_table = seq_manager.create_sequence(seq_id)
seq_len = 0

# 模拟生成 20 个 token 的过程
print("开始模拟生成 20 个 token 的过程...")
for token_pos in range(20):
    # 生成随机键值（模拟 Transformer 层的输出）
    key = torch.randn(num_heads, head_dim)
    value = torch.randn(num_heads, head_dim)
    # 存储到 Cache
    seq_manager.append_token(seq_id, token_pos, key, value)
    seq_len += 1
    
    # 每 5 个 token 计算一次注意力
    if (token_pos + 1) % 5 == 0:
        query = torch.randn(batch_size, num_heads, head_dim)
        output = paged_attn.compute_attention(query, page_table, seq_len)
        print(f"已处理 Token 数量: {token_pos+1}, 注意力输出形状: {output.shape}")
        print(f"当前使用的物理页面数: {len([pid for pid in page_table.logical_to_physical.values() if pid != -1])}")
```

这段实验代码展示了 PagedAttention 的完整工作流程。它模拟了生成 20 个 token 的过程，每生成 5 个 token 计算一次注意力。通过输出信息可以看到内存使用情况，验证了 PagedAttention 的动态内存分配特性。

## 7. 总结与思考

PagedAttention 技术已广泛应用于多种大模型推理场景。在**长序列处理**中，如长文档生成、代码补全等任务，PagedAttention 通过分页机制有效支持了万级别 token 长度的序列，突破了传统方法的内存限制。

在高并发推理服务中，PagedAttention 的**内存共享特性**特别有价值。多个请求可以共享相同的提示页面，显著提高了吞吐量和资源利用率。vLLM 框架基于 PagedAttention 实现了高达 23 倍的吞吐量提升。
