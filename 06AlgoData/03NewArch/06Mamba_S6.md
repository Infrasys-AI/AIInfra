# Mamba 与选择性状态空间模型：一项全面的技术综述

## 摘要

Mamba 架构作为序列建模领域的一项重要进展，受到了广泛关注。它基于状态空间模型(SSM)构建，通过引入创新的选择性状态空间(S6)机制，实现了对输入序列的动态内容感知处理。这一核心特性使得 Mamba 能够有效过滤无关信息，专注于关键数据，从而在保持对长序列线性计算复杂度的同时，展现出与 Transformer 相当甚至更优的性能。Mamba 的设计充分考虑了现代硬件的特性，采用了包括并行扫描、核融合和梯度重算在内的高效算法，显著提升了训练和推理速度。与传统 SSM 相比，Mamba 的选择性机制克服了其在处理离散信息密集型数据(如文本)时的局限性；与 Transformer 相比，Mamba 在处理超长序列时展现出显著的效率优势。Mamba 生态系统不断发展，催生了 Mamba-ND(用于处理多维数据)、ReMamba(增强长上下文理解能力)和 Mamba-2(基于状态空间对偶性进一步提升性能)等重要变体。此外，关于 Mamba"隐藏注意力"机制和"状态空间对偶性"等高级理论概念的研究，也为理解其工作原理和与其他主流架构(如 Transformer)的关系提供了新的视角。Mamba 已在自然语言处理、音频、基因组学、计算机视觉等多个领域展示了其强大的应用潜力和卓越性能，预示着其在未来人工智能序列建模中将扮演越来越重要的角色。

## 1. 引言：Mamba 在序列建模领域的崛起

### 1.1. 序列建模的演进格局

序列建模一直是机器学习的核心课题，旨在理解和生成各种序列数据，如文本、语音、时间序列等。近年来，Transformer 架构凭借其强大的自注意力机制，在诸多序列建模任务中取得了主导地位 $^{1}$。然而，Transformer 的核心优势自注意力机制也带来了其固有的局限性：计算复杂度随序列长度二次方增长 $^{5}$。这使得 Transformer 在处理日益增长的超长序列(例如，百万级 token)时，面临着巨大的计算和内存挑战。为了解决这一瓶颈，研究界一直在积极探索具有亚二次方时间复杂度的替代架构 $^{9}$。

### 1.2. 状态空间模型(SSM)作为基础

状态空间模型(SSM)为动态系统分析提供了一个强大的框架，其核心思想是通过潜在状态的演化来捕捉系统的时间动态，这些状态进而决定了观测值 $^{11}$。SSM 通过将历史信息压缩到一个固定大小的状态中来对序列进行建模 $^{6}$。系统的演化由其当前状态和新的观测共同决定，就是对过去信息的有效压缩 $^{6}$。传统的 SSM 学习方法包括最大似然估计，而在深下，则常采用基于变分自编码器等方法来处理潜变量 $^{11}$。SSM 在历史上已有应用，度学习领域重新受到关注，并被视为一种有潜力的序列建模方法 $^{11}$。

### 1.3. Mamba 简介：范式转换？

Mamba 是一种新兴的基于 SSM 的架构，它在处理序列数据，尤其是长序列时，展现出与 Transformer 相当甚至更优的性能，同时其计算复杂度与序列长度呈线性关系 $^{6}$。Mamba 的核心承诺是：在实现与 Transformer 同等性能水平的同时，具备线性的扩展能力和更快的推理速度 $^{6}$。Mamba 架构由 Albert Gu 和 Tri Dao 在其论文《Mamba:Linear-Time Sequence Modeling with Selective State Spaces》中提出 $^{16}$。该论文的官方 arXiv ID 为 2312.00752，提交于 2023 with Selective State Spaces

with Selective State Spaces Mamba 的出现并非凭空产生，而是先前研究成果与创新思想巧妙结合的产物。它建立在结构化状态空间模型(如 S4)的基础之上 $^{2}$，但通过引入关键的"选择机制"(selection mechanism)$^{2}$，成功克服了先前 SSM 在处理离散的、信息密集的模态(如文本)时所面临的挑战 $^{9}$。这表明 Mamba 不仅仅是对现有技术的渐进式改进，更通过解决先前 SSM 的核心弱点实现了一次质的飞跃。这种成功的融合预示着，未来序列建模领域的突破可能更多地来自于对现有概念的巧妙组合与精炼，而非完全从零开始构建全新的架构。

## 2. 解构 Mamba：架构与核心机制

### 2.1. Mamba 模块：架构概览

与 Transformer 由堆叠的 Transformer 模块构成类似，Mamba 模型也是由一系列堆叠的 Mamba 模块组成 $^{6}$。Mamba 通过将先前 SSM 架构的设计与 Transformer 中的 MLP 模块相结合，形成了一个单一、内聚的 SSM 模块，从而简化了以往深度序列模型的架构 $^{2}$。这种设计取代了 Transformer 中复杂的自注意力机制和 MLP 模块 $^{2}$。

在机器学习的骨干网络中，通常包含两个基本操作：token 间的信息通信(Communication)和 token 内的计算(Computation)$^{6}$。Mamba 采用 SSM 进行通信(取代了注意力机制)，并保留了类似 MLP 的投影进行计算 $^{6}$。一个典型的 Mamba 模块通常包括：一个输入线性投影层、一个一维卷积层(Conv1D)、SiLU 激活函数、选择性状态空间模型(S6)核心以及一个输出投影层。此外，通常还带有一个残差连接和一个门控机制(通过与另一个经过 SiLU 和线性投影处理的分支进行逐元素相乘实现)

Mamba 模型具有多个接口级别，但其核心是封装了选择性 SSM 的 Mamba

下图展示了一个 Mamba 模块的概念框图：

**图 1：Mamba 模块概念框图 $^{6}$**

### 2.2. 选择性状态空间模型(S6)：Mamba 的核心

S6 机制是 Mamba 架构的灵魂所在，它赋予了模型根据输入内容动态调整其行为的能力。

#### 2.2.1. 连续时间公式(A,B,C,D 矩阵)

SSM 将一维输入序列 u(t)映射到一个 N 维潜在状态 h(t),然后再投影回一维输出序列 y(t)$^{12}$。其核心方程组通常表示为：

●状态演化方程：$h^{\prime}(t)=A h(t)+B x(t)^6$  
●输出生成方程：$y(t)=C h(t)+D x(t)^6$  

其中，x(t)是输入，h(t)是隐藏状态，y(t)是输出。各个矩阵的直观解释如下 $^{6}$：  
• A(状态转移矩阵)：描述当前状态如何转变为下一状态，即"如何随时间遗忘状态中不太相关的部分？"  
● B(输入映射矩阵)：将新的输入映射到状态中，即"新输入中应该记住哪些部分？"  
● C(状态到输出映射矩阵)：将状态映射到 SSM 的输出，即"如何利用当前状态做出好的下一个预测？"  
●D(直接输入到输出/跳跃连接矩阵)：描述新输入如何直接传递到输出，即"如何在预测中直接使用新输入？"

#### 2.2.2. 离散化与 $\Delta$(Delta)的作用

由于现实世界的输入通常是离散的，连续时间方程需要通过离散化过程转换为离散时间形式 $^{6}$。Mamba 采用特定的离散化规则(例如零阶保持，Zero-Order Hold-ZOH)将连续参数($\Delta$,A,B)转换为离散参数 $\left(A^{-},B^{-}\right)6$。离散后的状态更新方程可表示为：$h_t=A^{-}h_{t-1}+B^{-}x_t^{12}$。在离散化过程中引入了一个至关重要的可学习参数 $\Delta$(Delta)，它代表步长或"停留时间"(lingertime)$^{6}$。$\Delta$ 控制着模型对当前 token 的关注程度：较大的 $\Delta$ 意味着对当前 token 给予更多关注，而较小的 $\Delta$ 则意味着快速跳过该 token$^{6}$。这使得不同的 SSM 层能够在不同的时间尺度上运作 $^{20}$。

#### 2.2.3. 选择机制：输入依赖的参数化

Mamba 的核心创新在于其"选择机制"，即让 SSM 的参数(特别是通过 B,C 和 $\Delta$ 参数 $A^{-},B^{-},C^{-}$)成为输入 x 的函数 $^{2}$。这使得 Mamba 能够根据当前 token 的内容选择性地传播或遗忘信息，从而解决了传统 SSM(使用固定的 A,B 矩阵)在处理离散和信息密集的模态(如文本)时的不足 $^{6}$。

具体来说，矩阵 B,C 以及离散化参数 $\Delta$ 都变成了时间依赖的(即输入 $xt$ 状态转移矩阵 A 本身通常保持结构化(例如，源自 HiPPO 矩阵)，但其离散化形式 A 会通过依赖于输入的 $\Delta$ 而间接变得输入依赖 $^{25}$。这种输入依赖性是通过对输入 x 进行线性投影来实现的 $^{20}$。例如，选择机制可以包含围绕 B,C 和 $\Delta$ 参数的线性层 $^{20}$。

选择机制带来了诸多益处 $^{2}$：  
- 动态适应不同的输入，以处理各种序列建模任务。  
- 重置状态以清除不相关的历史信息。  

选择性机制可以被理解为一种动态的门控系统。B,C 和 $\Delta$ 的输入依赖特性个复杂的门控机制：B 决定了当前输入的多少信息被"写入"状态，C 决定了状态的多少信息被"读取"到输出，而△则控制着对当前输入的"时间尺度"或"关注度"。这与 LSTMs/GRUs 中的门控机制有相似之处 $^{20}$，但被巧妙地集成到了 SSM 框种动态门控使得 Mamba 能够克服传统 LTI SSM 的"上下文盲点"，实现基于内容的推理 $^{9}$。这揭示了一个重要的设计原则：有效的序列模型需要能够根据内容动态调节信息流的机制，而 Mamba 在 SSM 范式内高效地实现了这一点。

参数 $\Delta$ 在 Mamba 中扮演着双重角色。它不仅仅是一个离散化步长，其输入依赖性 $^{6}$ 使其成为选择机制的核心组成部分。它允许模型学习对一个 token"关注"多久，从而有效地控制当前输入 $xt$ 对隐藏状态 $ht$ 的影响，以及通过 A 控制 $ht$ 本身的演化。较大的 $\Delta$ 会强调当前输入 $xt$ 和近期历史，而较小的 $\Delta$ 则会强调更长期的状态动态或直接跳过当前 token。这种对每个 token 的自适应时间尺度进行学习的细粒度时间控制能力，对于处理复杂序列至关重要，并可能启发其他架构中类似的机制。

## 3. 效率引擎：Mamba 的性能优势

### 3.1. 线性时间复杂度与可扩展性

Mamba 在训练和推理过程中均能实现随序列长度线性扩展(O(L)的时间复杂度)，这与 Transformer 在训练时 O(L2)的二次方复杂度和自回归生成时每个 token O(L)的复杂度形成了鲜明对比 $^{6}$。这一特性使得 Mamba 能够处理 Transformer 因计算量过大而难以应对的超长序列(例如，百万级 token)$^{6}$。

Mamba 的推理速度非常快，据称比同等规模的 Transformer 吞吐量高出 5 倍归推理每一步仅需常数时间，因为它不像 Transformer 那样需要缓存先前的大量元素 $^{13}$。此外，Mamba 的内存使用在训练期间也随序列长度线性增长，并且在推理期间状态大小是固定的(O(1)的空间复杂度)$^{6}$。

### 3.2. 硬件感知并行扫描算法

选择机制使得 Mamba 的 SSM 参数具有输入依赖性，这打破了传统 SSM 所依赖的时间不变性，而时间不变性是传统 SSM 能够使用高效卷积进行并行训练的基础 $^{6}$。题，模型将退化为缓慢的循环计算。

Mamba 通过采用一种针对现代 GPU 优化的硬件感知并行算法(扫描操作)挑战 $^{1}$。该扫描算法包含三个关键组成部分 $^{25}$：  
●并行关联扫描(Parallel Associative Scan)：利用了循环更新本质上是一种"扫描"操作的特性。并行扫描算法可以在并行硬件上以 O(LlogL)甚至更快的速度计算这些操作，从而显著加速原本具有序列依赖性的计算过程 $^{25}$。其关键在于优化向量和矩阵的组织方式，以最大限度地减少内存拷贝并实现并行化 $^{20}$。  
●核融合(KernelFusion)：这项技术旨在优化 GPU 的内存访问。它避免了在全局 GPU 内存(HBM)中物化大型中间状态(如离散化的 A,B 矩阵或隐藏状态 h)。取而代之的是，多个操作被融合成一个单一的计算核。连续参数从 HBM 和扫描操作在高速的片上 SRAM 中执行，最终只有输出结果被写回 HBM 地减少了 I/O 瓶颈，提升了 GPU 利用率。  
●梯度重算(Recomputation of Gradients)：由于核融合避免了在 HBM 中存储中间隐藏状态，而这些状态在反向传播计算梯度时是必需的，因此 Mamba 选择在反向传播过程中按需重新计算这些中间状态 $^{25}$。实践证明，这种重算比从 HBM 存储和读取大型中间状态更为高效。

Mamba 的高效率是算法设计与硬件感知协同优化的结果。其线性扩展能力不仅仅源于 SSM 的数学形式，更关键的是其选择性 SSM 的实现方式。由于选择性而失去的卷积形式所带来的并行优势，通过一种为 GPU 内存层级结构(SRAM 与 HBM)精心设计的复杂扫描算法得以恢复 $^{2}$。选择性提高了模型的建模能力，但破坏了卷积的便利性；而硬件感知的扫描操作则恢复了并行训练的效率。这种相互作用至关重要。这突显了现代机器学习领域的一个趋势：要达到峰值性能，算法不仅需要在理论上高效，还必须针对目标硬件架构进行细致优化。FlashAttention 是体现这一趋势的另一个典型例子 $^{1}$。

## 4. Mamba 的定位：比较与区别

### 4.1. Mamba vs.传统/S4 SSM

传统的 SSM(如 S4 模型)是线性时不变(LTI)系统，意味着它们的 A,B,C,D 参数在所有时间步上都是固定的 $^{2}$。这使得它们可以被高效地计算为卷积操作 $^{6}$。

然而，LTI SSM 的一个主要弱点是它们无法执行基于内容的推理或根据输入进行调整，这使得它们在处理信息密集、离散的模态(如文本)时效果不佳 $^{9}$。由于缺乏选择性，它们可能会丢弃必要的信息 $^{6}$。LTI 模型无法选择性地忽略信息；从卷积的角度来看，一个非常长的卷积核会聚合整个序列的所有信息，这可能引入大量噪声 $^{30}$。

Mamba 的选择机制(输入依赖的 B,C,△)使其能够克服这一局限，通过动态过滤和关注相关信息来提升性能 $^{2}$。Mamba 被认为是 S4 的一个坚实改进，通过综合性任务的对比，突显了传统 SSM 的缺点 $^{10}$。

### 4.2. Mamba vs. Transformer

Transformer 因其自注意力机制而非常有效，该机制允许 token 与所有其他 token 而理论上能够完美回忆上下文信息 $^{1}$。然而，这也导致了 O(N2)的计算复杂度 $^{6}$。将过去的全部信息存储在其 KV 缓存中 $^{6}$。

Mamba 则旨在以线性扩展的方式实现与 Transformer 相当的性能。它更像将历史信息压缩到一个固定大小的状态中 $^{6}$。

在效率方面，Mamba 的推理速度明显更快(高达 5 倍吞吐量)，并且随序列长度线性扩展，而 Transformer 则随着序列长度的增加而呈二次方减慢 $^{6}$。性能上，Mamba 在多种模态上均达到 SOTA 水平，例如 Mamba-3B 模型优于同等规模的 Transformer，并能与两倍规模的 Transformer 相媲美。

关于长程依赖的处理：Mamba 的选择性状态允许它通过过滤不相关信息来潜在地处理非常长的依赖关系 $^{2}$。然而，一些资料表明，与 Transformer 的注意力机制相比，Mamba 的 MLP 模块在捕获长程依赖方面可能不那么有效 $^{31}$，并且其固定大小的状态与 Transformer 理论上无限的 KV 缓存相比，固有地限制了内存容量 $^{1}$。这一点是持续研究和改进的方向(例如 ReMamba)。

在信息流和上下文处理方面，Transformer 将上下文用作具有高保真度的短期记忆；Mamba 则将上下文压缩到其状态中 $^{6}$。Transformer 通常在外部过滤上下文(如 RAG 技术)，而 Mamba 则在内部早期进行过滤。

下表总结了 Mamba 与 Transformer 在关键架构和性能上的差异：

**表 1：Mamba 与 Transformer-关键架构与性能差异**

| 特性 | Mamba | Transformer |
|------|-------|------------|
| 核心机制 | 选择性状态空间模型(SSM) | 自注意力机制(Self-Attention) |
| 序列长度扩展(训练) | 线性(O(L))$^{6}$ | 二次方(O(L2))$^{6}$ |
| 序列长度扩展(推理) | 线性(O(L))，每 token 常数时间 $^{13}$ | 每 token O(L)(自回归)$^{6}$ |
| 内存使用(状态/缓存) | 固定大小状态(O(1)空间) | KV 缓存，随序列长度线性增长(O(L))$^{6}$ |
| 上下文处理 | 内部选择性压缩到状态 | 外部检索增强作为高保真短期记忆 $^{6}$ |
| 主要优势 | 长序列效率高，推理速度快 $^{6}$ | 强大的上下文理解与回忆能力，成熟的生态系统 $^{6}$ |
| 主要挑战 | 固定状态的内存容量限制，长上下文理解的细微差别 $^{10}$ | 长序列的二次方计算瓶颈 |

传统的 RNN 高效但表达能力不足(遗忘过多信息)。Transformer 表达能力强但效率低下(二次方成本)。Mamba 凭借其选择性状态，试图在这一"效率与表达能力"的帕累托前沿上找到一个更优的点 $^{6}$。它旨在保留 Transformer 的大部分效能，同时显著提高效率。选择机制是提升其相对于传统 SSMs/RNNs 表达能力的关键，而硬件感知的扫描算法则是保持其相对于 Transformer 效率的关键。Mamba 代表了一种特定的策略(选择性状态压缩)来应对序列建模中的这一基本权衡。其他架构可能会探索不同的策略。

## 5. Mamba 生态系统：演进与特化

Mamba 架构并非一成不变，而是一个不断发展的家族。多个变体被提出，以解决特定问题或扩展其应用范围。

### 5.1. Mamba-ND：征服多维数据

最初的 Mamba 主要关注一维序列(如文本)$^{5}$。将其扩展到多维数据(如图像、视频、科学数据)并非易事，因为 Mamba 需要特定的数据排序，这与可并行计算的卷积或自注意力机制不同 $^{5}$。

Mamba-ND 通过沿着不同维度交替地"展开"输入数据(遵循固定的行主序)来扩展 Mamba$^{5}$。例如，对于二维数据，可能采用 H+H-W+W-的顺序；对于三维数据，则采用 H+H-W+W-T+T-的顺序 $^{32}$。其架构将一维 SSM 层作为黑箱进行堆叠，并在层与层之间交替序列顺序，而无需对一维 SSM 层本身进行复杂修改 $^{5}$。令人惊讶的是，这种相对简单的设计(交替固定顺序)在性能上优于更复杂的多方向策略 $^{5}$。

在性能方面，Mamba-ND 在 ImageNet-1K 图像分类、HMDB-51/UCF-101 ERA5 天气预报和 BTCV3D 分割等任务上，通常以更少的参数量取得了与 Transformer 相当甚至更优的性能 $^{5}$。例如，在 ImageNet 上，其准确率比 ViT 高 1.5%，少了 20.7%$^{14}$。

Mamba-ND 的成功表明，在将一维序列模型扩展到 N 维时，巧妙而简单的数据处理路径可能比在核心层内进行过于复杂的架构更改更为有效。这为模型设计提供了一个宝贵的经验：有时最优雅的解决方案并非最复杂的。数据表示/排序与固定处理模块之间的交互可以产生强大的结果。

### 5.2. ReMamba：增强长上下文理解

尽管 Mamba 对长序列处理效率很高，但经验证据表明，与 Transformer 相比，其理解超长上下文的能力可能有限 $^{24}$。类似 RNN 的特性和固定大小的状态可能导致远距离信息的退化 $^{24}$。ReMamba 旨在通过一个两阶段的重前传(re-forward)过程来增强 Mamba 的长上下文理解能力，且额外的推理成本极小 $^{24}$。其技术包括 $^{24}$：  
- 阶段一(选择性压缩)：使用前馈网络(FFN)判断 Mamba 最后一层隐藏状态的重要性，选择重要的状态，并对其进行压缩(例如，基于与最后一个隐藏状态的余弦相似度进行 top-K 选择)，然后用这些压缩表示替换原始 token 嵌入的一部分。  
- 阶段二(选择性适应)：在对压缩序列进行第二次前传时，将重要性得分整合到 Mamba 的选择机制中。对于不太重要(现已被压缩)的片段，会调整其 $\Delta$ 值以减少它们的影响。  

ReMamba 在 LongBench 和 L-Eval 等长上下文基准测试中提升了 Mamba 的性能，使其更接近 Transformer 的水平 $^{24}$。ReMamba 的方法实质上是在 Mamba 的隐式状态压缩之上叠加了一层可学习的、显式的内存管理策略。通过识别并重新优先处理第一次传递中的重要信息，它试图减轻固定大小循环状态在处理超长序列时固有的信息丢失问题。固定状态特性导致 Mamba 可能出现长上下文信息退化，而 ReMamba 的带有压缩和适应的重前传机制通过刷新和重新加权上下文来直接解决这个问题。这为改进其他固定状态循环模型提供了一条潜在路径：采用带有学习的上下文选择和适应的多遍处理。

### 5.3. Mamba-2：状态空间对偶性带来的新飞跃

Mamba-2 的基础是 Dao 和 Gu 提出的"Transformer 即 SSM：通过结构化状态空间对偶性的广义模型和高效算法”(SSD)框架 $^{8}$。该框架揭示了 SSM 与注意力变体之间深层的理论联系。

Mamba-2 采用了简化的架构和一个经过优化的选择性 SSM 核心层，据称比 Mamba-1 快 2-8 倍，同时在性能上仍能与 Transformer 竞争 $^{34}$。源于 SSD 理论的架构更改包括：将所有数据依赖的投影移至模块块的开头并行发生，内部 SSM 层采用 SSD 如，在 HybriDNA(使用 Mamba-2 模块)中，SSD 层将 A 矩阵简化为 A=al 高效率 $^{38}$。

性能方面，在 Chinchilla 缩放定律的比较中，Mamba-2 在困惑度和实际运行时间上均优于 Mamba-1 和 Transformer++$^{8}$。Hugging Face 的实现细节包括 $^{34}$：支持 torch_forward(无需编译更快)和 cuda_kernels_forward(原始核函数，预填充较慢)；不使用位置编码，但使用 attention_mask；引入 n_groups 参数(例如 Mamba-2 codestral 中为 8)，直观上类似于注意力机制中的 KV 头数量。

下表概述了 Mamba 的主要变体：

**表 2：Mamba 变体概述(Mamba-ND,ReMamba,Mamba-2)**

| 变体 | 核心创新 | 目标问题 | 主要架构变化 | 关键性能亮点 |
|------|----------|----------|--------------|--------------|
| Mamba(基线) | 选择性状态空间(S6)，硬件感知并行扫描 | 长序列建模效率与性能 | 输入依赖的 SSM 参数，高效扫描算法 | 线性扩展，比 Transformer 更快推理 $^{6}$ |
| Mamba-ND | 多维数据交替展开 | 将 Mamba 扩展到图像、视频等多维数据 | 堆叠 1D-SSM 并交替序列顺序 $^{14}$ | 在 ImageNet，ERA5 等多维基准上以更少参数取得 SOTA 或有竞争力的性能 $^{5}$ |
| ReMamba | 两阶段重前传(选择性压缩与适应) | 改善 Mamba 的超长上下文理解能力 | 识别并压缩重要上下文，在重前传中调整选择机制 $^{24}$ | 提升在 LongBench，L-Eval 等长上下文基准上的表现 $^{24}$ |
| Mamba-2 | 基于状态空间对偶性(SSD)的架构优化 | 进一步提升 Mamba 的速度和性能 | 简化的选择性 SSM 核心层，数据依赖投影并行化，SSD 原理应用 $^{8}$ | 比 Mamba-1 快 2-8 倍，帕累托优于 Mamba-1 和 Transformer++$^{8}$ |

Mamba 架构并非一成不变，而是一个不断演进的家族。这些变体分别解决了 Mamba 的特定局限或扩展了其能力范围，展示了 Mamba 作为一个持续发展的研究方向，而非单一的终点解决方案。

## 6. 揭示深层联系：高级理论洞察

对 Mamba 的研究不仅停留在架构和性能层面，还深入到其与现有主流模型(尤其是 Transformer)的理论联系。

### 6.1. Mamba 模型的隐藏注意力

Ali、Zimerman 和 Wolf 的研究(arXiv:2403.01590)提出，尽管 Mamba 没有像 Transformer 那样显式的注意力机制，但其选择性 SSM 层可以被视为一种注意力驱动的模型 $^{29}$。这种观点是通过使用一个"数据控制线性算子"(data-control linear operator)对 Mamba 的计算进行重新表述，从而揭示了 Mamba 层内部的"隐藏注意力矩阵"(hidden attention matrices)$^{29}$。这里的核心思想是，如果一个线性算子 Y=aX 中的 a 是输入 X 的函数，那么这个算子就是数据依赖的，可以被视为一种注意力形式 $^{42}$。据称，Mamba 模型产生的"注意力矩阵"数量比 Transformer 多三个数量级 $^{29}$。

这一视角的意义重大 $^{29}$：  
- 可解释性/可理解性：它允许将 Transformer 领域中成熟的基于注意力的可解释性技术(如注意力分配、归因方法)应用于 Mamba 模型，而 Mamba 传统上缺乏此类工具。  
- 调试与信任：更好地理解信息流有助于调试 Mamba 模型，并增强其在需要可解释性的敏感领域中的应用。  
- 理论比较：为比较 Mamba 和 Transformer 的内部机制及表示提供了一个直接的框架。  

后续研究 $^{41}$ 对这种隐式自注意力的概念进行了改进，旨在将 Mamba 模块的更多组件(如 Conv1D、门控机制)纳入考虑，而不仅仅是 S6 层，以获得更准确的类注意力表示。

"隐藏注意力"的研究表明，注意力的核心功能基于内容动态加权和混合信息-可以通过其他机制隐式实现，而不仅限于显式的自注意力模块。Mamba 的选择性 SSM 通过其输入依赖的操作，在功能上达到了类似的效果。Mamba 的输入依赖参数(B,C,△)创建了数据依赖的线性算子，这些算子有效地起到了类似注意力矩阵的作用，决定了信息如何混合。这可能导致对神经网络中"注意力"的更广义理解，即关注功能等效性而非特定的架构模式。这也为解释以前被认为比 Transformer 更"黑箱"的模型开辟了新途径。



### 6.2. 状态空间对偶性（SSD）：连接 Transformer

Dao 和 Gu 在其论文《Transformer 即 SSM：通过结构化状态空间对偶性的广义模型和高效算法》中提出了 **状态空间对偶性（SSD）** 理论框架，揭示了状态空间模型（SSM，如 Mamba）与 Transformer 的自注意力机制之间的深层数学联系 \[^{8}\]。这种联系通过结构化矩阵（特别是**半可分矩阵**）的抽象建立：序列模型可被统一视为矩阵变换或张量收缩操作。在此框架下，SSM 生成的序列混合矩阵本质上是**结构化、半可分的** \[^{43}\]。

#### **SSD 的核心理论贡献**
1. **数学等价性**：  
   SSD 证明 SSM 的循环计算与自注意力的加权求和可通过**半可分矩阵分解**相互表达。SSM 的离散状态更新方程 \( h_t = \overline{A}h_{t-1} + \overline{B}x_t \) 可重写为隐式注意力形式：  
   \[
   y_t = C \left( \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \overline{A}_j \right) \overline{B}_k x_k \right)
   \]
   其中 \(\prod_{j=k+1}^{t} \overline{A}_j\) 构成一个下三角半可分矩阵，其秩由状态维度 \(N\) 决定。这等价于自注意力中通过 \(QK^T\) 生成的稀疏注意力模式 \[^{8, 43}\]。

2. **计算统一性**：  
   SSD 表明 SSM 和注意力机制均可通过**块状矩阵乘法**高效实现。SSM 的扫描操作（原需 \(O(L \log L)\) 复杂度）可转化为 \(O(LN^2)\) 的批处理矩阵乘法（\(L\) 为序列长，\(N\) 为状态维度），与现代硬件高度契合 \[^{43}\]。

#### **SSD 对 Mamba-2 的架构革新**
SSD 理论直接驱动了 Mamba-2 的设计优化 \[^{8, 34}\]：  
1. **数据依赖投影的并行化**：  
   将输入相关的参数投影（生成 \(B, C, \Delta\) 的线性层）移至模块入口处**并行计算**，而非嵌入循环步骤。此举消除了顺序依赖，提升 GPU 利用率 \[^{34}\]。  
2. **简化的 SSD 层**：  
   采用 SSD 推导的简化状态转移矩阵 \(A = \alpha I\)（标量对角矩阵），大幅降低离散化计算开销。例如，在 HybriDNA 模型中，SSD 层将矩阵乘法缩减为逐元素运算 \[^{38}\]。  
3. **分组机制（`n_groups`）**：  
   引入类似注意力中 KV 头的分组策略（如 Mamba-2 codestral 设 8 组），允许不同状态子空间独立建模多样化模式 \[^{34}\]。  

**性能提升**：上述优化使 Mamba-2 的 SSM 核心比 Mamba-1 **快 2–8 倍**，同时在语言建模任务（如 Chinchilla 缩放测试）中保持更优的困惑度-时间帕累托前沿 \[^{8, 34}\]。

#### **SSD 的广义意义**
1. **跨架构技术迁移**：  
   SSD 为 Transformer 的优化技术（如 FlashAttention）迁移至 SSM 提供理论基础，反之亦然 \[^{8}\]。例如，Mamba-2 的核融合技术受 FlashAttention 的 SRAM 内存管理启发 \[^{1}\]。  
2. **混合模型设计**：  
   对偶性催生新型架构（如 **TransMamba** \[^{4}\]），在浅层用 Mamba 高效压缩长序列，深层用注意力精调局部交互，兼顾效率与表达能力 \[^{3}\]。  
3. **理论新视角**：  
   SSD 将序列建模统一为**结构化矩阵逼近问题**，为稀疏注意力、循环网络等提供新的分析工具 \[^{43}\]。  

> **总结**：SSD 不仅解释 Mamba 与 Transformer 的隐性关联，更推动算法-硬件协同设计。Mamba-2 的成功印证：揭示架构间的深层数学对偶性，能解锁突破性的工程优化 \[^{8}\]。

### Works cited
1.	RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers - arXiv, accessed May 21, 2025, https://arxiv.org/html/2403.18276v1
2.	An Introduction to the Mamba LLM Architecture: A New Paradigm in Machine Learning, accessed May 21, 2025, https://www.datacamp.com/tutorial/introduction-to-the-mamba-llm-architecture
3.	A hybrid model based on transformer and Mamba for enhanced sequence modeling - PMC, accessed May 21, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11968869/
4.	TransMamba: Flexibly Switching between Transformer and Mamba - arXiv, accessed May 21, 2025, https://arxiv.org/html/2503.24067v1
5.	Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data - arXiv, accessed May 21, 2025, https://arxiv.org/html/2402.05892v4
6.	Mamba Explained - The Gradient, accessed May 21, 2025, https://thegradient.pub/mamba-explained/
7.	The Mamba Architecture: Superior to Transformers in LLMs - Jon Krohn, accessed May 21, 2025, https://www.jonkrohn.com/posts/2024/2/16/the-mamba-architecture-superior-to-transformers-in-llms
8.	Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality - OpenReview, accessed May 21, 2025, https://openreview.net/pdf/54bf495d93336f1f195f264c1b6c2805169b3492.pdf
9.	arxiv.org, accessed May 21, 2025, https://arxiv.org/abs/2312.00752
10.	Mamba: Linear-Time Sequence Modeling with Selective State Spaces - OpenReview, accessed May 21, 2025, https://openreview.net/forum?id=tEYskw1VY2
11.	arxiv.org, accessed May 21, 2025, https://arxiv.org/abs/2412.11211
12.	Mamba Models a possible replacement for Transformers? - SciPy Proceedings, accessed May 21, 2025, https://proceedings.scipy.org/articles/XHDR4700
13.	Mamba: Linear-Time Sequence Modeling with Selective State Spaces - arXiv, accessed May 21, 2025, https://arxiv.org/pdf/2312.00752
14.	Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data - arXiv, accessed May 21, 2025, https://arxiv.org/html/2402.05892v1
15.	Princeton and CMU Push AI Boundaries with the Mamba Sequence Model - HackerNoon, accessed May 21, 2025, https://hackernoon.com/princeton-and-cmu-push-ai-boundaries-with-the-mamba-sequence-model
16.	Mamba - Hugging Face, accessed May 21, 2025, https://huggingface.co/docs/transformers/en/model_doc/mamba
17.	PeaBrane/mamba-tiny: Simple, minimal implementation of the Mamba SSM in one pytorch file. Using logcumsumexp (Heisen sequence). - GitHub, accessed May 21, 2025, https://github.com/PeaBrane/mamba-tiny
18.	Mamba: Linear-Time Sequence Modeling with Selective State Spaces - DISCO, accessed May 21, 2025, https://disco.ethz.ch/courses/fs24/seminar/talks/19_03_Mamba.pdf
19.	QMamba: Quantum Selective State Space Models for Text Generation - SciTePress, accessed May 21, 2025, https://www.scitepress.org/Papers/2025/133783/133783.pdf
20.	Mamba: Linear-Time Sequence Modeling with Selective State Spaces - Arxiv Dives, accessed May 21, 2025, https://www.oxen.ai/blog/mamba-linear-time-sequence-modeling-with-selective-state-spaces-arxiv-dives
21.	arXiv:2502.15612v2 [cs.CL] 24 Feb 2025, accessed May 21, 2025, https://arxiv.org/pdf/2502.15612?
22.	Figure 1: The Mamba block architecture is constructed based on SSM's... - ResearchGate, accessed May 21, 2025, https://www.researchgate.net/figure/The-Mamba-block-architecture-is-constructed-based-on-SSMs-mathematical-formulation-in_fig1_383792530
23.	state-spaces/mamba: Mamba SSM architecture - GitHub, accessed May 21, 2025, https://github.com/state-spaces/mamba
24.	openreview.net, accessed May 21, 2025, https://openreview.net/pdf?id=RMjyNzYv2K
25.	Here Comes Mamba: The Selective State Space Model | Towards Data Science, accessed May 21, 2025, https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451/
26.	Here Comes Mamba: The Selective State Space Model | Towards ..., accessed May 21, 2025, https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451
27.	Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient | OpenReview, accessed May 21, 2025, https://openreview.net/forum?id=7XIkRgYjK3
28.	(PDF) Bio2Token: All-atom tokenization of any biomolecular structure with Mamba, accessed May 21, 2025, https://www.researchgate.net/publication/385291570_Bio2Token_All-atom_tokenization_of_any_biomolecular_structure_with_Mamba
29.	The Hidden Attention of Mamba Models - arXiv, accessed May 21, 2025, https://arxiv.org/html/2403.01590v2
30.	Mamba's Performance in DNA, Audio, and Speed Benchmarks | HackerNoon, accessed May 21, 2025, https://hackernoon.com/mambas-performance-in-dna-audio-and-speed-benchmarks
31.	Decoding Mamba's Potential: Strengths, Weaknesses, and Where It ..., accessed May 21, 2025, https://dataroots.io/blog/analyzing-mambas-potential-strengths-weaknesses-and-where-it-shines
32.	arxiv.org, accessed May 21, 2025, https://arxiv.org/abs/2402.05892
33.	[2408.15496] ReMamba: Equip Mamba with Effective Long-Sequence Modeling - arXiv, accessed May 21, 2025, https://arxiv.org/abs/2408.15496
34.	Mamba 2 - Hugging Face, accessed May 21, 2025, https://huggingface.co/docs/transformers/model_doc/mamba2
35.	Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality - ResearchGate, accessed May 21, 2025, https://www.researchgate.net/publication/381109265_Transformers_are_SSMs_Generalized_Models_and_Efficient_Algorithms_Through_Structured_State_Space_Duality
36.	Daily Papers - Hugging Face, accessed May 21, 2025, https://huggingface.co/papers?q=State%20Space%20Models%20(SSMs)
37.	Mamba 2 - Hugging Face, accessed May 21, 2025, https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/mamba2
38.	HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model, accessed May 21, 2025, https://www.researchgate.net/publication/389090976_HybriDNA_A_Hybrid_Transformer-Mamba2_Long-Range_DNA_Language_Model
39.	Mamba - Hugging Face, accessed May 21, 2025, https://huggingface.co/docs/transformers/model_doc/mamba
40.	arxiv.org, accessed May 21, 2025, https://arxiv.org/html/2403.01590v1
41.	Explaining Modern Gated-Linear RNNs via A Unified Implicit Attention Formulation - arXiv, accessed May 21, 2025, https://arxiv.org/html/2405.16504v2
42.	A Unified Implicit Attention Formulation for Gated-Linear Recurrent Sequence Models - arXiv, accessed May 21, 2025, https://arxiv.org/html/2405.16504v1
43.	ICML 2024: Paper Review #1 - G-Research, accessed May 21, 2025, https://www.gresearch.com/news/icml-2024-paper-review-1/
44.	HYBRIDNA: A HYBRID TRANSFORMER-MAMBA2 LONG-RANGE DNA LANGUAGE MODEL - OpenReview, accessed May 21, 2025, https://openreview.net/pdf/1c165390594cbfbc23fd1739192494ffee9a2d2c.pdf
45.	Wave-U-Mamba: An End-To-End Framework For High-Quality And Efficient Speech Super Resolution - arXiv, accessed May 21, 2025, https://arxiv.org/html/2409.09337v3
46.	Wave-U-Mamba: An End-To-End Framework For High-Quality And Efficient Speech Super Resolution - arXiv, accessed May 21, 2025, https://arxiv.org/html/2409.09337v2
47.	Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model - Hugging Face, accessed May 21, 2025, https://huggingface.co/blog/mikelabs/vision-mamba-efficient-visual-representation-learn
48.	walking-shadow/Official_Remote_Sensing_Mamba: Official code of Remote Sensing Mamba - GitHub, accessed May 21, 2025, https://github.com/walking-shadow/Official_Remote_Sensing_Mamba
49.	Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling - arXiv, accessed May 21, 2025, https://arxiv.org/html/2406.07522v1
50.	DEMYSTIFYING THE TOKEN DYNAMICS OF DEEP SELECTIVE STATE SPACE MODELS - OpenReview, accessed May 21, 2025, https://openreview.net/pdf/5b044a47c58e6589d287df715519c764795be225.pdf
51.	[2411.15638] Learning state and proposal dynamics in state-space models using differentiable particle filters and neural networks - arXiv, accessed May 21, 2025, https://arxiv.org/abs/2411.15638
52.	Meta-Black-Box-Optimization through Offline Q-function Learning - ResearchGate, accessed May 21, 2025, https://www.researchgate.net/publication/391462077_Meta-Black-Box-Optimization_through_Offline_Q-function_Learning
