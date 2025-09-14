# 蒸馏技术简介

在大模型技术快速发展的背景下，大型语言模型(LLMs)的规模不断扩大，参数量从数亿增长到数千亿，性能也随之提升。然而，**模型并非越大越好**。从实际部署角度考量，大模型所需的庞大的内存使其难以在资源受限设备上运行，推理过程伴随高昂的计算成本；较长的推理时间削弱了用户体验，也无法适应需要实时响应的场景。



这些限制凸显了模型蒸馏技术的必要性和合理性。模型蒸馏通过"教师-学生"范式，将大型模型的知识转移到更小的模型中，在保持可接受性能的前提下，显著减小模型体积。这种技术实现了推理速度的提升、部署成本的降低和资源占用的优化，使 AI 应用能够在边缘设备上高效运行。蒸馏后的模型不仅满足实际部署需求，还能在性能与资源消耗之间取得理想平衡，为 AI 技术的普及应用提供了切实可行的解决方案。

下图展示了知识蒸馏在大型语言模型背景下发挥的这三个关键作用。

![图片描述](./images/3_role.png)

大语言模型的知识蒸馏过程类似于把能力更强的教师的"知识"传授给学生，其中学生模型（如参数量更小的开源大语言模型）学习模仿教师模型（如参数量更大的闭源大语言模型）的性能特征。

1. **增强**：与传统知识蒸馏算法相比，数据增强已成为实现大型语言模型知识蒸馏的主流范式，即使用少量知识种子来提示大型语言模型生成针对特定技能或领域的更多数据。

2. **压缩**：知识蒸馏对大语言模型还有压缩作用，使模型在不显著损失性能的情况下更加高效。

3. **自我提升**：近期，将开源大型语言模型作为教师用于自我改进的策略已成为一种有前景的方法，能够显著提升模型能力。比如自我奖励语言模型，其通过"LLM 作为评判者"的提示方式，让语言模型本身在训练过程中提供自身的奖励信号。模型本身带来的超人类的反馈可以提供充分的训练信息，使模型不受限于人类表现水平。

![图片描述](./images/KD_pipeline.png)

大型语言模型(LLM)的蒸馏工作流是一个结构化过程，旨在将知识从复杂的教师模型转移到更简单的学生模型。整个蒸馏过程可分为四个关键阶段：

1. 通过精心设计的指令引导教师模型专注于特定技能或领域；

2. 提供种子知识作为输入，促使教师模型生成更详细的输出；

3. 教师模型根据种子知识和引导指令生成问答对话或解释性内容作为蒸馏知识；

4. 使用这些生成的知识示例和特定学习目标训练学生模型，通过最小化损失函数使学生模型逐步获得类似教师模型的能力。

这一工作流确保了知识的有效传递，使更轻量级的模型也能展现出类似高级模型的专业能力。

下面是对两种蒸馏的典型算法——GKD（Generalized Knowledge Distillation）方法和 MiniLLM 方法的详细介绍。

# GKD：广义知识蒸馏

**简介：** GKD（Generalized Knowledge Distillation，广义知识蒸馏）由 Google DeepMind 团队提出，并于 2023 年 6 月在 arXiv 发布（后被 ICLR 2024 接受）。该方法针对自回归序列生成模型的蒸馏提出，重点解决学生训练阶段和推理阶段的分布差异问题。

## 现有问题：

作者观察到，**传统蒸馏方法在训练时通常使用教师模型生成的固定样本或现成数据集，无法反映学生模型在推理时真实的输出分布**，这会导致训练阶段和推理阶段输出序列分布不匹配（train-inference mismatch）。此外，由于学生模型通常比教师小，学生本身可能

1. **普通方法的局限性**：
   
   - **Seq KD**(Sequence-Level Knowledge Distillation)：从教师模型生成固定的输出序列集合来学习，可以看作是基于教师生成输出的 SFT（Supervised Fine-Tune）方法。这种方法计算成本高昂，需要教师模型生成大量样本，且学生只学习"最终答案"而非决策过程中的概率分布。
   
   - **SKD**(Supervised Knowledge Distillation)：使用固定的数据集，由教师模型为每个词元分配概率，学生可以学习教师生成的概率。
     
     这两种方法都存在训练阶段和推理阶段输出序列分布不匹配（train-inference mismatch）的问题。

2. **分布不匹配问题**： 使用固定数据集进行蒸馏时，学生模型在训练和推理阶段面临不同的分布：
   
   - **训练时**：学生模型学习基于"完美"历史预测下一个词（因为它使用教师提供的数据）
   - **推理时**：学生模型必须基于自己生成的历史（可能已经偏离最优路径）来预测下一个词
   
   如下图所示，学生对教师轨迹进行学习，但学生学习策略与专家轨迹之间可能出现不匹配问题，学生没有关于如何恢复的数据，出错后不知该如何纠正回到正确的轨迹上。这种不匹配会导致误差累积，被称为"模仿学习中的分布偏移问题"。
   
   ![mismatch](./images/mismatch.png)

3. **KL 散度优化的局限性**：
   
   - 传统蒸馏通常试图最小化教师和学生分布之间的前向 KL 散度(KL($p_{teacher}$||$p_{student}$))
   - 当学生模型表达能力不足时（例如参数更少），它可能无法完全拟合复杂的教师分布
   - 这会导致学生模型生成的样本质量较差，与教师模型的分布不一致

为此，提出广义知识蒸馏（GKD）来缓解上述问题。通过**让学生在训练阶段使用自身生成的序列并接受教师反馈**的方式，缓解了分布不匹配，同时允许使用灵活的损失函数来克服表达能力的差距。

## 技术细节与原理：

GKD 将蒸馏过程推广为包含训练阶段采样的自监督（on-policy）的框架，并引入可调的混合策略与损失度量。其主要思想可归纳为以下几点：

### on-policy 框架

在 GKD 中，学生模型不仅在训练中处理固定教师样本，同时也会**生成自己的序列**用于训练，即采用自监督（on-policy）框架，这样可以实现训练-推理的分布匹配。这种 on-policy 的具体做法是，对于每个提示词，学生先用当前策略生成一批候选回答（这些回答来自学生自身分布），然后教师模型对这些答案打分（计算概率）。通过在训练损失中加入对学生自生成序列的反馈，学生在训练时就能见到和其推理时分布更接近的样本，从而减轻了训练与推理的不一致性。即：

1. On-policy 数据：从学生模型中采样输出序列

2. 反馈：使用教师模型对学生生成的样本进行推理以获取 logits

3. 监督训练：最小化学生和教师模型在 token 层面 logits 之间的差异（如 KL 散度）
   
   ![teacher_drive](./images/teacher_drive.png)

### 灵活的损失函数：

 GKD 采用**广义 Jensen–Shannon 散度（generalized JSD）**来度量学生分布与教师分布之间的差异，并引入参数来插值不同散度。

基于 KL 的散度是衡量两个概率分布相似性的测量方法，其中 KL 散度（Kullback-Leibler 散度）是一种常用的度量。两个离散分布 P(C) 和 Q(C) 之间的 KL 散度定义为：

$$D_{KL}(P||Q) = \sum_{c\in C} P(c) \log \frac{P(c)}{Q(c)}$$

KL 散度不具有对称性：

$$D_{KL}(P||Q) \neq D_{KL}(Q||P)$$

因此，我们将 $D_{KL}(P||Q)$ 称为 P 和 Q 之间的前向 KL 散度，而 $D_{KL}(Q||P)$ 称为反向 KL 散度。在监督学习中，基于经验数据分布的前向 KL 对应于我们优化的最大似然。当使用分布 $Q_θ(C)$ 近似 P(C) 时，如果模型容量不匹配，最小化反向和前向 KL 分别会导致均值寻找和模式搜索行为。

> ## 前向 KL 散度与反向 KL 散度的行为差异
> 
> 下表展示了前向 KL 散度和反向 KL 散度，以及 JS 散度的计算公式。
> 
> ![图片描述](./images/Divergence.png)
> 
> 如下图所示我们展示了在最小化前向 KL 散度(\text{KL}(P\|Q_\theta))和反向 KL 散度(\text{KL}(Q_\theta\|P)时，单峰高斯分布 Q_θ相对于混合分布 P 的学习结果。
> 
> | 特征       | 反向 KL 散度 ($\text{KL}(Q_\theta\|P)$) | 前向 KL 散度 ($\text{KL}(P\|Q_\theta)$) |
> | -------- | --------------------------------- | --------------------------------- |
> | **行为特性** | 模式搜索 (Mode-seeking)               | 模式覆盖 (Mode-covering)              |
> | **主要特点** | 强制要求 $Q_θ$ 在 $P$ 为零的地方也为零             | 确保在 P 有概率质量的地方，$Q_θ$ 也必须分配概率质量       |
> | **结果**   | $Q_θ$ 集中在 $P$ 的某一个模式上                | $Q_θ$ 尝试覆盖 $P$ 的所有模式                 |
> | **倾向性**  | 倾向于"忽略"P 中的某些模式，专注于单个模式            | 可能会在真实分布模式之间的低概率区域分配过多概率质量        |
> | **适用场景** | 当需要精确捕捉单个模式时                      | 当需要覆盖所有可能模式时                      |
> 
> 下图所示的例子是用单峰高斯 $Q_θ$ 拟合多峰分布 $P$，可以观察到，当模型容量不匹配时：
> 
> - 反向 KL 会选择一个模式并精确拟合
> - 前向 KL 会尝试覆盖所有模式，但可能导致在实际上不存在数据的区域产生概率质量

> ![mode_seeking](./images/mode_seeking.png)

JSD(β) 使用有界系数 0 < β < 1 在前向和反向 KL 之间进行插值。

$$D_{JSD(β)}(P||Q) = βD_{KL}(P||βP + (1-β)Q) + (1-β)D_{KL}(Q||βP + (1-β)Q)$$

将 JSD 表示为参数 $\beta$ 插值：$\beta=0$ 时近似前向 KL，$\beta=1$ 时近似逆向 KL。广义 JSD 的引入增加了灵活性，使其能够适应不同权重分配的应用场景。例如，可以让损失从正向 KL 过渡到逆向 KL，控制学生对不同分布区域的关注程度。这种灵活性使得当学生容量不足以完全模仿教师时，可以调整散度侧重学生更容易生成的区域。

### 统一框架：

 GKD 实际上将传统的离线蒸馏（off-policy KD）和在线蒸馏（on-policy KD）纳入一个框架。通过调节 $\lambda$ 和 $\beta$，可以自由切换和混合这两种极端，以及介于它们之间的“混合策略 KD（mixed-policy KD）”。当 $\lambda=0$ 且采用学生对教师输出的交叉熵时，GKD 回退到标准的教师生成样本训练，等同于普通的有监督序列级蒸馏；当 $\lambda=1$ 且使用全局 JSD 时，完全依赖学生自生成数据，它就是纯粹的 on-policy 蒸馏。这样的设计使研究者能够按需选取最适合任务的数据策略和发散度，比如 GKD 的作者发现在多个任务上 on-policy 数据（高 $\lambda$）往往获得更好效果。

如表 Algorithm 1 所示，在这个广义知识蒸馏(GKD)算法中，`λ`是一个超参数，取值范围在[0, 1]之间，用于控制学生模型的数据采样策略：`λ`决定了算法在每一步迭代中使用哪种类型的数据，当随机生成的值`u ≤ λ`时，算法使用学生模型自己生成的输出，当`u > λ`时，算法使用原始数据集中的输入-输出对。它实际上控制了自生成数据与原始数据的比例，`λ`值越大，算法越倾向于使用学生模型生成的输出。`λ`值越小，算法越倾向于使用原始标签数据。这种机制允许学生模型在学习过程中既探索自己生成的知识，又利用真实数据的监督信号。

![Algorithm](./images/Algorithm.png)

### 与强化学习结合：

GKD 天生支持与 RLHF 相结合。在生成式任务中，除了利用 KL 散度作为训 练目标，还可以在同一个框架中引入额外的序列级奖励（例如任务指标或人类反馈分数）。GKD 能无缝集成这类奖励优化过程，在蒸馏大型教师的同时优化不可微目标。这种设计允许蒸馏过程兼顾模型对任务指标的优化，使学生模型更适应实际应用需求。

在某些任务中，从教师模型蒸馏可能只是为我们的主要目标提供一个代理，而这个目标也可能是不可微分的。我们可以通过强化学习(RL)直接优化这个目标。在线策略 GKD 很容易与基于人类反馈(RLHF)或 AI 反馈(RLAIF)的强化学习微调相结合，因为它只需要学生模型的输出样本。如果想要优化学生策略以获得标量奖励 r，同时保持与教师策略的接近度，那么我们可以使用这个正则化的 RL 微调目标函数：

$$E_{x\sim X} \left[ (1 - \alpha) E_{y\sim p^S_\theta(\cdot|x)}[r(y)] - \alpha E_{y\sim p^S(\cdot|x)} D(p^T\parallel p^S_\theta)(y|x) \right]$$

其中第一项是 RL 目标，第二项是泛化 On-policy 蒸馏，α ∈ [0,1]控制蒸馏损失相对于 RL 目标的强度。当α=1 时，将只执行蒸馏。

上述目标函数允许我们在最大化奖励的同时，通过蒸馏改进模型的其他能力，这可能减少"对齐税"——即在将语言模型与人类偏好对齐时通常导致的模型通用能力下降。

实现 On-Policy GKD 的可以归纳为以下三个步骤：

1. 使用你喜欢的 RLxF 框架（强化学习框架）

2. 关闭奖励最大化项（图中红色箭头指向的部分）

3. 将参考 SFT 策略替换为教师策略（改变不同方法的分配策略）
   
   ![LOSS](./images/LOSS.png)

实现了带有反向 KL 散度的 on-policy GKD。可以将这里的反向 KL 替换为其他 token 级别的 f-divergence（如 JSD、Jeffreys 散度）以获得最佳效果。

Hugging Face 官方文档对 GKD 的关键点总结为：**GKD 在自回归序列模型中通过在训练时使用学生自生成数据来解决分布不匹配问题；同时它使用通用的 JSD 损失来灵活选择不同的散度度量**。在具体实现时，GKD Trainer 将上述 $\lambda$、$\beta$ 等超参数暴露给用户，方便调节蒸馏强度、混合策略和散度插值。

**创新点与优势：** GKD 的核心创新在于 **“泛化”蒸馏目标**——它将学生蒸馏训练过程从传统固定样本学习，扩展到了在线自采样学习。与 Seq KD 相比，GKD 不再局限于教师事先生成的一套参考答案，而是让学生主动探索输出空间并向学生自身的弱点学习。这一策略有效缓解了训练-推理不一致现象，使得学生训练时更加贴近推理环境。此外，通过引入可调的广义 JSD，GKD 提供了按需控制散度的能力，当学生表达力有限时可侧重逆向 KL，以让学生更多关注教师认为“可能”的输出区域。

实验证明，GKD 在多个自然语言生成任务（摘要、翻译、算术推理）上的表现都超过了常用的蒸馏基线。例如，GKD 在抽象摘要和机器翻译任务上取得了比传统 KD 更低的困惑度和更高的 Rouge 分数，并在算术推理等高难度任务中也显示出更高的一致性和准确性。官方文档和论文都指出，**整合 RLHF 后的 GKD（带有附加奖励）能进一步改善模型输出的任务合规性**，例如在文本摘要中降低幻觉风险。另外，GKD 框架具有高度可调性，用户可以根据具体任务需求灵活设置 $\lambda$ 和 $\beta$，这使得它能在不同规模和类型的模型之间迁移应用。Hugging Face 的实践经验也表明，GKD 在使用领域大模型（如 Qwen 系列）时，同样能在不同 $\lambda$ 和 $\beta$ 配置下获得稳定效果。

尽管 GKD 显示出显著优势，它仍有一些限制。首先，GKD 需要在训练过程中频繁地进行模型采样和评估，比传统 KD 方法计算更昂贵，需要更多训练时间和算力。其次，参数调节较为复杂：$\lambda$、$\beta$ 等超参数对最终效果影响巨大，不同下游任务最优配置可能差异显著，需要经验或额外调优。此外，GKD 的效果依赖于教师模型的质量和学生的初始化策略（一般学生先要经过 SFT 预训练以产出合理输出），对于非常低资源场景或超大规模学生可能面临收敛困难。最后，从研究层面看，GKD 目前主要在语言生成领域验证，对视觉、表格、代码等其它序列型输出的蒸馏应用尚不明确，拓展应用领域是未来需要探索的方向之一。

## 实验效果：

### GKD 方法对于模型应对不同任务的性能提升

将 GKD 与不同学生模型大小的 KD 方法进行比较。我们使用经过监督微调训练的 T5 模型作为学生模型。我们使用经过监督微调的 T5-XL(约 30 亿参数)作为教师模型，其性能由水平线表示。监督 KD 和 FT 使用真实输出序列进行训练，而 SeqKD 则在教师生成的输出序列上训练。在线 GKD 在从学生模型采样的输出序列上进行训练。对于 GKD，我们在 WMT（Workshop on Machine Translation）数据集上使用 JSD(0.1)，在其他任务上使用前向 KL 散度。在评估中，我们对 XSum 和 GSM8K 使用贪婪采样，对 WMT 使用束搜索。

![tasks_compare](./images/tasks_compare.png)

GKD 统一了自回归语言模型的一些现有知识蒸馏方法，同时实例化了新的在线策略方法，这些方法大大优于其他基本的方法。在从在线 GKD 获得的对初始学生模型的性能提升方面，平均在不同大小的 T5 学生模型上，与基准 KD 方法相比（如上图），我们在总结任务上看到相对提升 1.1 倍，在机器翻译上提升 0.7 倍，在算术推理任务上提升 0.9 倍。

### On-Policy 蒸馏的有效性和效率

通过比较蒸馏与直接强化学习的性能和计算成本（以 GPU 小时计）来评估 on-policy 蒸馏的有效性和效率，两者都从相同的 off-policy 蒸馏 Qwen3-8B 检查点开始。为简单起见，我们在这个比较中仅关注数学和代码相关的查询。

![table21](./images/table21.png)

以上数据表明，On-policy 蒸馏实现了显著优于强化学习的性能，同时仅需约 1/10 的 GPU 小时。此外，从教师 logits 进行蒸馏使学生模型能够扩展其探索空间并增强其推理潜力，这一点从蒸馏后 AIME'24 和 AIME'25 基准测试的 pass@64（模型在 64 次尝试中至少有 1 次生成正确答案的概率）分数相比于初始检查点的改善中得到证明。On-policy 蒸馏在显著降低计算资源的同时，却获得了全面超越强化学习的性能。

### 散度对性能和多样性的影响

使用不同散度的在线策略 GKD，我们通过改变采样温度==（补定义介绍和散度公式）==来评估蒸馏后学生模型的生成质量和多样性之间的权衡。我们使用 Self-BLEU 来量化多样性，其中 100 分表示确定性输出，0 分表示最大多样性。

![trade_off](./images/trade_off.png)

从前向 KL 过渡到反向 KL，再到广义 JSD，会导致多样性降低，这归因于散度的增强模式搜索特性。模式搜索型散度通常能产生更优质的输出，特别是在高温度(γ = 1)时。降低温度会减少多样性，同时缩小不同散度之间的性能差异。

### On-policy GKD 与 RLAIF 在总结任务中的性能权衡

下面是关于的一个简单但有力的例子证明了 RLHF/RLAIF 对 GKD 方法的改进作用。只需要对损失函数加入教师模型的部分进行正则化，下面是这个正则化的 RL 微调目标函数：

$$E_{x\sim X} \left[ (1 - \alpha) E_{y\sim p^S_\theta(\cdot|x)}[r(y)] - \alpha E_{y\sim p^S(\cdot|x)} D(p^T\parallel p^S_\theta)(y|x) \right]$$

下图展示了在 XSum 数据集上奖励最大化与摘要性能之间的权衡。图中结果是相对于原始 T5-base 学生模型的改进。遵循 Roit 等人(2023)的方法，使用来自 T5-XXL NLI 分类器的文本蕴含分数作为奖励。参数α控制基于策略的 GKD 损失函数 JSD(0.9)的强度。

![RLFT](./images/RLFT.png)

随着α的增加，ROUGE-2 分数提高，而事实一致性的改进则减少。作为比较，我们展示了规模大 12 倍的 T5-XL 教师模型的相对性能。RLEF 对应于 Roit 等人(2023)的 RLAIF 方法，其中学生模型向原始学生模型本身而非教师模型进行正则化。基于策略的 GKD + RL 与 RLEF*相比达到了更高的 ROUGE-2 分数，同时与教师模型相比生成了事实一致性更强的摘要。

## 关键代码展示：

    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    
    # Compute log probabilities for student and probabilities for teacher
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    
    # Compute the log of the mixture distribution
    # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
    beta = torch.tensor(beta, dtype=student_log_probs.dtype)
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]),
        dim=0,
    )
    
    # Compute KL divergences using F.kl_div
    # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
    
    # Compute the Generalized Jensen-Shannon Divergence
    jsd = beta * kl_teacher + (1 - beta) * kl_student

这段代码实现了广义 Jensen–Shannon 散度（Generalized JSD）的计算流程，常用于知识蒸馏中衡量学生模型与教师模型预测分布的差异。：

1. 温度缩放 logits（softmax 分布）

2. 得到学生和教师的 log-probs

3. 按权重 β 混合两者得到 pmix​

4. 分别计算 KL(pmix​∥pteacher​) 和 KL(pmix​∥pstudent​)

5. 按权重组合，得到广义 Jensen–Shannon 散度

## 应用与挑战：

GKD 被用于蒸馏各种自回归语言模型任务，比如 Gemma2。GKD 也用于任务无关的指令蒸馏：即学生在没有特定任务数据的情况下，通过教师生成的通用指令-回答对进行训练，提升其对指令的整体响应能力。在工业界，由于 Hugging Face 等开源社区的推动，GKD 已被整合进通用训练流程。例如，Hugging Face 的 `trl` 库提供了 GKDTrainer 接口，开发者可以方便地在自己模型上尝试 GKD 蒸馏。此外，在需要低延迟推理的场景中（如客服机器人或对话助手部署），GKD 提供了一条平衡模型质量与效率的路径：可以利用 GKD 从大模型中提取能力，使小模型在保持较高输出质量的同时大幅降低推理成本。

GKD 技术面临的主要挑战有两个：一是计算负担和超参调优难题。由于需要重复生成序列并计算梯度，GKD 训练时间往往较长，尤其是在面对对大型模型时；二是需要在多样性与精确度间寻求平衡，过度强调 on-policy 采样可能导致学生过拟合于自己易生成的模式，而忽略长尾样本。



==继续介绍 Deepmind 的新论文的最新进展==

## 大语言模型蒸馏与推测解码的结合——OSD

在 KD 的众多变体中，**GKD（Generalized Knowledge Distillation）**是一种混合式的蒸馏框架，GKD 能够兼顾模型在不同场景下的表现，兼具精度和泛化能力，是目前蒸馏方法中的代表性方向。然而，不论是 KD 还是 GKD，它们的共性局限在于：**蒸馏往往是离线的、一次性的**。训练时，学生模型在固定的教师数据分布下学习；一旦蒸馏完成，学生参数就被冻结，不会再随着后续的用户输入而改变。学生模型学到的只是教师在特定训练语料上的“快照”，而非教师在真实应用场景中动态展现出来的全部能力。

这种静态的蒸馏方式有两个问题：一，当用户的查询分布与训练数据存在差异时，学生模型的性能往往迅速下降；二，蒸馏得到的小模型虽然能在推理时替代部分计算，但并不能主动适应新的输入分布。因此，在实际加速大语言模型（LLM）的场景下，它的性能上有一定的局限性。

为了让蒸馏不只停留在训练阶段，而是持续发生在推理过程中，可以使用推测解码方法。推测解码中引入了一个 **draft model（草稿模型）**。它会在用户输入后一次性生成一段候选 token 序列。与此同时，**target model（目标模型）** 并不从头开始逐个生成，而是并行地对这些候选 token 进行验证。如果其中大部分 token 都与目标模型的分布一致，那么这些 token 就可以直接被采纳。如果不一致的话，就由目标模型完成正确的 token 选择。推测解码的效率提升，关键取决于一个指标——**token 接受率 α**。接受率越高，说明草稿模型预测得越准，目标模型需要重新生成的 token 越少，加速效果就越显著；反之，如果草稿模型预测偏差很大，目标模型频繁回滚修正，不仅会抵消推测解码的节省，甚至可能拖慢整体推理。然而草稿模型与目标模型之间往往存在能力差距，尤其是当用户输入的文本分布与训练数据存在差异（也就是 **domain shift**）时，草稿模型的预测准确率会大幅下降。

推测解码虽然在机制上提供了一种新的加速路径，其效果仍受 **草稿模型预测不准** 的制约。为了突破接受率上的瓶颈， **OSD（Online Speculative Decoding，在线推测解码）** 被提出。它的核心创新在于：把 **蒸馏** 从训练阶段延伸到 **推理阶段**，让草稿模型在真实用户请求中不断学习，从而逐步缩小与目标模型之间的差距。

### OSD 框架概述

![](./images/OSD.png)

在 OSD 框架中，用户的输入首先进入草稿模型（Draft Model, 学生），草稿模型会生成一批候选 token 及其概率分布。随后，目标模型（Target Model, 教师）并行验证这些 token：预测一致的直接接受，不一致的则由目标模型提供正确结果。对于被拒绝的 token，系统会将草稿和目标模型的概率分布成对存入缓冲区（Buffer）。当缓冲区的大小超过阈值，或时间超过设定阈值时，就会触发一次在线蒸馏更新，利用 KL 散度、反向 KL 或 JSD 等距离度量，让草稿模型逐步对齐目标模型。更新后的草稿模型在后续推理中表现更接近目标模型，token 接受率不断提升，系统则从最新的正确 token 继续生成后续序列，进入下一轮推理与验证。整个过程形成了一个闭环，实现了推理加速与草稿模型的实时进化。

### OSD 实现在线蒸馏

OSD 本质上仍然是一种知识蒸馏（Knowledge Distillation）方法，它继承了 “teacher → student” 的学习范式，由目标模型提供正确分布，草稿模型不断学习模仿。不同之处在于，OSD 并不是在离线训练阶段一次性完成，而是将蒸馏嵌入到推理过程中。用户的每一次真实交互都会经历“draft 生成 → target 验证”，并产生新的蒸馏信号。通过持续更新草稿模型，OSD 能逐渐消除训练分布与查询分布（推理分布）之间的偏移，使草稿模型在真实场景下越来越接近目标模型的行为，从而提升 token 接受率 α，实现推理加速。换句话说，OSD 是一种“在线化的蒸馏架构”，让知识蒸馏从一次性训练手段，升级为推理系统的实时优化机制。

OSD 带来了两点重要优势：

1. **持续自适应**：草稿模型不再局限于离线训练时的分布，而是能在实际交互中逐步适应用户的真实输入。即使出现 domain shift，OSD 也能通过在线蒸馏快速恢复接受率。

2. **接受率提升 → 加速增强**：随着草稿模型越来越接近目标模型，它的预测正确率不断提高，token 接受率 α 稳步上升。这样目标模型需要重新生成的 token 就越来越少，整体推理延迟显著降低。实验证明，OSD 可以将接受率提高 0.1–0.65，端到端延迟减少 1.4–2.1 倍。

从本质上看，OSD 的提出解决了推测解码和蒸馏的共同短板：草稿模型的静态局限性。它既继承了蒸馏“教师指导学生”的思想，又利用推测解码的交互过程，把蒸馏嵌入了推理环节。可以说，OSD 代表了一种新的蒸馏形态：**让蒸馏从一次性的训练技巧，变为推理系统中的长期机制**。

### 参考文献

[llm_distillation.pdf - Google 云端硬盘](https://drive.google.com/file/d/1xMohjQcTmQuUd_OiZ3hB1r47WB1WM3Am/view)

[[2402.13116] A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116)

W. Yuan, R. Y. Pang, K. Cho, S. Sukhbaatar, J. Xu,and J. Weston, “Self-rewarding language models,”2024.

[Generalized Knowledge Distillation Trainer](https://huggingface.co/docs/trl/main/en/gkd_trainer)

[Online Speculative Decoding](https://arxiv.org/pdf/2310.07177)

# 

# MiniLLM：大语言模型的知识蒸馏

**提出者与背景：** MiniLLM 是由清华大学交互式人工智能课题组和微软研究院的研究团队提出的蒸馏方法，于 2023 年在 arXiv 发布，并作为 ICLR 2024 会议论文收录。该方法针对 **白盒生成式大语言模型（LLMs）** 的蒸馏问题提出创新解决方案。

**现有问题：** 在生成式任务中使用传统的 **正向 Kullback-Leibler 散度（KLD）** 目标会导致学生模型将过高概率赋给教师分布的低概率区域，从而在自由生成时出现不合逻辑的样本。因此，MiniLLM 提出以 **逆向 KLD（Reverse KL)）**为核心的蒸馏目标，从而更好地捕捉教师模型分布的主模式，避免低概率样本的过度学习。

## 技术细节与原理

### 梯度推导

![](data:image/png;base64,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)

方程(2)是使用策略梯度定理推导出的目标函数 L(θ)的梯度。

- Rt 是一个累积值，表示从时间步 t 到 T 的所有步骤中 log(p/qθ)的总和。

- rt'衡量每一步生成的质量，它的计算方式是教师模型概率与学生模型概率的对数比值。

- 这个方法有一个直观的解释：我们希望生成的文本在教师分布下有高概率(增加 p)，同时保持多样性(降低 qθ)。

- 计算期望使用蒙特卡洛采样方法。

这种方法存在三个问题：

1. 策略梯度有高方差问题

2. 存在奖励欺骗问题

3. Rt 偏向短句子，导致学生模型可能输出空响应

为了解决这些问题，作者提出了三种策略，即单步分解，教师混合采样和长度归一化。

### 单步分解

单步分解，用来解决之前提到的高方差问题。

前面标记中的错误会在整个句子中累积，导致高方差。因此将梯度∇L(θ)重写为两部分：

- (∇L)Single：关注单步生成质量 rt 的梯度部分

- (∇L)Long：关注长期累积奖励 Rt+1 的梯度部分

Eyt∼qθ(t)[rt]通过对词汇表直接求和计算而无需蒙特卡洛采样，这种计算方式对参数θ可导，提供了更精确和高效的单步生成质量估计，从而减少训练过程中的方差并加速模型收敛。该方法的核心思想在于将即时奖励(单步生成质量)与长期累积奖励分开处理，通过更精确地估计单步生成质量来减少整体训练的方差，类似于强化学习中通过分解即时奖励和未来奖励来提高训练稳定性的技术。

### 教师混合采样

教师混合采样是一种防止"奖励黑客"的方法，解决了学生模型在训练时可能产生低质量但获高分文本的问题。该方法通过混合教师和学生分布 pe(yt|y<t, x) = α · p(yt|y<t, x) + (1 − α) · qθ(yt|y<t, x)实现，其中α控制教师影响程度，p 表示教师分布，qθ表示学生分布。为获得无偏估计器，文本使用重要性采样重写了梯度计算，并采用近似方法降低多时间步长计算重要性权重(wt)带来的高方差。这种技术本质上通过教师模型指导学生模型的文本生成来稳定训练，防止学生找到技术上最大化奖励但产生低质量文本的退化解决方案。

### 长度归一化：

研究者发现模型倾向于生成短回复的现象，因为在训练过程中，长序列的 Rt+1 值（某种奖励或评分指标）通常较小。因此提出了长度归一化技术，将原始奖励 Rt+1 除以剩余序列长度(T-t-1)，实际上是对未来时间步的对数概率比进行平均。通过归一化，消除了序列长度对奖励值的影响。

经过以上三个改变，现在优化的目标函数变为

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA7gAAACMCAYAAABWOgV9AAAgAElEQVR4Xuy9XagbWZsu9pgw2foy+c5Wh4+xGkK2fGX5zCSWmQGrZy4scyBWh4EtQxKrL4aWSQ4tExKrb2L5JtHFgNU3sUwuWk2SY/nK8pXlXMTqDInlq9ZmBlw+EFrNMFg+k6HLJyFdPjPwlXMIzrPqRyqVqqTS75b2fuvjo7u3qlat9ayfWs963/d5z3zkBbkEAUFAEBAEBAFBQBAQBAQBQUAQEAQEgR1H4IwQ3B3vQam+ICAICAKCgCAgCAgCgoAgIAgIAoKAhYAQXBkIgoAgIAgIAoKAICAICAKCgCAgCAgCJwIBIbgnohulEYKAICAICAKCgCAgCAgCgoAgIAgIAkJwZQwIAoKAICAICAKCgCAgCAgCgoAgIAicCASE4J6IbpRGCAKCgCAgCAgCgoAgIAgIAoKAICAICMGVMSAICAKCgCAgCAgCgoAgIAgIAoKAIHAiEBCCeyK6URohCAgCgoAgIAgIAoKAICAICAKCgCAgBFfGgCAgCAgCgoAgIAgIAoKAICAICAKCwIlAQAjuiehGaYQgIAgIAoKAICAICAKCgCAgCAgCgoAQXBkDgoAgIAgIAoKAICAICAKCgCAgCAgCJwIBIbgnohulEYKAICAICAKCgCAgCAgCgoAgIAgIAkJwZQwIAoKAICAICAKCgCAgCAgCgoAgIAicCASE4J6IbpRGCAKCgCAgCAgCgoAgIAgIAoKAICAICMGVMSAICAKCgCAgCAgCgoAgIAgIAoKAIHAiEBCCeyK6URohCAgCgoAgIAgIAoKAICAICAKCgCAgBFfGgCCwIQT65SQuPHgb/LaD2/hxUEdqQ3XZ2df0y0heeIBgFA9w+8cB6gLiznavVFwQEAQEAUFAEBAETjYCU/fDuIbHHzsoTEDQRzl5AWHbaFx7jI+d0VNCcE/2GJLWbREC7oQ+e/EQmaSvYokCGo0CEgvX10Cv1UbfjFJAHOl8Hul4lHu37B6X4J69iMNJEFFoNFBYHMQta6xURxAQBAQBQUAQEAQEgZOFgN4qodTSfY0yoHVe4u2HMIKro1UqYfIxDZ2Xb/FBCO7JGiTSmt1BwCW41x5/hOeQaTUNYIGxzztIf1VFtVRAmiRPq6Tx+aN3OH/nFbpl/kHX0KpXUHlkoExLZ20XLZ0uwfUtZKsBUUoRBAQBQUAQEAQEAUFAENg8Aq6FNozghtQoZF8oFtzN96C88ZQisE6C287HUM30SWqTDroGGplPcOvoIu79rKEytGq2kDvTRDHQ/WMHOkYI7g50klRREBAEBAFBQBAQBASBeRAQgjsPWnKvILA1CKyP4LaRT5C06vznsLWKyH6B78/exiu9jvTw701kkxoauxrvKwR3a8azVEQQEAQEAUFAEBAEBIHVICAEdzU4SimCwIYRWB/BNaAbMSTisVGLukXErz6CeeM5zFbO01IThgHEvfduGIelXicEdyn45GFBQBAQBAQBQUAQEAS2DwEhuNvXJ1IjQSACAusjuJMvd9915eFHkOuenEsI7snpS2mJICAICAKCgCAgCAgCFgJCcGUgCAI7icDmCG5Y/O1OwjZeaSG4J6ATpQmCwMlEwOjWqAyaQK1RRPJkNlFaJQgIAoLAmhAQgrsmYKVYQWC9CGyO4DIW98x1PJuIv11v+zZSuhDcjcAsLxEEThUCJsM8zDjDPJZpNfM2xj7Hkw/7uPGc6SxynpARsHxmxEgs94JlKrd1zxq6gRjx8KK0dZWUCu0MAqahw4wnsNQUntFa6x0xvkMGrY3UStZNL+hCcHdmwklFBYGxqVtO4gIzVK8lTZD3RaHxtyegP4TgnoBOlCYIAluEgFZDOttAtjtAfaTGN38FrVRtT/DhgMJ+FPEbK4o6CGeKcTw3Kf43f8kn8gmN38Nst4SuVhnH6kS2Vhq1TgS0WhrZRhZd/7xb6Utt8tUt/cxsFcO0FCt9w84VZnA9S5YRb/bRyq/iaEEI7s6NAamwIKAQ2JQFt89UQRe+eYvL3/6CXmkVi84W9Z8Q3C3qDKmKILDjCKgNWqIINHXmJl9urewUYiC/5brb57qbHAdGEdwvgMe7mp5tLd1s0MqdQBFN6DwcWA79tVTwlBWqo10uoTlYstnJIhr1PDZFAQ3OLXsK04NirYPIJl/t/I8Y1FNLgnSCHtfKSF6i1+CrJQ8ILUiE4J6gkSFNOU0IrJXgml1UCnUuDzq0Z0d4S2DPXj5Ehl+ZbKWNcuaEIC0E94R0pDRDEDhuBDRuWC+hlX0BvZldsjKOe3L8Bp73SZr9G20huCH4dlFMXKUQIklDTUjDkoNw6ccHtISeu/t6qXI2erBukasWsi90LD2FZ7ZaCG4YREYjg0/KCTxVqSqXOmQQgjtzGMoNgsA2IrBWgruNDV5HnYTgrgNVKVMQOHUI2Jv5OL79pYulHV0c9+TUvTd0X0xOYikEN3R8GXQt/eSWgXtvNARBd+oG5rE22D5wePQO2Lv2eD7LupoDdMPv6g1s5jx9gFr6HO7Gv8UvdHVfildFwlwIbjhMPZQSn/GQwZ+WMhKwnpuE4M6LmNwvCGwFAkJwV9ANQnBXAKIUIQicdgRoaYhdR7fwAsYKTD9dbuyvtjJ4HOYmKQR3yoAjqYpfpSX9Kcx2/rQPzGNvv2WNu3XEepzFVz/o4H9GuGyy2Sn9whjV9VNNq0IcK7HrXRReGBuw3qoXCsGdNhDscaMOqvpLHFQJwY0w2eQWQWD7EBCCu4I+EYK7AhClCEHgdCNgWw01fLmSzbFN0LrFV4zNC1GpEoI7dcBZBwSP0quxpp/uob2C1o+suLgS0TqqyCZFwzSKPG3G0ZypELOf4Jb2JV4YTWRX0OrZRQjBnYqRQcv9J7cw+OoH6NFORQKKE4I7exzKHYLAFiIgBHcFnSIEdwUgShGCwMlCICh9h/qbQcfFxEROD2dz3GO87FRVYzu1T3xmKhsDvVYHyBWQCTNeTSO45gDddhd6IotcNhnsasm26JTt2dYsQ4auodfR2IY0cpn0sJ6mobFtfcSyOWSTUyx7jot39vFHin2drLG5i62xD4BesupneQg0K76V1ttUCt1qBKE2NY7NmCddlgGD6bnmTrtDMpUlmerdmO4Sa68Bq5o30wmuoXXQ1oB0Pot0YINMGLq5vamxmPKn3+ugNwCSmRwyKTeFl4FBt4OumUI+m57SVzqt+J/irnEbPy580CEEdxfXC6mzILAxFeUTDbUQ3BPdvdI4QWA+BKjEm0+hYnDj1fseZukFeuUBitkK+pkCcnoLjUEGjW4bxaRbsiMIlaZ1qhccu2e0C0iXdeTSA3zXieO+RkEqj2nK+r3YQ8ZKjxEhKWYIwVXlpAo9JKlKNXj2Gu/2D/F40B5Xg1UKw5Rnfo9DPP1I1+r5AFrv3RSVKRDnNhP95NIJ6NozHFHh8PxXz1GPlZFvAplsCiY3//10EwO2N5Dm6kzV9Old9GcQlvU2RkofIeASDf7lcvg8UfdbKsbVLPr9CoZTzAel2W+ikCuhF8vzn1RrbsVRYzrDdvY690ULpN1xDkTSoZkiNNQyeTSTGSTaT9DPP0ef9RyNvT7JWAa1WAW9XiWi1TmM4PJd6SxqZhpps4eXbz/g/J1X6DPGf3QpheFz+Jr6XQd3tk1QzUSvxsO16oDENsM+7KP78ie837+C++0iuvkStFQO6Rj/3gNKPYVd8FyxleRTuPcz4+kXktEWgiurkCCwkwiIBXcF3SYEdwUgShGCwAlBoEvCVMliQKLayZ3BF9+zXfvX8LDfoViOaqMtfvIdvsIPrviN2ohf+AZ6GJkakGylqUo/IBHu2blt42Nud7QAZ+geebSHG89NprqJgGUQwbUUYClw9aPaDNrufSry0b8Bbrnt2jaC66RYMuvc+LppkRw3RdUOqhRZMcm5bh7J689I0KfFdDqHDok7tP7UIhKOCLjLLQsjYBFXTqgPU624ipCk6ZpvqpDYwMstJ3H7B/TrGajjIBX3nrj50ir79it9LP+0IsPFYg09I45MpYXW6GRqWL6dClEPnX8q5Vcp0WW9ErZV8fUVfPtbzjX3LIrkOn71Ed5fvIefmYc5GhcLIrhOqqtYCwYB6Fmu9u9ZT99hlLtvCZjfC3fQih5U+agvtbN4odHV2zkBsEMGVDu4Ht1WoRcxWukv4O5P/APd1n9LUa+gYz2dzPdTqnAfPv0YOh6mV1sI7oq6VYoRBDaLgBDcFeAtBHcFIEoRgsDJQEBZDKpZnXlnBygnLuHBu0mXSpsg7o/ibR3rT+J2cD7LXom5WZM9qE20+vfPvnuHi/e8ViaHjH24gocfGa8YBcoJgmu7SVdTdrzayCXUT5rdd/ElMyxpUaoxvMdg6rhSH/lGZbipnet53tym5bqA1rgwlGcjf9Y5FBgR9ItTLDvOxlaf5TY+by3l/sUR8FhxSQTfkAgmfYVZ47aRC/zNutU6xHmAtwc+t1WjzgOdr3G0N97fdk5bE/U+ySh40HGujmyA0JVtKUzg9o+0jPqDfpX7cqKDsklvB8eV+aXvPS4RO0vypnvj5rUGCg1alxuFAGt0AMHlwVrisz6qlhK7c5hGBWouGGPE2X0fT99WFPdvd4TGejbiNTQK/p6J2OvqMOkcDxR8wlCjOcuDAdU2nf14gf3IYvemeVlY69z3vvUyYl2s24TgzoOW3CsIbA0C8xNcpUZ3Ad+oVWVj1x5P38JPYzdWjbAX7QjBbefPgEaLjV7nmaJEbcrlEgROBwI6WqUqUOWmlJth5eL6et8vOuPEhdE18Job3+lswg4CCa6nzIS7YT0/rgy6iPUngOB2KhUYJdY9OaojznoszdZ+b7SxnMu1kXG9nXoJpWoXyTpjfEnUvZf9LYov4UpooFMpYVBsoeR13R4q8HqIuiIN1Q6JSwP1fJitbMGN7c4MdBV/acC06hvzxXV7fwv6/RgbaSkVP6OlNchbQc2PLK2xYV4MThofzr0rdCUeU1d2Dpk+XHmIj5xP1uV4ViSGbse2eFuH4m1jJJS32gTsIJjg9urI9zJol0cHR3uHXoXucA8Mq9xu2CFLAMHtc2y3U2jRi0QFjytvjw+sn7+9i3phGLSqlgpltI0COvRA4Vs8F/Ntn/kC3WXc+lX9m0k0al73bRt3y4A7JOoG8SyiqWdRaZaRCYvKmLq2RhnHC64DIfvCMx95RXmt3CMICALLITA/wVVrvnLF8TDcs1/iucbN3KJVMVQcxQCm3kO704PWew2GjIxfUZUTF63DMs/tCMFV7koJa2PgXHuXcb/HeLVovlABCOnQlIgLRVs6DITRuhp+eu/rOP/meBmc5VlBYJcQoMvjGbo88nQOH8d8JV0LqGeDHnUT5m5YfZbTUOvPNLymiUxZVpS7UN5/rtXTLWpuy4/JmGBuRMuMDY7n62g1ihS88VdsFWIwwY21LWtqXaJ78kdiH3kMLbixjVz+cd9IEkuin898jZeEx3b7dL/i/I0Wy3LuczzSGffI70SJnRYhsnsDjVICUsw1qwanz4prpYVp5MOtt5Z18zu8w2VaAUmGPePQdYEdeUY4pHPgPeBx5m72MT76lMemEtwhKu7Bkd9qGuaBEf4+u8jpIlPKo+H6M4ve+tTA5/fC0EnSS4UKOqS01UaDYyM5OR7WJczmxMPzXILjNNjLJXTgRV1bQwtYcB0QgruBtUBeIQhMQWARgstoFZ5WJnhaOSIz4x/H5SFXSpdtnsaXvzvix0hdPovF8q9YXQk7QnBVg63Ylgejw4m9a4+hK8GYVaFh0DLD09Ri5ZlzSLHl1vdVtVuVY3RRK7WQqDU84kGrfIGUtUsIuMTqsl90xrW2ejedETdh7iZ8vMwF4m8VkFMI7ojETsYjRrb8cA1vlgootw2kinU06xSv8jMko4MKrah96jH3nnGtP3sRh5kkkK3SOrPwkalnmARZfqKOogU3tlGL35L73D0AHT15FkOxpbxH9ojfiwxT3qwiL/NKmzu0THq/LyqVEI8vWuEKy8NxfeCPq3aJp2e8O2SYiXdHKWZcojVxaDXDgus23o0H93t1+D0wlBWz0oHJw/8OxZVwcIWCaXEkJzwOphFcD4n1WqVVXSJ7YZgYdEhs6ZXSRQ61FoltZvJEvE+35EpHHYp0YFeXAlDxJIoNWq4XPkAfjZhRHuQ5NAbcxyOureHjc8F1QAjuSqe8FCYIzI3AYgRXbeYZS5K4Dutw0Lr26G4XQZJ/3hry9L9ZyOLms3cYd+mZt6A13r9DBFed+PpdzFd9OGEjzbFQziH/4DU+nGes1BQ1yzX2zEaLdgnNPt2zdH5Ut8PasVEI5GVDBKhwbMXfTsZ4ujG0Y+sZ4wITjAuMTbVOuHP3AHcY51cbuuG6G9k5LZRTCO6QxPriBOnziELsc1gG0ZD4W3PQRYMWW+7PyVMb/PcckrMmw4osP4NmHrlyF7FSF5qSVXVcTNWR3kRsI9OGZLJUXx343SzdTnT6MLZMipHtnxJqD5DVUoi9/B5vHREu6iFZl/otxz4fTASVHne7Rq7GNOlZKWASynrbLIaqkLvtucAD3omYTZd4esa7uzc6e/EQ6szF/qz18OzIH/9u/6SVEzw8jgW7KLtw0ZPjjIoT8hFkl3j7rZP2340pZU4huB4SOx6vz2Y4wkvh8bcGtFYFpXIT/XgedRLb4qTbhW8QrMgLgwfF5ZxyPU6jodnq7aPwqkmNgQ4Fv2rpHrrEP/ByPGmG4SBzD10huHNDpqxgusHcW3Mn2+KrmBtKZ56ubc0/twAY8sgxIbAwwbVXdFuswa277+O4yiZplRQufZPwudms8g1LlLVTBJfttJRGv8DIAM+4oVcqZcASGIQ8arSoVvpFD4UAUY7Vv23xElUsWmxmbtFp5bsb/2Asg3KiLl7b3X/SwiOeWJ3nwLZB4lp6JuJv3XHic5F0N9j0qPC7Po6aFrLRWiT+VhUaheD6hXgcIq68avzxt2a/hXKxjGY/jny1iTrjDaMab+zvkFrfx91GR22nhZd5ffvxDFO6kIwF9jcPXc/w0FX95ljoLNJzy9JPnlBR7VLS+mq/OoUQ2fGE369SSGvbxinr45LYTixnhx6xz1+RMKrPwQTBHTSRz9WAIl192zWSECfdFcWMmtUqXVe/RyJ7iP73VKg+W8Sj6j7+GT1aXibLeJhoo9L+Ce8+nMWNh02kmyXUNR3v3sdw+HiAtsuqo2LEcZ+g6jCPvnGN5cVLZcS7OtXEwwsII5L2XKDc+dDS6XpFjB8a2YeYyfH4d+d1rpVxKpEKtCbOiL/t+WP4ve2LRnD9dWpmz0BFT0woKytPCsbJK+sxspTUqtdQmHC7CMPXni+9L1+EWvxNWnhbPXp0qPzcIYuDNwTOVj72eGH4Le/WullFakpeZHttMZYQ0hKCyx53g/K9CaPDdn08lUgVYVBoQUmHh19OmTFfUmi1QeWCEbfy3a3MuTDqsiL3nSAEliK46gPoj8f1fBxXCxMFBejrUsn0t0+0aNcILjtmMh7XTp8x7x4jSh+rMZLu18eVTaM8uJF7OK4KKRSpoDlYxlXbsUApq1afrm3JsbrbH8huaYHcihvBYPMvUcqkyXKcZKiFE/kJc+Nvx0Ir3NCOOG489X+7nU0UplkL3Y2wlxzTypikpdjiJYvEpiEwLnVAC885ptag2wyeKuVXNUS4PhQzjMm0TjQnlVe1WgbZuxoSNxohcbZh42y25WdkcfIIc00U5xLcA+KroZXtIp8i4bViXLyurCrHJutaTZCgTVnzIlnVNz93Vv3GEYlNDEOP3NAV3WvBdby29NovzOus9p3OeNQ4Rph7ONuyDxPOkuD0KxSF7CeQ58CpKk8Gk/cwl3I+7pAVk5Y4WvqLCQN1prf6Wl8kFZM7HxxEouh0qHRbjC3vewWelH5IhmJw9EoYWTqD5qNT92SwerPr9ktza7jF2zmMItsaksBR6iO/B8as+FvV7mguyiOxR5Npkeg+/LVKh8TLf3jDPi6kr+MJ++desxkcZxs2AGd6YXi8P/wHZ54y3T3l/uV71GapIO6uReoerzeYynfNuvYK3tjxycrZZH5O75aA7/eDt3OWsRYXZY0+31TpW/bKVilZH8WiwQWyxtOs6vdvh+It+4ePw5OHw/4gtfPTO0Vn8vFMgWqDjGGJ0z2gn++gR/nzIZ21rGdcMNZkeVkWP3l+NxBYluBuIh53iKTBU+KmjmyZVsFtgndugqtRuVPFtCx5LRmntvZ43GHzeJBXo9sg3Razs1wVl4Rk3sftfHv5ocVi3ufd++2TfRKX5yQuOf+h43QhkEXfuevPzYc94+pyGZR5+q8uM16ERiuTPxvHtmDiuqufv3IFxiCOcpV5cRtVS/Gz0WmhGGAVsV2Xk1OsmGwdN+OZ7F1o6dto0ALZptXsmaXIt2hsWjDBVet6u5BG4Qn3NXv72Oe8Nek1Fo+9pcVNoezLqekCPxSU0hCjpa/VoLrpTDOuz/LDdb7TTyGX8cwjj7fQhKuxp9MHTA+ToYuyGYuxvkCaSskdEt1c/i6O3u1h324IYukSWowtdHNsBo0bm1TrDMGcbhXcijGnVHq5xk6/EsiUqGKdTbMfR3eOW2m9ByavoI428o6LcpC7rGu1tGLC4wXLCjpuLfQTxUniaLvDz0ke3OoPrbgh6sUBgNheRR2kb1Ngy2ygRmNlwjjC6/feeHMnHCA18qiwiaiGQqi10FE4T347xSvAnlfXnwA37leRYaxtrcnYczWF/flvnX3FUMGZObCZwZnGLW+jZnxb1HqRU2OfK8T+PmJq7NNTyXj7zuIrYSroQ0EpM41ind4YPASe9en2e2EMOgQ2x4PMYXW92i1T0pkpF+VsDo1+DLGYybU+j1aXXgNUhVbrEScx68K/mwnkGl20A3ISjxBySHWGfcI8uYuZA7fKgus5JVh05aEq7AudCYZnPK/iPTI3e8g87KAaq6PwxSNLdZAjNUTq3ulgWgt+4Sl/GNgqdiRzt4/cU2W14rCyTka41PgSuFuLC/3O1enZiTwFX7T/5LnICCxPcNWr/HNuTfG4kVu14RvnJrhqSrvKnovWdTK35vwleeKYnIfXE487f8028YS9YaEFIdQtMmotnPGfCjnZn6F0GfUtJ+8+2wJTTjyFzkOGKJsPg9/CJNNevJ9iATh+nNz4WydWNqEzpEjVaoZ3lyNqk/KnL5lo0CiFSyzWReETutF6La1RAZjiojxyqR95psUTfFfM0V2YaS3zujpW0SDRzYUF4rrtfvgR3aIaE0l0q2qvFNAQZfknmZseE8oQLgIeS/hc4OkWr/ohplzjZ+3WXeukvlvxtyY9B35l+Z5SIZ/rmmVo5aX6s9+pUizoOxy9dyzczqZxwg15GMJygCtXKDKU7lp423sFjMeDOt8+y2qZKW+e4DoHMWXUGTMabQ1xAGGIoJqUnJOxFt1cb03kv7UOnDpFxvdSUVx5McSvQytNN0zZh1Sp2aFUzlhU71cHB0r40Z9Sz6RB61e3+gwf0hk+pKzdZaR6GjU0vPNiGsFV84DkXcUyuu+jN6jZSNE7w9JHn3l4MxZ2UG/xUI2HI4Hri8kc2r/CrT7d23W6t6uDh3IKvYA8xbbVmQdfJOzTFM3VmDX4tvEwzjk8ZVU9HYt5xk3HFnVtHLtvqwgu+3IYdxH9VMduj73h65R8+bECQHHdCtIehUQ7yNyKTvGJQDgFWESVA3XaiaBj5o/ThUHnYmVdTjxPfyK3lH1i1Mw+hxn4NVioN+UhHwI9TtQcLYdzXcmKNbm31cLgtmU1BJelbTAed65+2MTNCxBcLlL2R1WtFnO6Flrug53SEieSHlA2GI+7ia6I/g6blHaLHoXM6A+P32l9RLlJChVZEwtuKLQWueFhsu/wNrwrnLjIbSa4ofG3swaY43KJ+zwAZwjS2O3K8pNBsUdBlW4HpaSzY3Hc9/ypfGa9yfo9hODaegfcAFNw54XZGh70D92Wub+JHrNPsRrGWhbKT6AnbgSL1SjX1yTJQ/YecjQsaPkWepVg9zkVN1vLKTHDSC1c/CYnTZKyno3lSl28xIAnVd7eIrrZJtsU5Xhn9suHe1C/NdB9dJgmZ7RHDRSS8nzP3e+T6yY+puDtuHEnj8uCOxuS2Xe4isz++Hd+G/PJEowi3WTbdeilNrqVzHRLpvNd5+mC48bteb2ypuZrMHJt9Ele7WuUHujbX7pjaYvQJ7G+UEX8dhWJThOxWi8gJDHk22LV/Qs8o7fFxfsMjRkKMI2MEXuMN9bVt2s2QjBJRuu0/t9lmPK1e8HCcX2q3l2oxnGbrv+dZgy1Xkj4icIoTfHQDXjgWGmSekX8wHy9U0KzZyCwZQSXtJ2nHUwKrLjmzJNGT9vUQC/G0Z0FhqvM5/vIDtjB5+jScvl2mwnNPe7E1iuc3F3xae4Lbr39Oau4AThzEy8npM1dMm8EBr1HGLdyS1QEmPKgnLmEHg8/uPewLA/1zCsG5KcRUydNOuXc6YJWedBD6ssc+o8M1D6yP6OWf0z3rYzgqhHujZVQ7VlbPO4xgRX22kUILsuyRE7sRWr2ie/w3eoDxY/SDDGNuRDyJIO3nlujWNhc9VrjzfZaHZ/uDhrx/VbqFm4UXGGWyceE4IZD6ZA6I6rS9g4Q3ND8txEGlLW3oLqvn/APFVHpHfbGseK4h1OJkShQhDeMbgkkuI6bpVqWPDmsTX70Up8pQUEl6MOQA65d812M/WsxfRhVWTWKGTW67YlUWoZtbpqy4VbK/U0UlcfafC+f+27Lw4Zttqx3cz8d9YFVrwtuqpsAxehhlZz5Y32e7VhRRX6UUrLuUxlUqYEStAYn3ANYEpMsD2VfeuI2rXW0msZzHoTkHAGljbooR4V6yn2T+W+9N9sWw+njcrxwK4a0nrEx8fzkKpN7FZztMCF9SgYK9X7brTjY6SB4DNnWX0tFihTIPaRRYR5J3FTqktyb/UBdjMxMTzrCYbgAACAASURBVAYfcFSRrjNsUqUESjMl3gTPsQRwp4vnqjGjxpvCaa2XFW9dRdKX+mr+d24dwfVacWeb4e0GKwKaomvMrFQn/CBnP4EaO6OB40CmiA5dAALdX5yTM+uky5td2oO2SxAmJMzdj1vQqbVzYjTw5uqavwfliQgIqAWqnHrFD0HMEo3Rqsqlyr8ANOmKdZMKtVNiDCK8a1O3rJLgbjQed1MARXnPggTXa8WNaoFRH4dUt7ra3LVs40Q87hynu1Eg2q57osRKRa+xwZNqFRtV8MYMjj2+6o1s9Lrtwp22x9Vgpruc3ZZtJrjKJXBAK0sGXx99gNLiULoZiamkbbKHrLnYyo6HSVlkllbdth3jbXJzmst8jR4Fb5QQy9wbVQtKpRzrj8G1xfyuP4vxoJ6WxTwFmXhwW3vyE97vX8btNi050wJXZw44uryS3PaTBV8s4awHlctyktoFg9D906wSIv9uxXXSHXvt+iarXhccQ8i0eGyPCvbFe300GC9aKT7ASx6y3m7WUKWXwMiiZ4fUVVLa0CXc7NWQzVVh0BJZYWqWes1AqUePgjhzHhdyJE/vcPbaQ3RaReZBJTnTqshe+gav9+gyzfuY/MUat0cf6AXwQkM1STfqrBIuO8CXz5lein08L+eK3K++G9WBit5vosQY1ZfkfOfvvGKqmaTt0rvUZccxt7IeL0yWp/ZaaR6a9Pn/pEqjV8ri8+8MWxBt4WSxIWPIMcLpB/SaoAdFkul26vUHoFQQDg4fotsuLqdjQoNPhyQ3MzZeIoCmPANyBrUIZoeARihtyi1O+Ev88Qr2SltIcAHPSWQUmXfGAMU4AGcKV7guHnPGvNi++YmQ2FzVTyq+5Srzy02qE/KLY+fOCnTLmq1AuNxAkaddBKYT3FHaJ3vDFsPDU2bBtXE6/nhcyxIQKc5K1XeJdF3uwFiU4PJ5NyemipkKT5HhvoiWixg/EBpT+qzcrLAD8bhWHJE3jtGAYQnfzLlGOWt44t50VWM7BsinYD/nq+zbp29kDa2DtkZBnHyWm8Kgxsw6xV+oUqt7iKf2fW5eewMgmaHIT8q1NhgkMx10zRTyPmGbsZc7Lr1GJFf9CAR3GN8WLd5SrRdmbBTrZXLz1mWHGCl1aDGHpZLPtdVzY41LIFv0Cq1EgT1E2ZsqtOVK3cI5kUwjX65Q1XiO+vlfHRqDS2srBZjqjR634cQwmaUaLsWGspsjHxMoKWx7TD80rrATBcz57tlohooVE9yhJ064YNPQa2giv3EYTFxjDa6xPs4XHB85H9THezdd59ttaOOTlYOda9W8pC2oIUqNeCJTikGPvwpqXCN0flfSuRLKzBEdOQtPIGBTxhCtrc1aHe2BijWmwBjncC5PBeVl+fsSHTfotqCnKaC75jqoQ8IsRaW6DBOMohU8vUlbSXBVukclIvI91cJmCbKoBqR5SkVp8xm+L644zISVdSpCDgkdTMlnpQg2SeyHgIVnGFcxzNE1/jK7Tqkp5HmJESmPDhGYRnAtUlvP4406OWOQfTY7QJWn7Nktx2+1FlynsRPxuLRm60pif81gWBL3ZejMkzj4rov4fY2WyRETNKhMni5SFM5Kr6WIhEvGqUD6wuSJ64L1W4LgenPSqvQKw7j7gKpYJ8CMWzFnLVILNmOT+XHnqaLJU/YCNwO9WJ7/1NFuxRmHWEQ7e52n4vOn3rFjyQZT8uJpqGXyaCaZx7P9hAr2z+lemfNYNvrUasigFmOcPa1ns88awjYhfE86ixqVKtNmDy+phqusCH3Wb3Sp/MTn8DUztoQpXs6D5WrvVelWsshVByS2GVoE+ui+VJa+K7hPC0E3X4JGkpiO8e890NKjcAuqwUik62duSKZTtikEV+XoZO5GNU7KFX77SYqazME5yDQsi9L4RtKgtYYEttFHPJOliAwtEjxkyNI6OYgVmNqkjS++i2pVXi2qdml2/eqpNto8HF/LRUtSuppCm/H8ybW8YNcKpVoySYBR6TId3bynZou0dbUEd1b8rcE1M828sW85P7/VGO8pnb5Ip0V/RqkB56j0TiK9rilMgR66HecYs67RszR61U70nSS2Gaqptyk6tpqVc0sJrntyrnLETeR78vSwRYQp499X5GRqz7sJh+eV5Hfys4UF/quttquqev4GvuXH2bu8dms38YiaD2Hy+G7wv50UefbQ7TdsH/rFrgQK9QYKp3BxnEZw7WD2kq0gtxiwx/LUWgguW7L5eFxlgUxTBELllu5ZAkJP4l95xAWCkqmPUiL4FQzn6oxlCC5f5PbBVCuu5aJId6++Sl8wV+3mu3nL4nHdQ8rE7R+Yv88W+HBjw9TBpa0wOWqiIsPFYg09Wh0ylRZaASkE2vkzdMEMU7pXa3GcGxKlHEpBm/SnuPuaG8LfckPoLsqOMuP7Kev5OOhBG1nb/a9IBU+Vk7in4ngfqfwrvvQrw9jL7SO4dpofutJqdDdzDq/ceDbVfluRm2IjqQuwRDuph/Fbbj4mqYMbOxiSemYMzBCCqwT96Ao5oFCMPpbP2JnjzC/rjZF2Y9+uPNQZZmJX3rZuGThUMVtZHoK1DWTmtr7ON93k7tOMwCoJrif+9svnzCY1WhR1rYVmvc64Zx7gVBtoqsOf0wy7tF0QmAuBrSW4XituGClVrsxZbppowQmSpfcC4W5sMGfOLldV0a/SNizbJc6MA/jyoUr35NmxMc/urSdMPzSFVDvB/aMk1dN7T8WMtfuLEtwYUvn1uxjMNf42dHMYwR2ejJ51JNI3VJ9VvGZdBHfj8bjK7ZRkpqcOqdwwgjECMlIrHHMddwhEyicjr/EQqBGvoRHlJGdJguu1nIZ5hihX5qzR3Iha+tbE47qeAP7QDKZZy3zCOC6ft4tFhosm6n2SUVAc5hw9KCYU692NYMgaroRUEh2UVVoIV1TF9x73QHFaPs6ZBNcao31ULeVMTziNjzS772LyvykW5/lXgrnGd1DxlsosDxPejB+4uEIqQ+E0nS71zC2izpineT3Zz4UfOoyqEERw3Q1IiJu/4x3FJMXO/HEOnBl3OLYWuIc7kQ8u5sddnhAERgiskuCO4m8vM79ryeG3BmMvG81n3D9exh3GT9eWip+WvhMETiMCW0xwh+rF6gSZH643vrxMBu36n9Cc7f97UDcOLS1RYnq9BczaAA9ThkxaJOwcuE/ouszTbbXpCqqYQ3DnTTey7qFqxTYt8hIKdS0d57/Ie2c84ye4zfg1urZ1cTRI4TYt8A86GbHgjmG4uXhcvVVClf9rFBLDuNaxA59Qq5vaMDNO40clMudW3t5EdyfScoUMkFnzO8JYtFV9rUVqpJLqPmcRrQZyrnpqhPKWu2Ub4nFHdZgQ83PXRG/IhiOqodJ62CJ+9qFhp6hE4bw+FTM+Vsyxme9l0C5nqAHGbwPVBPcOn3rcwoM8AWahHbCR7dMLpp1CS51meqzm/raOCGMU6+aoHgatqqVCGW2jgA6zAvAtnmvO8R34MWT9m0k0al7X7dFBrfrW2u7GjCctU1pGz6LSZO7DEM9Pu51R0voFEFx3bocdMLoHzO7B9PB76zvkcAV4IqYgOjo6wp//+Z/P6nz5/QQj8Kd/+qf46quvprRwlLN48iaqYac/Ryc3bnEdu88TGz4VxuEaEiRu6WbnoODqC7phjy8GG+kdpUQcrgQcXAXDMPBnf/ZnG6mfvOT0IHD37l388R//8RwN3mqCy3a48a0TVlA18XMwo1hv6etuu6vRynrvzXwS17M2wEM3tMlNjOvyNTXmdysJLtXxGH81b/pYNeoSRUqI0x1x2y4/we3lHqOUYm4yqtN19T18iIuL8kSfjcXjMsVEaK7QVfW2S6rHrTmhVje12aU7atUr6e9sFrJRk4PPmt+RmqYEpK7jGdUc/XNduU7mzM1Yb4dV9eXHtd1NN+h8P8zXOGmVm0zr4JDOgdcl3RkH2cf4OJY0M+rHyl3v/ZbTEE+AqX083VJjhTeojp9IF+U5IIp4qKqToJcKFUaXZ1FtNBgDFiAMNO/4jjR+edOQSM6f23kZgmul5fhGyYPeZloXxrxN1NdNjeL2pXt4Mk6obXHAI86/F7T0zmYB7969w1/+5V9GRUfuO4EInDt3Dr//+78f3jKqxeYyZYZMBN9ivn9P48U+9kMOfuIZiop1qOo8A7tZ8bdBKWomv9U8tGpQ34DeS8lp71Mpozjnut3y1HrZKsUdHoJV8YD5U2/zEHkeccQPHz7gL/7iL07gqJEmHScCf/RHf2Qp20e/ou4ZfCWG7AvPfOQV/eVR7vRYJM6P8u1ZH7RmkUnVS7OTHQ9Pfc/Pn3N21gY4NA2QeyI+451bSXCj9Mtu3RMeg+vEeZniohzUoyqm8fMn752YvDWTJNea4yMEzewZ3HwZ4ObPuROj9VcJN7mx6QY/ytTKwcEVpTiYRLFBq9609XDW/I44zEdxy575bq07dAX9hW6s6xbp8tVTWT4/pbrR3kRcY8QGLXHb0DV3Ive3Szw93i4OGWaeGehc063LJVu0vn4cEyaI+LFy1/t9nzDg3PG3qjLTCK6HxPpFBCPH3zL1ClVvS6Uq7dY51FoktgHquguP74j96BJEHtHQGzhCyI+n3KUILmOBLyihjZkE11MvFadLzyjz8n0SCG7Y6f6fTzG9W/IOXvWZPzFim+U2QWBxBFblojwr/60KI7gAdQa0TyFDpWEQdFlzsHtjIn/rxL1qv1lJhRwmuXc7KsVmEnHOteuPYnMT3MVxlScFgVUiEHHP4H/l5ggu3zx04XCtSBqttwXmgqMKWfB8H6uu60IYbbPHyd3pIU51s6RVihMfERaD6wLh2+AMXeRmuUo6ieXHE2yHd3CHohqfW4Imi1w8BX/OPFaz4pUXKXrLn5kmMmVtyOtZcVH29aErEkTT7Qrykc0eIK41Z9xdP8zqpnJaJ9Ape9XTF0i7tSKC602x5K4zGq23Beb+nKauPBuVBe7wxL96xXkWKGmhR9xwkAnPFZd4euJi3XvPXjxExl5wSXB7eHb0jp6ykyrL9mHHDB0FNzWbjyC7xHu+cJApG1kPifXXdXb8Lb8zrQpK5Sb68TzqJLbFmTkgFhjfQT1oKYQq1+M0Gozz4xkWHaWUeJe6edJNssP4+BrzZnbLwSdFkfrEqkeAi7L7bfcfRrj1dtzX33pScamYdiXw1UgzRrE9YCgN02iUymxTamN5OBeaGPLQCUJgVQR3Rv7boXFmmmdFmMdLANyRCO7oOXt9nt+Ce4I6Wpqy0wjsAsFlLFAj8wnogWSd9D4tdXGd8vi/UNVxtmHEjWE4YAxDD1TWZzxBIvg5rYZM9i6O4l53KccSO/Y3b48HqSw77zTotqzz9ymVtBcQY6UiJDs9HtdU+al5cAddNPu09rl5+lTKmiw3gRoFcdZUn1UUuz6RKcpMuWkJON82RZKC2+Nsiv0bYCWUwyT2da97srOB7k056Z7AfWUEl5g5bpJcpHD7KfO4Xe+gZAkRraK3I5YxYMxm6hZeqphFnRufTb7b5adWKp/XXKp/pGu0x+nU8VbB8DDQXdfHCautSp8M9LaxXZzj0y0KgV4xi8TfqgZFI7j+A0qb9Knn/aErOvMpllCodCiLSjmteo05F6OmNbHnwrTxbdKDoUWfylSOYoIhngtDt2BVO0u93xN/67e6WxvsKlKM/ws+THY2EMaUNHrDoRskMuV+n0M8nZy+9B5OW2t50mPxjzg1jvs2lX/UVDm+11URlUN4Jbmf11XBTZe7gjzpoVVeEcGdGn/Ll3ti/EdrzABdGnfSaQ0lriOm0UfHdltCjodkyWID9TC3JSG4xGvN81CNGZmL4zOH+dZ15r3fvD7PThBcYsUNd4J5wN5ZsDHgfkJhM3gl8n7MvXfsHVxEhpuAUoEJlDMpxGO0bNIidPNlEndejef8s12wwq0GVhxWt4AXdJXK8iW2kilPvV6pPIjTF/Xop9+b/jicrPdNJbi+ploWf1p0f6DAy6kkuEMLIDetnjQiax8RTrx9wkOMRmllvJYlNVcp8lPih93L4BaJT1whwbXEkRJX8chepHDW63a7dvDUR1WlI/oC36t4UKVGnNzESwPewcOH9Lm76HsFnnh4mM7cxWuGq46snc7Hh2lgRvGXDtlKTooKWm9yCM9UjxfHFZmnhkOXvlFe9TlV9CO6KI+0HUxqENA9/uuXzOHOyx9/a+V7vo4n5hXcazaD42zDum3m+Pa4TE8RW3K/ifuX7zHHbQVxHkicUwIV6vKEAYGHs6quvcK0GO4ZHk7etriu5/7DKme90a89pLu2J2bRHc+xQzwe2JZmdSkL7mffvcPB5UOkhyQ+hmQ2h1w2jyw3+VGPDDY1Q1TqlywVgrqMM16XC7X6xhXjrtr0plq2ze9R4UfEnAYRrbJq1FdDcIepucLUv4MIrvpOco3RHQOP7S1iRHMjPuUEdxPz0P5MyVwcWxnUWp4sI95kCrdpFr+VLye7QnA9VtxorsZumqEuUvc6VLsz0em00Wy10Xv9zt58+K69gxs8/W5NxOzZ7sZauJWVnZdPfgEtdx8ls46qRtevbhsBaRx9b3Q2JJlvI1qjV977p6bAqARXbxeQvv4E72hl+i03ytu2UfJ22FosuEOSdBwWQKq2FtK4/sTE4b0aVcfbqNaBajNDheW70JJfoVZOMd6WuVIznMdOblUXExuPBL51Yl4HHVrJhqEGIUN9pQTXY8Xd2zR+bl5gWo8jHKyte+IbTNSe/KKDNNNe5M0GauyKhHGE1++9avNOfFlqJCZlE1ENhTCLoZP+R5tqpXfHEXDjfhUZqh7Xmkd4pxb9udPIzNjIKq+fHL1+eKixt7+PmGlaiqPGW/sbc3CHVuyRxPcQ9qGglJlGsd5EvTDbvXb2+Lbz837xvS16NZZGx9vhykWZZLDRjyEWM2lVzKPVrUHluCs8UcF+bAcdf02TAmlMwNme9iFzNuAZKmB3Q10VeGCcyaCmvcd758NrYZVr8kzGyS0wYBwtLdpKYKtUplDOoIW6Sv6Za6DTKsJr5DZJErMpppr6sMeq2iu0JfjjtpHf8afMH7rRfdSUCWWnwGKw05o9KqyDePiF2dY907e8fOt7ZoE/fhi6dLWXILg6x2aGHj6GiffuhGDsuxrL8WwDPe5BRs4XnDtOLurL99+gk6VHIOdEpjMywlj93pv0oFCWSr5i/GoV8GktiefqwNH3U0x5F/g2PCfJRXlT81DBKnMxYIJZB5kcvxvdn+wMwSVgSkAi00KhR4n2SblFH6J9JsVu88NSRm7CBYzy7/0eOr0BdE2Dmc7QNTmHbDLEecjZVPWnWmRogtfVahKjyldEJyTH0pCJqvi69KJ8egtQlvJqWrlMch9H0YZe9bd0Chit5gY3RBXGcSlX5XRCw4AuykPRmy2FbfUEd0tIknJnsb7M/OBzLrm9NExdFZiKymRM7q9wq++IhSmPD5Lhni+12ERXrpjgKpfWGteTVqFHq8HMRWpFI8slNtiA0vUcVfb0Y4Ixk5lPbk3kv7WscbTa/ahczlUqtfh1aKVpFkMVe81wlb5XdTmkTspNzNrgxahblaZXzdv5VfSnWnDVmk/irtZ7910cm2bDTRs129PI7LdQ5jeq2Y8jX2dcaTEd4sI6z/hWH3Wm9RnwAHVKd1kbYL4tMbardVOjRPuO2RaoDB5/nP6uyKNm2GfstYANt30A0kX24YBppHw+2CZdN+tF5O7Seh6atz5yTVZzo7WpayEb6uI9/2sGzRIqOnN980DCu9OQTXUIlpb3Hz1DxtLJzY/7+BNLENy5X00hOhpMuj1+U2I5FItKQNEtJCz+1kCnUuQhlu9lA+ob8GDrSs6/zsSQq7WYWcLXypMSg7uGeTitG2UuBqNjhXFRy2FW6ObcUyT0gV0iuKtr9dwlqXQfV9u5oRvy3AUEPGC5NveKW+8Ku4q2HlcZXfpp5hmvYp2UOpL+9ml/yOk/79n78B6Zh+ME+LjqP+29qyW4W0qS5gC+T0vZhSrjM6sUn2rGUFPeGLPOmlZOcOeo8IputUMi3m5G6XrROruudn7iYXm/lGAU6SrbrkMvcSNXyUz3nLA2rG3kXihXdV+FlEU1X4ORa9PS7/44Eir7du6Y6JCNrOO184x6fxfvUwxrKMA0chPeoxeIrg4xI2BmkozWSyXcZUqOa/caaJQpcui3pkQd3ypuNs0D3sC0OxEqE/kWOz1Wr7ipeFjH4j9V8d6Jte5HiQmO3NAFb3QyQMRX6KE1TOfEnNs/86Dfw/FlUx3WTc6hmBES9rBQ79pjsVUI9tBYqMhFHnK+X8Mc4lxHOlRjd+VEJoo8lS7Ka5iHM/pK5mIYQMwokfiM3+1NhVIIwY22rDB2Kp+4jkH1DS0zyWjPTLvLilOrIvlUp8hHlC3Q8q+UEk4WAqsjuMqlM0XXYMa1rS1nqm3t8lpk19IblpgBrU9+X6uwl+04wVUxRRkVP7k2pWvbokczZSSiFgbzZP5b753zvoPjlSIq1wdVvPFZ6INyRtoHAPqC1u1ggmuSPP/qlqUihStD91wKMXFneVO5CDMG9oc+RermjXGggnS9VEClYyJdaqFDLYCxr0OE8a1conPc5qpY23VeVmqsanKDp/HOIVw3g4e6itUOap0TB5+4j59ptZknY+LKsbI0Bbp0tw84iFnwZa46d9DhiWyqp4BqecvRkv7Uq7q/YCe4j3EuGrH4UuvikjWgt4hah/oMS9Gp96LGPtNmKQ/HsKl/GgnuGubhrH6TuRiOkC3GacyfsnUW6IG/C8GNDJvtHmWgvnReS/uUuRzfTPqVyA2UG3cKgVUR3JEF8AcK5Mywni2IkFrUEp2KlbN2q64dJrijNE4+cZ5VAqyspfRb66l0LQuUq9zK9X4TJcapviTvO3/nFdPNJKOHcYS904kVN+q/UHRoxHTUnEj3yujz/0lqynZKWXz+nYEbTzWKWyxCd0IsuE76Gp3xnnW6jCaZbqdef4DvVUrXw4fotosL4TVsrqFRM8JEppCZbwOtXPFyJFS6LXi4tstSV6ZoyONVxzbOqLETP/xgkMbtVgtVHijYvc9DEoaZFPM38QzbEIPrWA21VVqS3ZypB9QD0WgFGWf4sqmeNnYc8br0Cq3pa5tccxTMsL3UhSrit6v8vjYRq/Wmi/hEJLi9eh61roF+9yV+opfK/vkrjHuPI8uwv/I2K29OQLeOeTi7f2QuTsHISX012IgIpxDc2aPVc8cqVNgUochSAa9L68Oqtf3maozcvNMIrILgbiTX7dBbYYWn56vquV0luJvIdau8VlLX0S+/WdAayFyv7TY0GoDHrlgK+XmJW1B/K6EUirJku161eoPiYxXU2pqVMiWdK6FMd9/ImXgm3jMl1o7W1matjvZABfoyF2ueivx5b3zcqgZp9HIGXeZdTjNF0FqdgtapThutrYbWQaNBQZ7+ABoHWDydRjLBcBTqKBSyySXEAW0tDW8qQYN6APGoHiFu9R3djt6NGa54Vpy6E8c9q+muezKF0vyeC+rRaZtqk6nw2l3OiClaI+owallPjVlNWPz3kW4Kkhkn84VdmsG2dbomUvksY1PDXSbs9GPZ1cWLL96YFT+pPGBscbuZDiMRCe6KK7hAcVs8DyO0RubitLm4onzuEfrBTfP34O2c2RNC9oVnPvKK9N4dvmnQzFPoqozeMMZrjsaQ2GYaObSpHrmIPWGON52YWw0qfBZKVM7lIq6UPdOlBqrJOso1zRJJSVfpyldawVEBN/SlXAVdKptWGA+YapdRbvId3JAXa02KjK511zh3fy1NcDdAksyenVv6dTyCKNDcCKzggV0kuJtQunZSxDx5x5RDc8etrqBfohah1Hcp6FXukehGfWau+5TbcQ69ikZxq7kePLE3d0tUNWeM82IW8S2FxWTMciGHUi/G8cQULO0W0yfR6tfO4jrXiJ95GD3X99qJOU/Tfd3rYeBtvVbLIN9MMl9xG0/6eTyn4FjO84npq/CDWgwVigspvTrbPVln7uJ+YFhT8KbaDj8p9JIse4BnzCCxf/gYAyr1er9mKt3a509oqmNqr49b5mWjviHZXBUDEttMkltWy6q4jyv3mamim0dJo0t+Osa/98AOpBZp8F7Axq+PG89NlW3sdF5qXFLYT1t7fP6C8O7APIzSMpmL0+eifdiUmtARiILtfPeIBXc+vOTujSLgSrs3+irtEs8nqSpz5iZj3xjj9ooxeH0rLUZ6mBrGrtwATZ7mlp69tcSkDg4bIS6Dym2phpyhFECVKwtdMasa0pVP8fXRHmPr+kx9wS+qFcNBq8VKVRiXh3EpgrtukuTEEn7NPlDXKE/o8u1eaQk7R3DXrHTNDUarXETpuyNwu4s9by7blQIvhQkCW4KAuxYmPHHTKqd94qblVn+WugT6rIT2vqbYuYb1cDJFohEn0VB5cRMW8XrN7w2FDUuuDc5xqX3vikk54lm6+u4F59IN2lRb4Sc8VP9REXQr9u2INT3AnbFvGYn1GZU/m9e2EVwntUjW65LtZJ9Q65O9D6jTNZcCg3d/4h94IPdb5gAPMmU6hw6JkNRdWzIa11wNWnsNZifYrrN6u807MQ+jdY/Mxelz0dUSOHz6UW2v13gJwV0juFL0cgjwpDWfonWWhNU5cbU3Dm+ZzpLKpZU+CrGreJK4Y6cZsVdJW4CmX8DDRgUZiq1Uil/jmR6Ql1Q9Q7XRhtrMMOdqIcX8kP0c6rHP8STrySfokCDuCgJzWi7XxsWfXpjg0lJdoOvpk3ery5lqx1rSDY5xcW3mnz166800Pan4uXirV/zkThFczUpBZGtKrSb20UoVM2B8arNJF8YuXtLCM7r2Tre1Y8VDTYrbRgQ8B0YkffVhmhRqbVAnQx10Tlj8lJWpWKRKO/2HMhW0mKc36WuabaFI4PZYme5N6jCVSu9lFbLhxAi+9L3HFpT+2AAAIABJREFUdUc+66Q9c2KeYw91prcLZicTm2rLTbqK1A86NT/c97AOezfw3CSpdavjKpzzvy9PsThvvveofps6R6x9IRLKzfYLi447Am8602JdAIXkJ9vmrbSz1uuz3MY331B5I3ZkHkbsKZmLM+aiM4ftffxcvjERe8C9TQjunIDJ7ceHgHOKfbRPkY0QZUq1EUgwn9uYyiaVqgtpqgTH8OVzxs3R6qusvMpiW6V7WJ+Tbbhl8Mvuq8ZubDLOh+xiBNf9kKgFaB/MLb/E5U1UP6WYK1ss7LEzBNdN46QODkYprhbrvIj9dnZL3coXa7Q8JQhMIGApQasTo4k1apRa6uFHWlPdJy0rUxFmXXn3wCJg9exkmiRbzZsHiIEEl0rZ+R4yDIHJOLG6L/cO8VTlgXbeYyuNHpGLujG8zPvb7CPFHKjJkH6c3FTzcLdioNQoIDlML0SLtE/cxf2OTFp2lxwwWgOFRhw19f6FiuLhc6GJZINeVh5O7yqyM0HX0MXRIJbFpk4RpCZFkEI+ajuz1i8E1k4/tDvz0IF5xtiWuThjLjp76oPbNBqNThXXMIaF4K4BVClyVQgYnSKVRLvItJQVlxuAM9epksnNwEd3M0CLbZnqvBV+TBVvVR+xYgpar+RTHzXRq2WRu0vXSyfX7YfzX+GFit3zfDxt1wljbGNi5St+xniBN478vsENSqWMRqtvCdmkCkVuTNpoqv9OZFFn/FZxcVWbyNDNT3Ad6/Yzr3U18usWvHGPXm9bKC7ltmZHNj2u58KCnbDQY1vrVr5Qa+QhQcCPgJ2T8Ts6LUxYLl03WAo6jeJvnTy8CR7YOd8Xi2zxG/VKH3cbnk5wR/VwXfX2v3wBw5PU2bYAf8CVhx9psY3Wc9OEbVTqqHOWC+9ZfGVZdN0yXWVm/vf+fIrPbu7majeJOoWe6HE9dln16fqsxdGaMuUuT33pnvzjPLGkO7LWLw3RzhWwe/Nw1tiWuThjEArB3blZKhVeAwL2RsFx3zJVLOwzfPB82HT6LieZJqQ9dLniSXcXyFFJM/CiamW/18EgnkUmPZnb037fFQxP7R0hJiaa40kTxSucOBEt/xSaEgpzY4LPcnPA/IyV9C0cjblMrwEUp8j5CS6TaSiXVCX6urGL8T5R1B03Vh/fi3Zl08N+o8baRi+vkuxGXywvEwQ2gcDQqumPSXUFnV6Pxd/aVtXBGEEMU+aNRnDDPJL88bfRwAjfVHtIoevyPCxy/vhbk2JY5WIZzX4c+XoLjWI6IJWVYwH3hvpEa8b0u5wUIyqSeGTdjljwrqz1EZtzYm7buXk4e2zLXBSCe2LmpzRkjQjQDyxJy208HYc+4Ac1TzL73feIXb6C+KAHPV5Eo9vAQikuJ6rtLFyxszhI5VFW8VH1HlJU02wXHcI8YB64mo5io2SleHJP4C0hknydyeQfIf7VCww2ILm6CMFdY0/tZtGy6dnNfpNaCwLLIuDOfX9MqhIcZPztrbH4W8fKBK/bvkscvR5FdqW0cgKXHsRCXJTdigd5JFkfFaQ/pfL8nBbK8E31iMROkMI54m8NjfmsC2W0DWYVqDdRpzrzhDNwn55UlQ4PUvvovKTF+OAKFY7j9Kpq0C173lg7gzmzc5brcbqhMcyIrlZUpDnDQ251TVi3aUlP0gOr1y0HK17zsDpx6QFia3eLXHZgnrLnd2UezjG2ZS7OmIuOYeja44/gErTGS1yU1wiuFL0qBAyDgh4e6b+15OxzFtq4Cnwvx0LzEw6azK3Zy6DeKKKbPYObYwIhKn8bNmK1FIK7gtElBHcFIEoRgsAOIhBKJF1C6PHkcdeJsxdxqHLVWJeO3rMjvBtzY7Z/cWNop2/gnPf4iOxk/G00bKNsqv0xbzYRV8Jyk1Zs+60mBjxALhcrlGvMosrcw+Xc7DzDQaE+Y62gUFen1YORyqGQCSG+/QqSF76BpcPvqDuP4m/99bUPJaqpF9A9rt5j73TcIrdLSCta357ou3ZsHs4c2+wsmYvT56K9dzXCtXRWNuCF4K4MSilotxGIsnAxXwRP178GkzowNVEZnQRjgj9cHqYpstIy0JVtTKVyTbAIwV0BsEJwVwCiFCEI7CICVOlNn8PdvlfgyVUqp06Bh7gGElbH+pm851P5VVC4yvtTrYWOKzI8sa9uqhS+fl7rRvim2mkntbS8ab90pW/x+SObQE7E35oUX6RwU7mJfjyPapN558OIaEDXW3Xphcf0ujHGilgHC3EpDB2Cu38Z95jjtsKUfulztGxb7ztPTQwVE63+3RGR7BVCUyipu6JZ1XdxHO96nXdrHs4a26o3ZC5On4tNyzDEzCYfVYrOdV5CcNeJrpS9Iwj06lkUK0wg/2EP568UUW05olUT9e8zfUQGXw9SuBzToJmM433HLcLlQypVdvjfaVQ7/BgH55pfKRpCcFcApxDcFYAoRQgCO4oACWU++QU66dto5E00ah0gYeDo9fux+Ft7rcUYGbOticmR+OAYBI5Lc3IkSBWEkNEuIH39CXDjPqoZqgbXmjh6p0QA50+tNk3YBio1XLqAJ0zftre/T9diEyY9omJv31n5rifz35LoZ7K4qyVwoxEWZxvW57NjFK389kz1o1oaLqSlXJQpDNnoIxZjjU2GKLW6qKGCbOEJibmdBcA0TSRyYbnu3To67uS6KMNv5UzdmXk4e2xPJ7j89dTPRQfDzCayawjB3cr5LpXacgQoVqXzg0v9JOuyXKbjJLtLpd2Zr81bTXB9+MzXsg3efRIILrE2YvEAoZcN4iivEgR2FgGT67dBygfEEjG0JuJvlTFR5V9PjSwOrthgIdwltkdZ4c++S9G7hwrDwalrHcRUWIutIBdT7pqME317/h7eKAvmHJiGE1zVPpNtI6FV67JSGeR6kegWbNFGRTKZ/7YbUMmhoJQWQ65GolvOBMe3euvpT7U36NC9OQd6Nk9c6htWzlAYcpoZx6rzZNiPK5gYSQzPEafSJf52jhG16Vt3YB5GHNsyF6eMHUehPrP2+FtVByG4m57F8j5BYCUILE9wVYqlEvr5BireXEmL1s7aiAygNWsU4nqCnzKP8XG9CgKL1nT03M4TXArVxFTqrBt4QSXx7KKIGBRxaWuU1xld8XQeeQrEDC//PUl6PYSplS9aD3lOEDhWBNz8tz73OaYFSnzWQZF5bWsplfYsjutaaapLLANxkfnkFnD/F/TKfoarLKR51Iwc2hSvceftMD1QCOGcBk3wppqp7VKX8A31nvZuvIDZct80clsGY4BfMd3OVKcjnanxSgVQPwrZaoNEl9kLQg5zTQos/upWn4kHdNTTdMNOlJHqOSn2xhqg4mbT6DcHWGsqTL7TTpOUiHDYcKyDT14+RGA752HUsS1zMXwoW2k3e0X8oHN9XPuIF4K7dojlBbuDwKDTgk5XrlnhRkavhV6iEHgqHa21FChpaUgUuFGI9sDEXUsTXK+oVmVehcuASjMXYrM7QCJtoHbpa7y8JgR3wa6N/pgjnKJi63QqjE41FE0tVVmRuiinruPJ+wN89aLHDbjfI4H3NAtI3uwj/7iNei499GCIXmG5UxDYYgQC89+q+hpo5ZMoGUXGgnLs6yW0uxVkZnjsWJbfeoAmgzNvyTpHeg1OSjqd66bOg8F553Lgptoi5t9ByUid/eoH6FYCXOaEL6fw2QOG1uxdw0NaWItRl38ecjVJdMtPmAH+Rh0tCi16z8Csnu3XkLpQRfx2FYlOE7Faj9gFtGbA+3JAZ05L9fyjxyZLvfKPFtGVawcQ2NZ5GHFsy1wMGWOc8+lzVSSfMm4+aE1Y+dAUgrtySKXA7UDA7DdRyJfRpfew+Z5xOtdKqNWqjEXiiW6H/yz3Ueq6H3daM/lDJdZEjyqMszYXagErJV7ACFNsjACBwUU8UzRR0xgHNuuFAeUtS3AjiWpFaMfkLY46qBDchdCL/hAtIFmmM3kZpoCqStLRqRVpUe9iwMg77348nqOADHM5e//Wzp/B9WdO3mluPscvZXFJ0uqvXAoXGLDRGyZ3CgKbRUDlmdb7JG853H1Jp93zd/CK6WaSifFc6ZZbLJ103dCU2ZXUUE5eQivrc2VWh4vpHsp9/j/JWdopIfv5dzBuOPnVZxc8cUfgpprxfnklhBi7jNv0rMmbPTSqNTz56T32L99Gu1PHQs47VEFulbmuNDXEAuNfPW7RgW1RuOT4/dSxxCc0EkpdsverdMeeaaWOVJrctFYEdmIezhrbISJTp34u2krn5fhiB3iLjTshuIvhJk9tNQJmr4xCM4d63XGlMgfoNihPQcGMvplErsy8fJ54IuuknZuMN1q0uKdVEFwF4IB5+841cvhRnWbPiehiBNdAp8LcwX1uqnrPKGhyFhcPM7QiMwVEuzzdTS1y/YTgRoZqmRvpApmlC2QvzHqrxCxoke1m7qNV5wY6zKfQUwc3r/NFlSrLZ9VXYzXTr4en4VimLfKsIHCMCBhaG23N66CvKhNDKk9vnmXPcqx5WIRRHz8YMnp1VGptaDyATaRzKJXLyKUWF3EIjftTKXnqdTR6fBHblMwy9CCfj7QezOwS9V1t95Gc1xOJ3j4tPc0UQcuCO72GSswqSffwZn+xQ+SZ7ZcbVorASZiHChCZi5PDQmUYyXZL6HKPvQENVqcCQnBXOkGlsB1EwHKbIFlgfFXUWKBVEVw3CL5bekNCkZwLvMUIrvuKEDVALkDJfGssFnNqpVJVaDxMGK+5ENy5OtJ3s9GtoVSlddXMUZG7PrTuW39vxKnwXbIOQ+zUJTruWHGB/jfaMXaN9AvGRM/2SBg+7bqGOXknh39XboQ5xh7OitVbpuHyrCBwUhEw6P6fqyPVbtPzZz2N7JWp4J9qo1Oa7zuyntpsQamM8cvnDVQiuJJvQW2lCptAYAPzUDVD5qKvM7mvzNCQ06bH2JqWv5DRIwR3E9NK3rHFCFhql3RHm0e1cnUE1xHAYJqieYPulyK4fjXAlfaPENyF4VSHLVm67Q1oUad4VCfvCsPY7j23BhSEUbmY6XpcS3+KajIk9lbl6KT7e6tPF8W5jCRNZM/cxMuz7ntUS+hOmKLMaYspsjZ39LowhPKgICAICAKCgCAgCJx2BITgnvYRsNH2D5p55Cpdqu3GkWIS+U5tZF3SahnmucujQxeGed11JxqhNZArVjGggmODwjhGrURrGP1y4zmUmeO2ODzkpspj/Cq6RQpQRDXf8mWBBHegYn6r6AyYcoKZGJI8sW8UGGNVaTOlUAyJfJ1u0gFCQBbZJLF4YcwVj7QMwV1f/K3qCSG4i06qPudAOcG0Gsky4lcfIT1UUyVRPUPCe+M5lVBVcCxzRNYqlgJ2KWCyqLGRpmW9xzyboSemsaBYQpLZxCU8eHeZiqPM5UlyrFGQpsA+7Qu7XbRb5TlBQBAQBAQBQUAQ2CgCQnA3CvdpfpkSZEpU0+j2i2hReMPaRP+Wm2gr7MiVhT/E04/czLtAkTTmsyU8Y4J67B3gsNFFe8ROR3Aq18oaRSustDQsk8Q2TcW7XvIqHlEV9ssXVH+kmoaVLqCawlPdsWypvIOf3kVizpxcEwTXUsBkmT9oqKu4Iv53gvkM3+1dwUOq06arSVx6YLAeQSTWJoR6QNzjtPGyDMG1YkR6X+KFQWJNstQj+U/lMogbVLxkTJWdlTHCFU/TDSztE+USghsBuam3WOlC2jk8NZ254LgOpx9+5GHM7NK1cgKZNmPN0+EOQbEMD38qk+7LttDUvj1W4xzXFGlRcTPJ2a+VOwQBQUAQEAQEAUFAENgCBITgbkEnnIYq6FR8zaBXGXDT7KQu8Cazd2P/Ln+LX5jawPKqdFTn+oWH3IgzvUGnguLXzxCURkER1zTzCVqqxiS2GW7wO1RozJK8Gt7k7k56hsOnHylTzndY1lOSiMD4W6WsnEGxn+cGn0TY001+gqtVUiTVHcZeODSAhZ+5/oxZIJTFLU3Xzk/xdZ/k3SXWY11uT8J2fj4r8uIEVx0AfIbvUg/xkbirWM5ktwolyEGHaeJGC3vEIRljnofCRE4lIbgR4Qu5zfYqeJS2+0ddlgjaN3Hc+5n5JCMEsVjxudWM4848X21GQlPfIlXjAcjAtuTKJQgIAoKAICAICAKCwG4gIAR3N/rpBNWyW4zj6qP3uOJJZu9uqg/ujHLVqU16goI6uiK/bvtJEAtp5uqMfYnnvSZyarM/oEAUxY4StDy2cp6duEVme2NWU5sU6rjxnPGJis9NJbhd5s+7iicfJtOwzIrBddt4mW3szWQHmya4JO7M6Xhdy+JejvGeWh6t3gqU7fQWSiXmEdY1PDtijkUc4PIhc6UyX3CjMcVV9jjHthOL/HarUhrZ7si926+gW27BdqqRB6Y3LnYGaAYPGRJfAM2PTOkzJ8AqfvfzJ/iAPVx5qNNiLOx2TgTldkFAEBAEBAFBQBA4VgSE4B4r/Kfv5e6AG8X4qXhCS0DnyHGLzNqoKDJaTGmTBJG5/GpZ5is8eo+9/T18eP8B5796gV5j3N3ScvN8ksXjj3R/tkpUarPncHdwA89NEgD7JbTgNpBxCW/EDpkguEyfkslVgTLrWzZRoUX2m7fjxFgjyeumW2B2It/lJKIvLWbBPXvxEExPOn5FIZUq59xcOR0jgrNLt7kE9+xFHE6CiEKjgUIEi+mqm6w8Ai7VE7hN5cEBczc/+4l0cxh/G+1tyiKf+EJD/ikPfuZRmmL6oQzTD2kk/TrJrtDbaHjLXYKAICAICAKCgCCweQR0GsNKLb/voQGt8xJvP1zz8ABv3XS0SiWmLPPV19DQefkWH3yGjzMfeW2+afLG3UGg75A/xqZ+pCsmK65y1qY+e4C38MffdtAlDc36yZvTWNPoM3Z0gDhdn9MJ/zbcIdIeq5e94e8i+1inVcu937aW6ZEsrSOU/QTXdRe+eI/pfgotpM7dxU8YTSq7jV2U3tDF1N8eh1AkXLfpiJ2pBIkyNQpnBV2pCnq0yi4t1hWxLjt7m3Jnz9QQjGIKlV4P5JqbvUyDImzMjUliresmYnTZT1zvIPeUaXqGwenRqmT2asjm7+IIJPCFIvLp0TyJpfKheSgHnQ6QYw7JaK+RuwQBQUAQEAQEAUFAEDgWBKbuh8kjmvRqm9w+9VHLZBC2jUauydC90VNCcI+la3frpQZT8mRzD/A6cRlX4gP0Xr+jOyQvb/ztsk1yxKP6Z88iQTGdcrKDesukSjP/SbEp7xWalHtKHSYsuMpNmu8ZJOlSqulIl7L0nH7E/75CktmDZmZRa7VpvbUUtcYvy5XapGotCbCYy5bt+R1/XrfjtV87hz30ViinPkMjtZw11dD70Lo9DLzqYYksijmhsDs+YKT6goAgIAgIAoKAILBmBITgrhngnS+eKr31qlIVriAbo6WK6XOU9fPqo3djMblLt9MijV071jajXHHjSMQDyKV6kRVzSGIxB8EMi8E1DZre4nG4bzLpBmwyHUvYqy337OwnTM3ipn9ZuuVSwI4joNyTc90cKnkKs9W7iBeaaNfpfbDj7ZLqCwKCgCAgCAgCgoAgsIsICMHdxV7bYJ3tFCQgmf0tuswPpDNPZ/r6ExgX76HHVCRepeJlqjUZfzutNIOCUwkr/6iuVJgjXLNEpiIUYd+i0gtlNFSZSki0fCKjdgpuNGGflYQcypwCBKSJgoAgIAgIAoKAICAIbAMCQnC3oRe2uA7Knz31xTOY5y8iQbfJgclcnUxF0qTy0mq8c1XQeB7l747wDmep4ltBvV3GhK6THyOmJCqkSjAbfcY5zq7JSgiuUrpNlhFvzikCtMX9K1UTBAQBQUAQEAQEAUFAEBAEThICQnBPUm+etrYYXZTzTDdEt9DKDFNyt5S04nqVO+li1wD1XBGDSnsiJnix8uQpQUAQEAQEAUFAEBAEBAFBQBBYNQJCcFeNqJQnCAgCgoAgIAgIAoKAICAICAKCgCBwLAgIwT0W2OWlgoAgIAgIAoKAICAICAKCgCAgCAgCq0ZACO6qEZXyBAFBQBAQBAQBQUAQEAQEAUFAEBAEjgUBIbjHAru8VBAQBNaNgKUA3vsKL3TmPF73y6R8QUAQEAQEgfUjYHA9/+QWBnd+xKC2qKZGtGrKNyQaTnKXIDANAaORxSe3Brjz4wBrnrJj1RCCK+NSEBAETiQCKl/zhQdvsXftMXTmTp6ttX0iYZBGCQKCgCBwQhDQKBZ5CQ/e7uHwqR4pg8IyDZdvyDLoybOCABFQqTUvPcDbvUM81duIkPRkZbAJwV0ZlFKQICAIbBcC7mYIOHvjKTSmvEpsVwWlNoKAICAICAJREDB7KKc/w4OfsMFDS/mGROkauUcQCELA7JWR/uwBfsIerj3W0Sls1swgBFfGpSAgCJxcBAY1pM/dxWvVwoNDPGw3UUxvdpE9ueBKywQBQUAQWDcCJgadCgqFBzh6z3ft38BzlZN+3a91y5dvyKaQlvecFATMATqVAgoPjmBP2ecwWhubsUMUheCelAEl7RAEBIFgBLQKUpe+4Smic529iBvlCkr5LNKphOW6/Fd/9Vf4h3/4B0FQENgqBH7nd34Hf/Inf7LSOv3d3/0d/vqv/3qlZUphgsAqEPiDP/gD/OY3vwEMHfpAQ7NRRaN5hLcfnNL3D/F40MaGDUF0s5RvyCr6d54yfv3rX+MP//AP53lk5r1/8zd/g7/927+deZ/cEB2BbDZr3WzoOgZaE41qA82jtxhN2ccYtI8nREwIbvR+lDsFAUFgVxFg7EchfR1P3oU0YP8c8P7NrrZO6n1CEfjNfgz/xnsTxv4+Esk0cgUezJRymMsJwdDQaTRQa3WgDXS8f/+7ROv/OaGISbN2GYH/kKeN/6sR0oLzFAzsUWDquBxw5Buy0aF18d/7t/H6X/xr7HMNTKRyyJVKKBeySMbmqMbE2rfHh//VHAXIrbMQ+D3e8C9Dp+wL9CgwdVxTVgjurN6T3wUBQeCEIMAYkFoRpbvf462/Rb8+AP5+4q8npN3SjF1F4De//h3833//ryeqf3DtHi1bFeSmBZXrHZTzRTw48p/q/DtCcHd1QJzweqfZPs3fxr0DHDY6aBVTmIfbrAcq+YasB9fJUn//9/5N/B//8v/1/bCP8zfqaDeLSE0bDKFr3z8SgruBDtxjOFij00JxaietvyJCcNePsbxBEBAEtgoBg4JTVVRrbXT7b/Fe+dL84/8M/+1/8u9uVS2lMoLAv/W3bfw3/+z10N1rHJGzOHzYQ7uYnABqwFPz9K2XVvzT5JXA5f/iK+ToCSqXILA9CAzw+r97hPbfU0Rq/wCpdA6FSgmlXPrYLEDh2Mg3ZN3j5nf/+f+E//rp/xn8mr2LuN3poh5gzp++9v0e175bsvatqvMYLlB/9JrfmT3sH6SQzhVQoaU9N5eL0aoqM1mOENz1YSslCwKCgCAgCAgCyyNgGtCNAQ9mGOPUbOHZ65FV9uDLF9CarhuYgW4xjauPRt4IZy8eolAsIJ/LIpVIIHFc/mLLoyAlCAKCwGlDQMVj63102y3UG00cDQOy93DlW/695B7wydp32obGrPYKwZ2FkPwuCAgCgoAgIAhsEwKMLWsW87j5zCayB7dfYVBPM+VgEpeY+9n62+FDy5VvSw7Ttwk9qYsgIAjsKALmoINKocDQC+WfMko/I2vfjnboGqstBHeN4ErRgoAgIAgIAoLAuhAwuiSwVx8xpnwP/1Hln+B/q/0vdGc+ixtP+2jlxVS7LtylXEFAEDheBJQrcophGB9wgNv/fRqN//KZrH3H2yVb93YhuFvXJVIhQUAQEAQEAUEgIgJaGclLDxzhtAN8+YLW3WOTmo1YZ7lNEBAEBIElEVC5VRNffI8PZ1jQR1n7loTzxD0uBPfEdak0SBAQBAQBQeA0IdD/p/8IF/7Hv8ev//Mf8a/+h9Rparq0VRAQBE4tAgba/+Q3uP6//3/4zZ0f8X/VZO07tUMhoOFCcGU0CAKCgCAgCAgCgoAgIAgIAoKAICAInAgEhOCeiG6URggCgoAgIAgIAoKAICAICAKCgCAgCAjBlTEgCAgCgoAgIAgIAoKAICAICAKCgCBwIhAQgnsiulEaIQgIAoKAILAYAgN0WjrShQwSixWw3FN6Dy0tgUIuuVw58rQgIAgIAicIAXPQRbs7gIkYUvkCMo4wvNFvo9MzrL/H0znkJRfaCer11TVFCO7qsJSSBAFBQBAQBHYJAaONQrqCWLPnKA8b0NptaMZ8jYin89M3WSbT9lSqqHc66JspFJsd1IdKxwa6xQyKZg1aKw9J7mNjb2g9GOkMhPbPNxblbkHgJCFg1DP45Osj4PK3+KVXGq6P3WIcebRhNLMnqbnSlhUiIAR3hWBKUYKAICAICAK7gkAfleQFdEpvoFVGNGpQS+Pc3ddsxB7292MzGmPi/fvE1NQ8Zq+GTK6BVLWKdLeCu8/esegbeG62kBuWPkAtfQ6N3I8YiBIo2S2xSVSQ0gaoizDqrkwoqacgsHoEmAroDFMBqevg9isM6mnr3/vlJNfPDv9bFojVg34yShSCezL6UVohCAgCgoAgMAcCFpFtZPHjoI7xLVIXxcRVPHq3T+Jq0LIbXqgqQ5HWQacQbHm1ctQ2kLjfR48bMqCJ7Jmb0K58C61bGrdO9nnvhS5KbzQS7zkacqy3GjCMOOJxE4ZuuwwmEpM2aNPQYZisaCyByZ/dMgzoOktI6KgmL+HB2wPc/lEI7rF2r7z8VCAw6LSgp+kCfCwxGib6HVpiMyMX5DHQFcFt5vAYFSiee+2xjk4hPk5w6SHTLJbQinHh7PWAUhPtchqDVgXlahexdAxaR8e//199id/9nx+hT4+blNYGvZ+RqbeQ75RR69EVmodqnV7Z9z04FUPgRDZSCO6J7FZplCAgCAgCgkA4Aj2UEp+hW36DfhCbJGGNff4EHw5uBxBgp1SjwQ2v8Q9BAAAR00lEQVRhE8U+ywokpLZV9q5xG69Iom27gyJxQJwsb9I2zPtT5+i6/AP0RmarO8/QmqjS5brx/VvEr9xAkpvFo/cf7Dqfv4NX/ZrdXssFvIBeljFz5T5Kma+h5R5j0OaBwKCNcrmK1rPXiH15G6nWA3zPIn7nP/gM//jdD3j9bg8HV3JIpwpoNArHEx+91b0glRMElkWAeWQLaVRiTfR4kqeOpgbdpkX85rqSWRSzgYugUwxJLMlmtd5Bp28iVWyiU7ffZ12DBrLZBnJtHu7ZC+XosghuER87KZTdg69XA3JY14KbQCPzCappZ91U6/Int2Deo2dOro7EpQdI3n+DRqoHkwy+kbyKdu45dJbbo5vz1Ucp3P+lh7KuDhj57HMT/OmYrgH5eRwZN9j4mGpxUl4rBPek9KS0QxAQBAQBQSAaAt0i4ldpqQ21EBrWpumWFfr1C3qlSauksiKUEt2hy9zEi9t5xK4/Q0pttCKaZJXb3QVuNF8YtPRGa8mx3WWQhH+iAOIhwA/9OjIxHU0KZd0kS9278Rwmd4mdQgyfP0kMLbEqbu7qozjuEHfliT0sY+8aHnOTG6vV0M/XkKx9QmuNWHCPrXPlxacCAXW4d6FTwhutMvImcQ/3iMDe/n7AQZwXGhWiodZIeqgEn/IBZg+1TI4Es0oS2kXl7jO8Y/jHDT+RVO/Nk1zqbeS9y+2Q4Bac0IUv8D0OcaegoRWnizIttIrQxh9/5Hqj6uas3eY9/EzxwMwFljdc5/skyRfQzjMUhK7N1nr7IIXHHzsogGERZ74A/8MpZ/NDQGN9LvFw8KPdELmWREAI7pIAyuOCgCAgCAgCu4WATtfiT+8mnI1NSN0HtEKeu4vXe4d4anKT5L2NBDlRBNWPSUQDVaG4ycqSIL+8iHvzuBxb8WY67v1MS8axuAvO0Y9ObNzBbXuzaF3ELEXMfnJijDPdGkqtBGqNorWB7pUS+Oy72Mj1OKgM3tfKnRGCO0dXyK2CwNwIWOsbLacTh3yzD/eG71KhGZzT3aGHir8WmmV1bSTuo0/XX7UGNLNncFO7gm81hmOoPwwvg/M+gWK8bR2ODS8vwVV/tMI+HuAt/9VaeyptpD+9Cy6aPEi0F01r/ejfwY8dE7k1EFzToAphnF44DL1gVAVi8QTiEy45dsgFfw3w2GFIh12EFdrBG0CzMhL0wf5w7bEQ3LkHc/ADQnBXBKQUIwgIAoKAILAbCFgn97SwTsbfjte/W0zQ4vgO+1++8Kh1KtfjNLoVpheiFTf4sq0B35+/hzd9j3VkFjxWHK7X4sAHHAXm1gBI5kvIMK1QolKhhbePOuPWuuU23QzHdoqz3mL/bgzQbddRr5soao1xAh+lhEByyg1tgvGz72iRtawi1ougNcsoN01ucDt49DIuBDcKvnKPILBGBKzDpm45eH1yD/fgncf+yiitgqI65XMU6Ccrawv2GbhNl2JHG4qEzorRCCCEfN7yejFQ/4Xk111a/QRXrSj8myKDCetwDZZV9kGSKsvUNYjDDvVoFhl+kq/71tNlLLhcL2uM6a0/YfjEZRzeYCzwk7ewAzP2cfh4wHXYrrRWyyBbi6HapbhgPYfPH8Vw51UftWQP9UoZ9eYR3mZu4/bgAbUG+MCF/xj/6e/9JZ69ZHlnL+Iwk0SWxL283ZEqaxydqylaCO5qcJRSBAFBQBAQBHYEgagEV8WQ5hPX8ezDeVpileoyN1d0IU418oyVsi0SgZfj5mdvwOZQ+ZwguLbScyv3A/p0CTY6RWQ+7zoud4yfKxfRzTZRH/Ppi94JWjmBSw/S0y3ZYcUFElx7A/ngLa3eH4mdRgXpbBWo2iJbtksghOBG7yK5UxBYAwIkp/GrTE8Wvj4FH+6NqjJTYI8pfPIxrp0pHvJ5XaCntsY+GDTcsBBD49qWwde9DO73SPiG+W659uUTXGs0e31VMbxM94ZqG1WmWyt18+hwDTZILjNMMZS68wrdKoN79Rpy9DDpHz7moVsGnXyKXjZJks8eKozBTX3+CLGvfoDGtTb46NLG7dH7PVx7yANOHoCaSpTqM2VRdrx1YrSM06L82rXEOjHBR4dP8ZEEnqAzPOYR3oMhGHxvoVtFM15hSIwi4yxHLLgrG+9CcFcGpRQkCAgCgoAgsAsIWESLm5jxVD3BNR/GiVp5GJO0LJSQoAqLa5EIeqqdj+F6J41vwwSoTFo1mQInmUmPb6QsYkzRKtdtUKc41adfI3afccBle8ulYt3q+cFKhFDGY9Dm7LlAgkurbexzPMkoawrFayjk9Z0xcvEWgjsnxnK7ILAOBEj0FAlLTI03deay53BvWJWZAnsqvj7LGP2XuDiHBgGT/1gHZI2MHcM//XLV10d3KbV2k0rtk+7CqwLRPcAbt2xb6/2zD2yrcpM20SxVoBcaqKj4FZNhLL+6iZcucbUOMQOIbNjfV1X1U1iOENxT2OnSZEFAEBAETjUCPEk/c13Ht1TPDNCP8kHjbmr2cPFiAka6ydN/jwKoH0jXvW//Iq7lssjlcsjmMki5ysmOVbiTfQzdl17IJtMJ2/pp01lL7fm7d2dx4ylFVWi1cC+z30KlUkM7VmXoaxw1us81eynkUryPBDyWaUBTasV8wKArYr5KQs0NZC+WQ4nKqeks44jrXpEVGkKaBZTaDCbrd2HkfEqn/nY6BHfMfdtyMdRQslwS2YYztODgPEWllHserdGZC/jmtS0w08iYiHfYD3Q1HIvj5XvsGNyztHDoqMd76MUzEGHRUz1jpfGrRCAoFCKg/PHDPeX+a18zBfaGc3jk+RK1+tbcZ1Kg7RRaCia4KpfcmZsvx9Yxc9BBvVxDN0mn6QfP8JMQ3KhDYGX3CcFdGZRSkCAgCAgCgsBuIDCnYqYSlaJb2Tul9qvTshEWeqvS4jCtTS/J9DbM59rvavjJTZ9DYFxV0mSeeRqb+QkXZ2tzZ7qxZA6SVBUtZK6D4V7YO7iBVo95Gy2e68S7pu3NoG0dTeD+z0x50bdVopUluJayXQXNJtVBc3Tlo+Ko4QiyjFlwlfW4moFOl7u4RVQ7yE9LmeEQXNYKB4dUSM30UK1qyPz/7Z09bNtWFIVvJ2uzOpmZIg9F2KVmJ6tT5SlElzJDAWUKiyKIMoWeIm/cKg9tlY2doq0aCkRBF6JLlKnqVGWKsjEbM5XdtKXnPVKxpOjPsWRZyiFgBLD59z4yBA/fveegJ6+ZlUy30ed39OsbPZCd699KzYnl+BGcl2VPvvn5J/ns6Q/y6Dm62PZuyo+Nuni2qV1bB7M/ACbXUXrY6lSzmKXNuLt4liRwpQksKHDVjKpqkTh9vSPfPumrFtnU5MlBPvhUgz0tgdNKDmNGzNoUQBspcAfVLA9R8l0zdPTSrQ7c8DWjzI+BAvfS/0tQ4F46ch6QBEiABEhg3QR0hA1KgFNjkvlLFCIfEtZOLqJwpi74ah8mRbHf9YqpNVM3zbgbSi8H8xHLFEzmvr+g7K/0qSfGRFGJHEnMrpa+VxEbB/ILnJk9zIjqfldzWOBmkRfDL7D5tBwxp/va4jQmo6xexEZjMvLg4USuBJUzZ5O85YgzMpah0x4qUe75eUn6k9xCUaGHskH1NyMbtDKZUWWEExkM7X58u/lXiGuQAAksREA/HxbMfB183Nu9g/gyXzpzDfZwBtkHMmtqfBBchLvony0UZfTxEsPA75qcGOkzrR+1JexEeH5gn7m8WLYztv5Co13iSpNncFNX/lju/YXz76al2TdQmp1mrFPgLvECnGtXFLjnwsWVSYAESIAEtoJAghcPA4IyRH5raf0jUhmIxa4vsTIhyU4nbmH20oRpSuZT1cffCphJRu2uxPXcYgLXTOC0XJSjFnqOg7xUIXL9LmaBcZDhGVy4u8jtRPUZLyb4VRPwpPLi9ZPkGZAACcwmkLYPxFMyvke3TeN7biPf+vrBAT5OubMN9pSLsbUvJy925eCmas+wxS7B7Mkc9MamBlG3wtKEapjM/KqSfoBTi/p3/yQ6m0Fe66UdCNzDofaW1LXZNx7rZ3essoVPX8sOTKUSCH31YyEP/c0BcnnxbM1FvlizenC/fixvYZLV6SRShJsylw8nQIH74ey4JQmQAAmQwAYTSNCjalbQD9rDC98i07grGquKvSjARKqBvtqR88Dvc0H5TPRmjpyxLoXL4jHmzeAqgeu60qo0xDcxyzrkwDIscK2Bw3EW6aFezBzk2LanuUBT4K7obuBuSWD1BM5VCjxwAtbOv7MM9iBey6aUYcZnW4bE6OXvvvovi9LBmNBysKt6EAqONFoNccb1m55ZxodHzITCjiBd9HOmd+a8vno0M44wELgYyt6huFUEoTXhgyAeZpqzNgo4OhfN+/K3zg/ak8MH+IjaPJGn6NTY+fyuHH/xp9R0v8mB3AkCqblFSZ0V0FJiHMFvQWG6IU6j867VY61D3uCDU+Bu8MXjqZMACZAACVyMQGrAZEi9vaY+TwQ52m4kVWTSKtPNkSUrJZRDVzxM4/bqiJSQinQQv1PAi5SDF6nnhYfyD8qsO3ixTCMv8KLUgXBHmZyFMjmV11tR0Rb97OUyj4zFSl0aELQhesVuP83LvWeI4yhFEMBfyumrHdnFW2jOqkkH+x1/B1Xn10dvWcXF7HKE2kHDkooPx1Acf43fCC52E3BrEvjYCGjHdpi4DWfOzmCQoPe/FRXEKU+L0FEbIzonxMyjPeoOr9sS4q6EvRxiwyxtuDdp0dm88AAYiRWaI3Dj0JNyLcY+kVPbs6QW4rnYb8803Yvgf1Bu5MVI2hKZFd37bzmLlD8Plyg3pYTWE5nYbpG2peQGeb9wzY8TFf+b1x4D05ex7T62e3LJ46XAXTJQ7o4ESIAESIAElkNgEIWRvvioPjTjvBkYKMV2bOQ8otw5FasxZgZcOCZD1E6I6F191MZyyHAvJEACFyGQlh57RijxVejRSGDIZKB8Fz23IxFsswSu7g/Gh7lMpOvs3mZJx78VppnuIafW3A/EVlFsCSLfvsJxZ5npjSCe4qJ8kcvAbVdGgAJ3ZWi5YxIgARIgARJYL4GW84nc6j2Ul5j1RYUyjFtC8TzM8rZgPrXeU+PRSYAE1klAu75XpB/00Be7zvoLZZpXknYFJc3VsafSDIHbdvNyBJH7ElUw+lud8i84QokzBKsfDkWgDZvuRWrmGi7Pz+AEXUoNoHJP3qYO0XMXCty5iK7QChS4V+hi8FRIgARIgARIYKkEUIZXcz0JkI2bGMjJdX2pV+2s72upR+LOSIAENo0Ang+e4yMTuy3j2vJyhtKX0C1JALHZcgvvH3KawIU7++93C/LdH46esbXVllrIPhILgrXWniJwzQjHssTr+xJCULto9l0shiyRsOqIF3RRA5NDGzGeo74rpcLsouPLYcijTCJAgcv7ggRIgARIgARIgARIgARI4GoRmChw0TtctOVflCSf3u+KOzClUvFEyvep3xJTlyhPiE1TAtepwrwPvbpo+civc+L6apHeurOhwN26S8oBkQAJkAAJkAAJkAAJkMAGE0jgE2AX5RiWxHuH98RDVne33ZVe57m8yD1AaTJi1ZDbbYeYhW3Y0vRqksNMdFBMJIBx1ETTPbclpWvH0t3dTQ2fUNVSrjYlmDR7vMHoeOoiFLi8C0iABEiABEiABEiABEiABDaQgDLhg1ad4s48MqBuVYq+Kc0AEWzqDxDRVRtO9vAmwOQvly0iQIG7RReTQyEBEiABEiABEiABEiABEhgnEEnN3MfPb9JDBrrKn026dXFrhjSaZcacbdkNQ4G7ZReUwyEBEiABEiABEiABEiABEhgjECHDu1yTZi+WfMES2wuk7ppz8mlJcRMJUOBu4lXjOZMACZAACZAACZAACZAACZAACbxHgAKXNwUJkAAJkAAJkAAJkAAJkAAJkMBWEKDA3YrLyEGQAAmQAAmQAAmQAAmQAAmQAAlQ4PIeIAESIAESIAESIAESIAESIAES2AoCFLhbcRk5CBIgARIgARIgARIgARIgARIgAQpc3gMkQAIkQAIkQAIkQAIkQAIkQAJbQYACdysuIwdBAiRAAiRAAiRAAiRAAiRAAiTwP+8BxUo5V3emAAAAAElFTkSuQmCC)
