<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 07.PP 并行：基本原理

***Megatron-Core：PP 基本原理（朴素流水并行原理，Bubble 空泡率计算，Gpipe 原理解析与动态内存峰值分析）***

**1）朴素流水并行原理：**

朴素流水并行原理较为简单，如图所示：四种不同的色块代表不同的 rank（或 GPU），rank0 将本地计算后的激活值传递给后续的 rank1，以此类推直到 rank3 完成类似流水线的前向传输过程，此时 rank3 开启反向流水线计算过程，将本阶段的计算梯度传递给 rank2，以此类推完成反向传输过程，最终得到各个 rank 的梯度，进行模型各个流水线阶段的权重参数更新，此时完成一轮流水线迭代训练。

![alt text](image.png)

**2）Bubble 空泡率计算：**

此时显然可见，图中大部分的时间为空白，计算与通信缺乏 overlap，即存在计算等通信的现象（rank1 要等待 rank0 激活前向传递之后才能计算），因此对于图中的“空白部分”，我们引入 Bubble 的概念来定量的评估流水线并行的性能。

空泡 Bubble 的产生，是因为算子内并行和算子间并行解决单设备内存不足的问题，模型经过拆分后的上一个 rank 的 stage 需要长期持续处于空闲状态，等待其他 rank 的 stage 计算完成，才可以开始计算，这极大降低了设备的平均使用率。这种现象被称为并行空泡（Parallelism Bubble）。
总的 bubble 占用的时间跟流水并行 PP 切分策略相关：

$$\begin{equation}
t_{bubble} = (p - 1)(t_f + t_b)
\end{equation}$$

其中 $p$ 为并行度，$t_f$ 为前向时间，$t_b$ 为反向时间，pipeline bubble 占据了 $(p-1)$ 个前向、反向过程。

Bubble 占有率比例 bubbletaion，又称空泡率，计算由公式给出：

$$\begin{equation}
\mathit{bubble ration} = \frac{t_{bubble}}{t_{bubble} + t_{ideal}} 
= \frac{(p-1)(t_f + t_b)}{(p-1)(t_f + t_b) + m(t_f + t_b)} 
= \frac{p - 1}{m + p - 1}
\end{equation}$$

其中 $t_{ideal}$ 为理想迭代时间，$m$ 为 micro-batch 的数量，$t_f$，$t_b$ 为单个 $micro-batch$ 时间，因此 $t_{ideal}=m(t_f+t_b)$。

根据上面的公式，$bubble ration$ 跟 $micro-batches$ 有关系，micro-batch 数量（m）越多，Bubble 的比例越会降低到可接受的水平，因此在衡量大模型性能优化的过程，$bubble ration$ 是作为一个衡量指标去看待其利用率。

**3）Gpipe 原理解析：**

Gpipe 是基于上述特点推出的流水线并行技术：将一个 batch size 的数据切分成四个 micro-batch size，每个 micro-batch 作为朴素流水并行方式中的一个 batch，前向过程从 rank0 流向 rank3（又称为 warmup），再反向回溯（称为 cooldown）。

![alt text](Gpipe.png)

计算空泡率：

$$\begin{equation}
bubble ration=\frac{t_{bubble}}{t_{ideal}}=\frac{p-1}{m}
\end{equation}$$

因此为降低空泡率，通常需要增加数据切分 micro-batches 的数量 m，即令 $m>>p$.

在模型的反向传输过程中，由于 GPU 需要保存前向传播时的中间激活值，以便计算梯度，因此划分 micro-batch 的数目 m 将由单 GPU 计算卡的显存约束（eg. 相同色块为一张 GPU,对于 GPU1 来说需要保存 m=4 个前向过程的激活值，因此当使用多个 micro-batch，激活值存储量线性增加），在 warmup 阶段结束后所有 GPU 显存，达到称为动态内存峰值。

**4）动态内存峰值分析：**

为解决 Gpipe 带来的动态内存峰值问题，重计算（Recomputation）技术被引入解决显存瓶颈问题，其核心思想为：与其在前向传播中缓存所有中间激活值，不如在反向传输时“重新计算”一遍前向过程，来获得需要的激活值，从而节省显存。
(待补充)

## 总结与思考

!!!!!!!!!!!!!!

## 参考与引用
