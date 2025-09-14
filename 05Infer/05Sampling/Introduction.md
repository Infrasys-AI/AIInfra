  1. 输出采样基础 1（随机采样、贪婪采样、束搜索采样 beam search）
  2. 输出采样基础 2（Top-k 采样、 Top-P 采样、温度采样、Self-Consistency、联合采样）
  3. 采样加速 1：投机采样（Speculative Decoding、Medusa、EAGLE1、EAGLE2）
  4. 采样加速 2：并行采样（Multi-Token Prediction、Lookahead Decoding）
  5. 采样加速 3：早退 Early exiting 与级联推理 Cascade inference
  6. MOE 推理采样（路由机制、专家选择

你是大模型算法专家，现在要针对于大学生写一份关于输出采样的学习文档。请你针对于大模型推理输出时候的流程，包括随机采样、贪婪采样、束搜索采样 beam search 这几个方向写一份详细，由浅入深的文档帮助学习。


## 大模型输出基础


在自然语言处理（NLP）任务中，生成模型（如 GPT 系列）通常用于文本生成、机器翻译等任务。为了使模型能够生成符合上下文且连贯的文本，我们需要对生成过程中的输出进行采样。采样策略决定了模型每次生成的词或标记，并直接影响生成文本的质量与多样性。本篇文档将介绍三种常见的输出采样方法：随机采样、贪婪采样和束搜索采样（Beam Search），并从简单到复杂逐步深入，帮助大家更好地理解这些采样策略。

---

### 1. 随机采样 (Random Sampling)

#### 原理

随机采样是最直观的一种采样策略。在这种方法中，模型会根据当前的输出概率分布从词汇表中随机选择下一个词。

* **过程**：在生成过程中，模型会计算出每个候选词的概率分布（通常由 softmax 函数输出）。然后，根据这个概率分布随机选择一个词作为下一个输出。
* **优点**：简单直观，适合生成多样化的输出，避免生成重复或单调的文本。
* **缺点**：由于是随机选择，生成的结果可能不太连贯，甚至可能会出现不合理或无意义的文本。

#### 示例

假设模型计算出的概率分布如下：

* “apple”：0.5
* “banana”：0.3
* “orange”：0.2

那么，模型会根据这个概率分布从“apple”，“banana”或“orange”中随机选择一个词作为输出。

``` python

import torch

def random_sampling(prefix_idx: list[int], model, max_len: int) -> list[int]:
    """
    随机采样生成序列。
    
    参数:
    - prefix_idx: 初始序列列表。
    - model: 用来生成下一个 token 概率的模型。
    - max_len: 生成序列的最大长度。
    
    返回值:
    - 完整生成的序列。
    """
    sequence = prefix_idx[:]  # Initialize the sequence with the prefix
    hidden = torch.zeros(1, 1, model.rnn.hidden_size)  # 假设模型需要初始化隐藏状态

    for _ in range(max_len):
        input = torch.tensor([sequence[-1]])
        output, hidden = model(input, hidden)
        probabilities = torch.softmax(output, dim=0)
        
        # 从概率分布中随机采样一个 token
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        sequence.append(next_token)
        
        # 检查是否生成了结束标识符（如果有的话，可以用来提前结束生成）
        # if next_token == end_token:
        #     break

    return sequence

```
---

### 2. 贪婪采样 (Greedy Sampling)

#### 原理

贪婪采样是一种贪心策略，目的是每一步选择最有可能的词。与随机采样不同，贪婪采样并不会从概率分布中随机选择，而是直接选择概率最高的词。

* **过程**：在每一步生成过程中，模型计算出当前候选词的概率分布，然后选择概率最大的词作为输出。
* **优点**：保证了每一步都生成最可能的词，因此生成的文本通常较为连贯且符合语法规则。
* **缺点**：这种方法往往会导致生成的文本过于单一，缺乏多样性，容易产生“模板化”的内容，且可能卡在局部最优解。

#### 示例

假设模型计算出的概率分布如下：

* “apple”：0.5
* “banana”：0.3
* “orange”：0.2

在贪婪采样中，模型会选择“apple”作为下一个输出。

``` python
import torch

def greedy_sampling(prefix_idx: list[int], model, max_len: int) -> list[int]:
    """
    贪婪采样生成序列。

    参数:
    - prefix_idx: 初始序列列表。
    - model: 用来生成下一个 token 概率的模型。
    - max_len: 生成序列的最大长度。

    返回值:
    - 完整生成的序列。
    """
    sequence = prefix_idx[:]  # Initialize the sequence with the prefix
    hidden = torch.zeros(1, 1, model.rnn.hidden_size)  # 假设模型需要初始化隐藏状态

    for _ in range(max_len):
        input = torch.tensor([sequence[-1]])
        output, hidden = model(input, hidden)
        probabilities = torch.softmax(output, dim=0)
        
        # 选择概率最大的 token
        next_token = torch.argmax(probabilities).item()
        sequence.append(next_token)
        
        # 检查是否生成了结束标识符（如果有的话，可以用来提前结束生成）
        # if next_token == end_token:
        #     break

    return sequence


``` 
---

### 3. 束搜索采样 (Beam Search)

#### 原理

束搜索采样是一种在每一步生成时探索多个候选词的策略，目的是通过保留多个最有可能的词序列，从而获得更优的输出。

* **过程**：

  1. **初始化**：从初始词开始，保留多个候选序列（束宽度为 Beam Width），这些序列是根据每个词的概率排序得到的。
  2. **扩展**：每次生成一个词后，模型会根据当前候选序列的概率分布选择多个最有可能的词，并将这些词扩展到已有的序列上。
  3. **剪枝**：每次扩展后，只保留概率最高的`Beam Width`个序列，丢弃其他低概率序列。
  4. **重复**：该过程持续进行，直到生成结束符或者达到最大生成长度。

* **优点**：束搜索相比贪婪采样可以避免局部最优解的困境，通过探索多个可能的输出序列，能生成更具多样性且较为合理的文本。

* **缺点**：计算成本较高，尤其当束宽度设置较大时，搜索空间会显著增加，导致计算效率下降。

#### 示例

假设我们设置束宽度为`Beam Width = 2`，并且模型的初始概率分布如下：

* 第一步选择：

  * “apple”：0.5
  * “banana”：0.3
  * “orange”：0.2

  选择概率最高的两个序列：

  * “apple” (概率：0.5)
  * “banana” (概率：0.3)

* 第二步生成词汇：
  对于“apple”扩展的候选词：

  * “pie”：0.4
  * “juice”：0.6

  对于“banana”扩展的候选词：

  * “pie”：0.5
  * “smoothie”：0.5

  选择概率最高的两个序列：

  * “apple juice” (概率：0.5 \* 0.6 = 0.3)
  * “banana smoothie” (概率：0.3 \* 0.5 = 0.15)

如此进行下去，最终生成一个完整的序列。

``` python

def beam_search(prefix_idx: list[int], model, k: int, max_len: int) -> list[tuple[list[int], float]]:
    beams = [(prefix_idx, 0.0)]  # 初始化 beam，包含一个元素: (当前序列, 对数概率得分)
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            logits = model(seq)[-1]  # 传入序列并获取模型的输出 logits
            log_probs = torch.log_softmax(logits, dim=-1)  # 计算 log softmax 得到每个 token 的对数概率
            top_k_log_probs, top_k_tokens = torch.topk(log_probs, k)  # 选取概率最大的 k 个 token

            # 扩展当前的序列为新的 beam
            for log_prob, token in zip(top_k_log_probs, top_k_tokens):
                new_seq = seq + [token.item()]
                new_score = score + log_prob.item()
                new_beams.append((new_seq, new_score))

        # 从扩展后的序列中选择得分最高的 k 个序列作为新的 beam
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:k]

    return beams
```
---

### 4. 对比与应用场景

| 采样方法  | 生成方式          | 优点           | 缺点             | 适用场景          |
| ----- | ------------- | ------------ | -------------- | ------------- |
| 随机采样  | 随机选择概率分布中的词   | 多样性强，生成新颖文本  | 可能生成不连贯、无意义的文本 | 文本生成、创意写作     |
| 贪婪采样  | 选择概率最高的词      | 生成文本连贯，符合语法  | 生成单一且重复的文本     | 对准确性要求较高的任务   |
| 束搜索采样 | 扩展多个候选序列，保留最优 | 可以生成连贯且多样的文本 | 计算开销大，速度较慢     | 翻译、摘要、对话生成等任务 |

---
