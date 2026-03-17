---
title: "My Knowledge Map"
date: 2025-11-24T16:00:00-07:00
lastmod: 2026-03-15T00:13:47-07:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
tags:
description: My personal knowledge map.
summary: My personal knowledge map.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Tools

- Teamwork: git & github + CI/CD
- Environment:
  - Docker & Singularity
  - Slurm
  - Linux (Debian & Ubuntu & RockyLinux & Fedora)
  - Windows & WSL2 (pin memory)
  - CMake + vcpkg
  - pip/conda/uv
- IDE: VSCode (Linux/Windows) & VS (Windows)
- Others: Bash & Powershell & Vim & Tmux



## 2. Languages & Frameworks

- Languages: C & C++ & CUDA & Python & pytorch & libtorch & Triton
- Inference: vLLM & sglang
- SFT: trl & unsloth
- RLHF: verl
- Compute Graph: MLIR

## 3. 基础 DL

### 3.1. MAE (L1) & MSE (L2) Loss

MAE, i.e., Mean Absolute Error:

$$
\text{MAE} = \frac{1}{n} \sum^{n}_{i=1}|\hat{y} - y|
$$


MSE, i.e. Mean Squared Error:

$$
\text{MAE} = \frac{1}{n} \sum^{n}_{i=1}(\hat{y} - y)
$$

**MAE & MSE loss 通常被用于回归任务**; 即单纯让模型的 output Tensor 中每个元素都数值上靠近 label Tensor, 不存在 "将 output 看成不同类别的概率" 这个操作.

> 比如有个模型的 output 是 [10, 4, 5] shape 的 tensor.
> - **在分类任务中**, 我们可能会把 10 作为 batch size (B), 4 作为 seq len (L), 5 作为 vocab size (V).  
>   接着我们会对 V 维度做 softmax: `outputs = torch.softmax(outputs, dim=-1)`, 这样每个 token 对应的 V 维度的 5 个值自然地被 map 到 (0, 1) 这个区间, 代表这个 token 是 0 号单词的概率, 是 1 号单词的 概率, ... 是 5 号单词的概率.  
>   **这其实就是对 (10 * 4) 个 token 的分类问题, 一共有 5 类.**
> - 但是**在回归问题中**, output 和 label 的 shape 一般相同 (**每个维度都有意义, 每个元素都应该靠近 label**).  
>   例如, 场景是: batch size = 10, 预测一张 (4 * 5) 的图像内每个像素点的灰度值是多少.   
>   这种情况下就应该直接将 `MAE(pred, label)` 或者 `MSE(pred, label)` 作为 loss 来优化.

1. 惩罚强度不同。  
   MAE 对误差是线性惩罚，错 10 ⽐错 1 ⼤ 10 倍。  
   MSE 对误差是平⽅惩罚，错 10 ⽐错 1 ⼤ 100 倍。  
   所以 MSE 会更强烈地惩罚⼤误差。(即在优化时让模型更趋近 ground truth)
2. 对离群点的敏感性不同。  
   MAE 对 outlier 更稳健。
   MSE 很容易被少数⼏个特别⼤的误差主导。   
   所以数据⾥如果有脏点、极端值，MAE 往往更稳。  
3. 优化性质不同。  
   MSE 更平滑，可导性更好，优化通常更稳定，尤其在深度学习⾥很常⻅。  
   MAE 在 $e=0$ 处不可导，⽽且梯度⼤⼩基本是常数，不像 MSE 会随误差变⼤⽽增⼤，所以优化时有时没那么“顺⼿”。  
4. 统计含义不同。  
   最⼩化 MSE 往往对应去拟合条件均值。   
   最⼩化 MAE 往往对应去拟合条件中位数。  
   所以如果你的⽬标更像“平均值预测”，MSE 常⻅；如果你更想抗异常值，MAE 常⻅. 

⼀句话总结：

- L1 loss 适合你怀疑数据⾥有噪声点、离群点，或者你更想优化“中位数式”的稳健表现。
- L2 loss 适合误差近似⾼斯、你想让模型更重视⼤偏差，或者需要平滑、好优化的⽬标。

Code:

{{<details title="Click to Expand">}}

```python
import torch

# MAE, L1
def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  # pred: [...], target: [...]
  return torch.mean(torch.abs(pred - target))  # Scalar

# MSE, L2
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  # pred: [...], target: [...]
  return torch.mean((pred - target) ** 2)  # Scalar
```

Which is the same as:

```python
from torch import nn

# Callable, whose forward function takes #1: pred, #2: y
mae_loss = nn.L1Loss()  
# Callable, whose forward function takes #1: pred, #2: y
mse_loss = nn.MSELoss()
```
{{</details>}}

### 3.2. Cross Entropy Loss and Binary Cross Entropy Loss

CE Loss 一般被用于分类任务; BCE Loss 一般被用于二分类任务.

#### 3.2.1. Target Distribution Cross Entropy

假定我们要做一个分 `NC` 类的分类任务; 一般模型的 output logits 的 shape 为: `[B, NC]`, 其中 `B` 为 batch size.

> 在 LLM 中, logits 一般理解为 `[B, L, NC]`, 这只是 layout 的区别罢了, 因为本质上我们就是要为 (B * L) 的 token 分类.

在 Target Distribution Cross Entropy 中, label 的 shape 则为 `[B, NC]`: 即对于每个 sample, label 给出这个 sample 是第 $i (i \in [1, \text{NC}])$ 类的概率 (是一个 prob distribution).

> 这可能和你认为的经典分类任务不太一样: "每个 sample 不应该就是有一个确定的类别吗?" ------ Wait, wait, wait! 谁说一个 sample 只可能有一个类别了? 比如我说 "Tom 的性别 label 是 [0.5, 0.2, 0.3], 即 50% 像男人, 20% 像女人, 30% 不知道" 呢? 此时我的模型是不是就该输出一个 `[..., 3]` 的 Tensor, 并且尽可能向 [0.5, 0.2, 0.3] 靠近?

为计算 CE, 我们的第一步是对 logits 的 `NC` 维度 apply softmax, 进而转化为 probs: `probs = torch.softmax(logits, dim=-1)`. 此时对于 `probs: [B, NC]` 中的每个 sample, 其 `NC` 个数字就代表了该 sample 是第 $i (i \in [1, \text{NC}])$ 类的概率.

> **注意: softmax 不仅能把 `NC` 个数统一映射到 (0, 1), 还能保证他们的和为 1**.

接着, 对于单个 sample, CE loss 的计算公式如下:

$$
L_\text{CE} = - \sum_{i=1}^{\text{NC}}(p_\text{label}^i \times \log{p_\text{pred}^i})
$$

计算结果是一个标量; 而对于 `B` 个 sample, 最终的 CE Loss 就是将所有 sample 的 Loss 求和取平均.

> 看到这里, 我们会发现, CE Loss 与 L1/L2 Loss 都在 "将 output tensor 中的所有元素向 label 中的所有元素优化".  
> 但是他们有个很重要的不同: CE Loss 将最后一个维度 (`NC`) 对待为 "有 `NC` 种类别, 它们分别是什么概率", 因此 apply 了 softmax; 而 L1/L2 loss 无论有多少维, 都会平等对待所有元素, 统一向 label 优化.


Code:

```python
import torch

def standard_ce_loss(
  logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    # logits: [B, NC]; labels: [B, NC]
    log_probs = torch.log_softmax(logitis, dim=-1)  # [B, NC]
    each_sample_loss = -torch.sum(labels * log_probs, dim=-1) # [B]
    return torch.mean(each_sample_loss) # Scalar

```

#### 3.2.1. 明确要分第几类任务下的 CE Loss (NLL Loss)

模型输出 logits 还是 `[B, NC]`, 计算 probs: `probs = torch.softmax(logits, dim=-1)`.

现在, label 不再是 `[B, NC]` 的 "概率矩阵" 了, 而是 `[B]` 的 "第几类" 矩阵 ------ 每个元素的意义就是对应 sample 是第几类.

那么, 对于一个 sample, 我们是不是可以理解为, **是这个类型的 ground truth prob 就是 1, 不是这个类型的 ground truth prob 就是 0**?

此时计算该 sample 的标准 CE Loss 就可以简化为:

$$
L_\text{CE} = - \sum_{i=1}^{\text{NC}}(\text{label}^i \times \log{p_\text{pred}^i})
$$

这里的 $\text{label}$ 是一个 0/1 向量, 只有 "确实是这个类型" 的位置是 1, 其余都是 0.


### 3.2. L1 & L2 正则化



## 4. LLM Tech List


{{<details title="Click to Expand">}}

- Inference:
  - TP (Megatron)
  - {{<text url="https://gemini.google.com/share/bad0ffe21878">}}Quantization{{</text>}}: PTQ & QAT & GPTQ
  - Pruning: Unstructured & Structured
  - Paged Attention & Flash Attention & MQA & GQA & {{<text url="https://zhuanlan.zhihu.com/p/16730036197">}}DeepSeek Sparse Attention (MLA){{</text>}}
  - Prefix Caching
  - Continuous Batching
  - Chunked Prefill
  - Speculative Decoding
  - Sampling: Top-k & Top-p & Temperature & Beam Search
- Training:
  - Pretraining
  - SFT
  - RLHF: PPO & GRPO & DAPO
  - {{<text url="https://zhuanlan.zhihu.com/p/621700272">}}PEFT{{</text>}}: {{<text url="https://zhuanlan.zhihu.com/p/702629428">}}LoRA{{</text>}}, Prefix Tuning, P-Tuning, Prompt Tuning
  - Efficiency:
    - Mixed Precision: FP16 & BF16 & TF32 & INT8
    - {{<text url="https://zhuanlan.zhihu.com/p/596977579">}}Gradient Checkpointing{{</text>}}
    - ZeRO Optimizer: {{<text url="https://zhuanlan.zhihu.com/p/694880795">}}Stage 1, 2, 3{{</text>}}, {{<text url="https://zhuanlan.zhihu.com/p/513571706">}}ZeRO Offloading{{</text>}}
    - DP (DDP vs FSDP), MP, PP
  - Position Embedding:
    - {{<text url="https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003">}}ROPE{{</text>}}
- Cluster:
  - Apptainer + Slurm + Docker + Module
  - torchrun & deepspeed & accelerate & bitsandbytes
  - NCCL & Gloo & MPI
  - Ray
  - InfiniBand
  - Evaluation: HPCG
  
{{</details>}}


## 5. LLM 基础模型结构

{{<image
src="/imgs/blogs/my-knowledge-map/model-structure.drawio.svg"
width="100%"
caption=`Decoder-Only 模型的经典结构`
>}}

上图展示了 Decoder-Only Model 的经典结构; 省略了每层 Decoder Layer 的 Residual 结构和 Attention 内部的 Causal Mask.

另外要注意, 最后输出的 logitis: (B, L, V) 在 sampling 阶段需要对 vocab 维度做 softmax 进而转化为 probs 或者 logprobs:

```python
import torch.nn.functional as F

probs = F.softmax(logitis, dim=-1)  # (B, L, V)
log_probs = F.log_softmax(logitis, dim=-1)  # (B, L, V)
```

### 5.1. Embedding

Embedding 是模型内部的一个层, 和模型一起训练, 能够将一个 token id (int scalar) map 到一个 tensor (of shape `(H,)`).

> - 一个 word (str) 对应一个 token id (int scalar).
> - 一个 token id 被 map 到一个 tensor of shape `(H,)`.
> - 一句话 -> list[int] -> a tensor of shape `(L, H)`, where `L` is the number of token ids.
> - 一个 request -> 一句话
> - 多个 request -> 组 batch, 把短的用 pad-token-id 补齐到最长的 -> a tensor of shape `(B, L, H)`. 

### 5.2. LayerNorm vs RMSNorm vs BatchNorm

对于输入 Tensor `[B,L,H]`, LayerNorm 和 RMSNorm 都在 `H` 维度做 token-wise normalization (也就是对每个 token 的 hidden state 独立 normalize).

LayerNorm:

\[
\mu = \frac{1}{d}\sum_i x_i,\qquad
\sigma^2 = \frac{1}{d}\sum_i (x_i-\mu)^2
\]

\[
\mathrm{LN}(x)_i = \gamma_i \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i
\]

RMSNorm:

\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}
\]

\[
\mathrm{RMSNorm}(x)_i = \gamma_i \frac{x_i}{\mathrm{RMS}(x)}
\]

> - LayerNorm：关心均值和方差
> - RMSNorm：只关心整体幅值

从计算角度, LN 需要： 1. 求 mean; 2. 求 variance; 3. 做中心化. RMSNorm 只需要：1. 求平方均值; 2. 开根号; 3. 做缩放. 大语言模型有很多层，每一层都有 norm。即使 RMSNorm 每次只省一点点，叠加到几十上百层、长序列和大 batch 上，就会变成很明显的收益。所以从工程角度看，RMSNorm 是一种更划算的选择。

**为什么不用 BN?**

BN 需要在一个 batch 上统计均值和方差。但 NLP / LLM 中：

- 序列长度不同
- padding 很多
- token 分布差异很大
- batch 内样本语义高度异质

这会让 batch 统计变得噪声很大，也不稳定。

此外, 在推理时, batch 大小相比训练会小很多 -> 推理训练 batch normalize 数量级不同. 因此不适合 BN.

### 5.3. Rotary Position Embedding (RoPE)

为什么要 Position Embedding? 因为 Attention 机制对 token 的 position 不敏感; 需要注入 position 信息.

1. **Learned Absolute Position Embedding**  
   最经典的一类。给第 1、2、3... 个位置各配一个可学习向量，然后与 token embedding 相加。BERT 一类早期模型常见。优点是简单；缺点是**对训练长度之外的 extrapolation 较差**，而且位置是“表驱动”的，泛化到更长上下文通常不理想
2. **Sinusoidal Positional Encoding**  
   Transformer 原始论文里的做法。每个位置用不同频率的正弦/余弦函数编码。优点是**无额外可学习参数**，并且理论上可以推广到更长长度；但它是把位置加到 embedding 上，和 attention 机制本身耦合得没有那么直接.
3. **Relative Position Encoding**  
   不直接编码“第几个位置”，而更强调“两个 token 相距多远”。典型代表有 Shaw 等人的相对位置偏置、Transformer-XL 等。这类方法通常更贴近语言建模需求，因为很多语义关系更依赖相对距离而不是绝对索引。
4. **ALiBi (Attention with Linear Biases)**  
   它不是显式给 embedding 加位置向量，而是在 attention score 上加一个和距离相关的线性 bias。优点是实现简单、长上下文外推通常不错。很多长上下文讨论里会把它拿来和 RoPE 对比。
5. **RoPE (Rotary Position Embedding)**  
   它把位置信息编码进 Q/K 的旋转中。相比绝对位置相加，RoPE 更“贴着 attention 在工作”，因此在建模相对位置关系时非常自然。RoFormer 论文明确指出，RoPE 同时编码绝对位置，并在 self-attention 里显式体现相对位置依赖。

为什么现在很多 LLM 都用 **RoPE**？

1. **它对相对位置建模很自然**。语言任务里，很多关系并不取决于“这是第 137 个 token”，而取决于“它离前面的主语/谓语有多远”。RoPE 通过旋转后的内积，让 attention 更容易反映相对位置信息.
2. **它对长上下文更友好**。相比 learned absolute embedding，RoPE 在长度外推上通常更有优势，因此非常适合现代 LLM 追求更长 context window 的趋势。

### 5.4. Attention

$$
O=Attention(Q,K,V)=softmax\left(\text{Mask}(\frac{QK^T}{\sqrt{d_k}})\right)V
$$

1. **Causal Mask**: 对于单句输入 `I: (L, H)` (忽略 Batch 维度), 经过 attention 最终得到的输出矩阵 `O: (L, H)`; 我们在训练中期待 `O[i, :]` 这个 token 是 `I[:i, :]` 对应的 next token; 因此无论在推理还是训练时, 都需要一个下三角矩阵 mask 掉 `QK: (L, H)` 的 "未来" 部分: 比如对于 `QK` 的第一行, 只有第一个值有效, 这样 `O[0, :]` 就表示了给模型输入的第一个 token, 它预测出的是什么 token.
2. **Attention Mask**: 标注哪些部分用来计算 attention, 比如 pad token 在 attention mask 中就应该标记为 0.
3. **为什么除以 $\sqrt{d}$**: 因为点积 \(q \cdot k\) 的方差会随着维度 \(d\) 增大而变大，不除以 \(\sqrt{d}\) 会让 softmax 输入过大、容易饱和，导致梯度变小、训练不稳定，所以要用 \(\sqrt{d}\) 把尺度归一化。
4. **MHA** 是每个 query head 都有自己独立的 K/V head，**GQA** 是多个 query head 共享一组 K/V head，**MQA** 则是所有 query head 共用同一组 K/V head，所以它们本质上是在“KV 共享程度”和“效果与推理效率的权衡”上逐级加强。

### 5.5. Sampling

LMHead 的输出是 `(B, L, V)`, `V` 是 vocab size, 后续要通过 sampling 过程选出具体是哪个 token:

- top-p sampling
- top-k sampling
- temperature sampling: `prob = prob / temp`

**一般 temp 设置为 0.6-0.8 模型的效果最好**, 因为此时模型有更多的采样可能; 每次都只采纳最高 prob 的 token 会使效果变差.

- temp=1: 不改变模型 prob 分布.
- temp 越小: prob 分布越尖锐, 模型采样多样性越小.
- temp 越大: prob 分布越平滑, 多样性越大.
- temp=0: prob 没法除 0, 一般推理引擎把这种情况直接处理为 greedy sampling (只选 prob 最高的 token).

## 6. 推理引擎和典型加速策略
### 6.1. Model Parallel & Tensor Parallel & Pipeline Parallel

这三个并行策略主要争对 "大模型" (单卡放不下) 推理场景.

- MP: 一个 GPU 放不下一整个模型, 因此把模型**按顺序**切分放到不同 GPU 上**串行**执行; MP 是 transforers 库默认的并行策略; 缺点是串行执行不同 GPU 需要等待 (当前 GPU 需要等上一个 GPU 的数据传过来才能计算), 速度慢.
- TP: 把模型的一层 (可以认为是若干个矩阵计算操作, 例如一层是一个 Transformer Layer) 切分到不同 GPU **并行**执行 (例如, $A \times B$ 可以切分成 $A_1 \times B$ 和 $A_2 \times B$ 并行算); 速度快并且能降低单卡的显存占用, 但需要手动修改矩阵计算的代码; 目前工业界已经对 TP 做到了很大的支持, 推理引擎 (例如 vllm, sglang) 都已经默认支持 TP; 但是由于 TP 需要模型使用 "支持 TP 计算" 的模型层定义, 因此可以看到推理引擎内都会额外自己定义一遍模型.
- PP: 可以理解为高级的 MP, 把模型的不同部分切分到不同 GPU, 而相比 MP 进一步做流水线优化 --- 有些部分的计算不依赖于另一些部分的计算结果, 所以可以做到流水线并行; 目前对 PP 的支持都需要模型的开发者自己手动在推理引擎内部打补丁, 所以普及率远不如 TP.

### 6.2. 常用推理引擎

- vllm: 老牌开源引擎, 提出 paged attention (需要详细了解)
- sglang: 社区更加活跃, 提出 radix attention (了解即可)
- tensorrtLLM: NV 亲自支持, 纯 C++ 实现

### 6.3. 推理加速核心技术

1. **Prefill and Decode**: Decode 能用之前的 KV Cache, 具体的自己搜一下吧.
2. **PagedAttention**: 解决大量 request 不停申请 KV Cache 时, 显存上的 Memory Fragmentation 问题.
3. **组 batch**: 推理时所有 input 一定要组 batch, 通过对一个 batch 的 inputs 中较短 input_ids 填充 pad token id 进而组成 `(B, L)` 的 input batch, 这里 `B` 就是表示有 B 个 request (一个 request 就是一个 input_ids).
4. **Prefill-Decode Aggregation/Disaggregation**: 是否 Prefill 和 Decode 一起算. 什么叫一起算: 当前 batch 同时包含 prefill request 和 decode request. 一起算的好处是能同时处理很多不同类型的 request, 在高并发的场景吞吐率高; 不一起算的好处是计算流程清晰, 计算效率更高.
5. **Continuous Batching**: 批量 request 时的 decoding 阶段持续把新到请求动态插入同一批次 (假设设定死了 `max_batch_size=10`, 一个用户提交了 10 个 requests, 还没跑完时另一个用户又提交了 10 个 requests; 第一个用户提交的 10 个 requests 中有 8 个很快跑完了, 剩下 2 个一直跑不完; 这时候利用 Continuous Batching 技术能将第二个用户的 8 个 requets 和第一个用户的剩下 2 个 requets 组成一个新 batch 跑).
6. **Chunked Prefill**: 把长 prompt 的 prefill token ids 切开、让计算与其他请求交错执行以减少首 token 等待和显存峰值 (假设定死了 GPU 一次只能处理 `100` 个 token, 如果有个 prefill input 占 `90` 个 token, 那当前 batch 其他 request 没法跑了, 所以把 `100` 个 token 切开一点点算, 不要一次全算).
7. **Prefix Caching**: 如果多个 request 共享 prefill token ids, 比如问十个一样的问题, 就没必要重复计算这十个问题的 KV Cache 了 --- 我们为每个 prefill request 计算 hash value, 如果匹配上就利用之前计算过的 kv cache.

## 7. 训练

### 7.1. 训练的几个阶段

本节会大致讲一下几个阶段的特征, 这些特征都需要熟记.

**Pre-training**:
- 是资源消耗最大的阶段, 一般都要使用万卡集群. 所以学生/高校自己一般不会做预训练.
- 目的是让模型从 0 学会语义理解和说话.
- Tokenizer 中的 vocab mapping 是预训练前就决定好的, 但是模型内部的 embedding layer 是跟随预训练一起训的.
- 预训练完的模型叫 Base Model, 比如你在 Huggingface 上能找到的 Qwen/Qwen3-8B-Base 等. Base Model 具备说话和理解能力, 但是不知道何时听 (不停说), 而且输出的文本没有规范的格式, 不符合人类对话的偏好.
- Base Model 不支持 special tokens (比如 `<|user|>`, `<|assistant|>`, 一般在 {{<text url="https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json">}}tokenizer_config.json{{</text>}} 文件里可以找到), 因此无法跑 chat completion task (chat completion task 就是用 `tokenizer.apply_chat_template` 方法把输入变成对话形式的 text).
- [Interesting Fact] Qwen/Qwen2.5-7B 本身就是 Base Model; 而 Qwen/Qwen3-8B-Base 才是 Base Model, Qwen/Qwen3-8B 是 SFT+RL 过的 reasoning model! 自己跑实验的时候一定要查清楚这个模型是 Base 还是 Instruct Model.

**Post-Training**:
- 后训练的资源消耗一般是可接受的, 所以很多科研现在都在研究后训练.
- 后训练大体可以分为 SFT (Supervised Fine-Tuning) 和 RL (Reinforcement Learning) 两类, 主要区别是:
  - SFT 会提供完整的 Question 和 Grount Truth Response, 模型每个 output token 都需要和 GT Response 的 token 算 Cross Entropy Loss. 形式化的说就是模型输出必须强制和答案对齐.
  - 官方 SFT 训出来的模型一般叫 instruct model, 比如 Qwen/Qwen2.5-7b-Instruct; Qwen3 没有提供 Instruct 版本, 只提供了 Base 版本和 Reasoning 版本 (SFT+RL).
  - SFT 完的模型, 根据你提供的 SFT 数据集, "能够" 具备对话能力 (chat completion), 知道何时合适地结束输出, 具有初步 reasoning 能力, 或者拥有专业领域的特殊知识. 你可以用不同的数据集多次 SFT 模型让他一点点有更多能力.
  - SFT 分全量微调和 LoRA 微调, LoRA 下文会讲.
  - RL 只会提供 Qusetion 和一个 "打分方式", 训练的目标一般有两个, 1 是得到更高分, 2 是不要和上一轮的自己偏离太远.
  - RL 完的模型, 由于能得到 "更高分", 会更加符合人类偏好, 或者具有更强的 reasoning 能力. 训练什么能力取决于你对什么目标打分. Qwen/Qwen3-8B 就是 RL 过的模型.
  

### 7.2. Data Parallel & Distributed Data Parallel & Fully Sharded Data Parallel

这三个策略主要用于训练加速.

- DP: 每个 GPU 都有一份完整的模型副本, 每个 GPU 处理不同的数据 batch, 每个 GPU 计算完 loss 和梯度后, 需要把梯度同步到其他 GPU 上; DP 的优点是实现简单, 缺点是每个 GPU 都需要存储完整的模型副本, 显存占用大.
- DDP: 是 DP 的一种优化实现, 通过使用 NCCL 库进行高效的梯度同步 (RingAllReduce), 可以显著提升 DP 的效率; DDP 是 PyTorch 官方推荐的分布式训练方式. 使用 DDP 的方式是从 transformers 库中 import DDP 模块, 直接包裹模型; 另外 dataloader 需要一些额外配置 (例如 DistributedSampler) 来保证每个 GPU 处理不同的数据; 启动 DDP 我们一般会用 `torchrun` 命令来 launch 训练脚本, 如果你有 4 个 GPU, 在 launch 时就指定启动 4 个进程 (进程对应一个 GPU); DDP 的缺点是每个 GPU 仍然需要存储完整的模型副本, 显存占用大.
- FSDP: 是 DDP 的进一步优化, 通过把模型参数切分到不同 GPU 上 (Sharding), 可以显著降低每个 GPU 的显存占用; FSDP 的使用方式和 DDP 类似, 也是从 transformers 库中 import FSDP 模块, 直接包裹模型. FSDP 和 ZeRO 的三个 Stage 有千丝万缕的关系, 可以认为 FSDP 是 pytorch 官方实现的 ZeRO (详见下文 ZeRO).

### 7.3. ZeRO

Zero 优化技术看这篇: {{<text url="https://zhuanlan.zhihu.com/p/694880795">}}Zero Stage 1, 2, 3{{</text>}} (至少了解训练中有哪些显存占用, 每个 stage 优化的点是什么).

模型训练, 以 7B 模型为例:

(7B * fp16) Parameters + (7B * fp16) Gradients + (7B * fp32) * 3 Optimizer States + Activation (relevant to batch size and seq len)

- Zero 1: 切分 Optimizer State
- Zero 2: 切分 Gradients
- Zero 3: 切分 Parameters

### 7.4. Gradient Checkpoint

前向传播时不保存某些层的中间激活，只保留少量“检查点”；反向传播走到这段时，再用检查点重新做一遍前向，把缺失的激活现算出来，再继续求梯度。

更具体一点，通常是这样实现的：

1. 把模型切成若干段: 例如一个很深的 Transformer，有 24 层，你可以每 4 层或每 6 层作为一个 segment。

2. 前向时只保存 segment 边界的张量: 正常训练里，每层输出激活都会留着给 backward 用。checkpoint 后，只保存每段输入/输出，段内各层的中间激活不保留。

3. 反向时按需重算该段前向: 当 backward 需要某一段内部某层的激活时，框架会拿这段的检查点输入，再临时跑一次这段 forward，恢复出中间激活，然后立刻计算梯度。

用“额外计算”换“更低显存”; 所以它本质上是：**显存下降，训练时间上升**。

常见情况是显存省很多，但计算开销增加大约 20% 到 40%，具体看切分方式。

### 7.5. SFT

SFT 我一般用 trl 或者 unsloth 进行训练. trl 是 huggingface 的官方库, unsloth 是 trl 的优化版, 自己写了一些 kernel, 并且自动做 memory offload.

LoRA 的具体做法是创建两个矩阵 (LoRA Adapter): `B: (H, r)` `A: (r, H)`, (通常) 对 LLM 的每层 decoder 内的 `q_proj` 和 `v_proj` 做 LoRA:

原来: $Q = W_qX$, $V = W_vX$ ---> 现在: $Q = W_qX + BAX$, $V = W_vX + BAX$

一般把 B 初始化为 0, A 随机初始化, 这样能保证开始训练前模型输出和原始一致.

### 7.6. DPO

### 7.7. PPO

### 7.8. GRPO

### 7.9. DAPO

1. Clip Higher: GRPO -> $[1-\epsilon, 1+\epsilon]$, 上下界一样模型会收敛到一个策略就不再探索; 增加上界 (0.2 -> 0.28) 让模型往好的方向更自由探索.
2. Token Level Loss: GRPO 的 adv 只在组内归一化, 长短序列 token 贡献被稀释; DAPO batch 归一化到 token.
3. Overlong Filtering: 超长没结束的直接 mask 掉.
4. Dynamic Sampling: 一整个 group 都错都对直接扔了.


## 8. HPC 相关

### 8.1. 一些基础名词


如果说 conda 是打包 python 环境, docker 和 singularity 就是打包系统环境, 包括你装的各种软件和库.

因此在 AI 领域, 为了让别人更容易复现自己的实验数据, 经常会把自己的实验环境打包成 docker 镜像给别人运行.

- Docker Image: 可以理解为系统环境的安装包, 不能实际运行, 需要用 `docker run` 命令创建对应容器才能运行.
- Docker Container: 运行了一个镜像的容器实例 --- 可以理解为在你的操作系统里运行了一个 "新的子操作系统", 你可以把它当成独立的服务器对待 (因此甚至能通过 ssh 登录你的 container!). Container 需要先用 `docker start` 启动, 再用 `docker attach` 连接到其终端.
- Singularity: 可以理解为一种更加安全的 docker 服务. Docker 不安全的原因是创建完容器, 在 container 内部, 用户是 root; 这会导致用户能够修改原本在外部无权修改的文件. Singularity 保证用户进入容器后的用户和外部各种权限保持一致.
- Slurm: 这是一种任务提交系统. 用户可以通过 `slurm run` 申请若干 GPU, 若干 CPU, 指定运行时间, 指定 RAM 大小...... 当任务超时, 资源会被释放, 进程自动被 kill.

### 8.2. 如何在本地电脑跑 DL 代码

Windows 系统:
- 首先要安装 Nvidia 的驱动才能识别显卡. 驱动的作用是让 CPU 能够识别并且与外围设备通信.
  - 安装了驱动后, 用 `nvidia-smi` 命令 check 当前显卡状态.
- 在 WSL2 上安装 Ubuntu. WSL 是 windows subsystem for linux, 由微软官方研发的 windows 下 linux 子系统. 
  - WSL1 的做法是由 Windows 系统作 Host, 上面跑一个不完全的 Linux 虚拟机; 这样做的坏处是 Linux 系统运行的非常慢. 
  - WSL2 将 Windows 和 Linux 直接抽象为两个平等的虚拟机, 并且重写了 Linux 内核, 让 Linux 系统的运行速度提升到近乎和原生系统类似; 虽然微软自己写了文件管理器的映射和显卡驱动的映射 (这对深度学习开发者来说极为友好, 否则 1: 你需要自己用 `scp` 命令在 windows 和 linux 之间传文件, 2: 你需要在 linux 系统重新安装显卡驱动), 但是对外围设备的映射还是困难重重 (比如你自己的电脑上插了一个 u 盘, wsl2 里几乎没办法读到这个 u 盘的内容).
- 在 Unbuntu 内安装 docker, 启用 docker 服务, 然后 `docker pull` 一些 image. 
  - 如果你复现别人的代码, 别人提供了自己 build 好的包含代码运行环境的 image, 你就 pull 就完事了. 
  - 如果你想从零搭建一个环境, 一般我们会从 {{<text url="https://hub.docker.com/">}}Docker Hub{{</text>}} 上找 Nvidia 官方的 {{<text url="https://hub.docker.com/r/nvidia/cuda">}}cuda repo{{</text>}}, 在 `Tags` 选项里找需要的 image. 比如我想用一个 ubuntu24 的 并且预装了 cuda13 的 image, 那我就会选择 `docker pull nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`.
  - 需要注意的是, 作为深度学习开发者, 我们一般不会 pull `runtime` 镜像, 只会 pull `devel` 镜像, 因为 runtime 镜像里面只包含了运行必要的 library, 无法利用 cuda 提供的接口进行二次开发 (比如你想用 cuda 的 lib 自己写新的算子, 就叫二次开发).
- 有了 image, 用 `docker run` 创建一个容器. 这个容器就是实际运行的 "虚拟系统".
- 用 `docker start` 启动创建的容器.
- 本机打开 vscode -> attach to wsl -> attach to running container.

### 8.3. 如何在远程服务器跑 DL 代码

一定是用 ssh 连接到远程的 linux 系统:

- ssh 连接的方式是修改当前系统下 `~/.ssh/config`, 把远程系统的信息写进去.
- 如果要免密连接:
  1. 在本机, 利用 `ssh-keygen` 命令创建 private key + public key. (其实在哪里创建无所谓, 只要保证本机有 private key 就行).
  2. 在远程系统, 将 public key 中的内容复制到 `~/.ssh/authorized_keys` 这个文件中的新一行. (只有在这个文件里的 public key 对应 private key 可以免密登录).
  3. 在本机, 在 `~/.ssh/config` 的对应 Host 配置内将 `IdentityFile` 指定为本地的 private key 路径.

我这里举一个 ssh key 配置的例子:

```
Host REMOTE_NICK_NAME
  Hostname remote.ip.com
  User jamesnulliu
  IdentityFile ~/.ssh/my-private-key
```

有了这个配置, 当我在本机运行 `ssh REMOTE_MICK_NAME` 时, 就能用 `~/.ssh/my-private-key`, 以 `jamesnulliu` 这一用户身份, 登录 `remote.ip.com`.

能够登录远程系统后, 分两种情况:

1. 如果远程系统本身就有 GPU, 即你用 `nvidia-smi` 能看到 gpu:
   - 一般用 docker 创建 (`docker pull => docker run => docker start`) 自己的容器来跑代码.
2. 如果远程系统是计算集群, 本身没有 GPU, 要申请计算资源:
   - 一般用 slurm 申请计算资源, 然后用 singularity 来创建容器 (当然你想用 docker 也是完全 ok 的, 只不过计算集群一般都只允许用户用 singularity), 运行代码.



### 8.4. GPU 相关

#### 8.4.1. CUDA Core V.S. Tensor Core

CUDA Core 是 GPU 计算的基础单元, 每个时钟周期进行一次标量运算. 主要争对 FP32 和 FP64 的计算问题.

Tensor Core 是从 Volta 架构开始引入的新硬件单元, 每个时钟周期进行一次矩阵运算, 大大加快矩阵运算效率. 支持更低精度的运算 (FP16, BF16, INT8, ...).

#### 8.4.2. 历代架构图

| 架构名称 | 代表产品 | 算力版本 (CC) | 核心创新点 | 新支持的计算精度 |
| :--- | :--- | :--- | :--- | :--- |
| **Volta** | V100 | 7.0 | **第一代 Tensor Core**，独立线程调度 | FP16, FP32 |
| **Ampere** | A100 | 8.0 | **TF32**, **BF16**, MIG (多实例 GPU), 结构化稀疏 | TF32, BF16, INT8 |
| **Hopper** | H100 | 9.0 | **Transformer Engine (Gen 1)**, FP8, DPX 指令集, NVLink 4.0 | FP8, FP16, BF16 |
| **Blackwell** | B200 | 10.0 | **Transformer Engine (Gen 2)**, FP4, 72-GPU NVLink 域 | FP4, FP6, FP8 |
| **Rubin** (还没出) | R100 | 11.0(?) | **HBM4**, 下一代 NVLink, 针对极致推理吞吐优化 | 高精度与低精度全覆盖 |

我们一般训练都会用算力 8.0 以上的 gpu, 因为 8.0 以上才支持 bf16 --- bf16 比 fp16 训练稳定很多.