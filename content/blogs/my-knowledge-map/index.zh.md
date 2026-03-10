---
title: "My Knowledge Map"
date: 2025-11-24T16:00:00-07:00
lastmod: 2026-03-10T02:57:01-07:00
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
  - Docker & Singularity (Apptainer)
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

## 3. NLP

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

### 3.1. 基础模型结构

{{<image
src="/imgs/blogs/my-knowledge-map/model-structure.drawio.svg"
width="100%"
caption=`Decoder-Only 模型的经典结构`
>}}

上图展示了 Decoder-Only Model 的经典结构; 省略了每层 Decoder Layer 的 Residual 结构和 Attention 内部的 Causal Mask.

#### 3.1.1. Embedding

Embedding 是模型内部的一个层, 和模型一起训练, 能够将一个 token id (int) map 到 `[H,]` 维度的 tensor.

#### 3.1.2. LayerNorm vs RMSNorm vs BatchNorm

对于输入 Tensor `[B,L,H]`, LayerNorm 和 RMSNorm 都在 `H` 维度 normalize (也就是对每个 token 的 hiddenstate 独立 normalize).

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

- LayerNorm：关心均值和方差
- RMSNorm：只关心整体幅值

从计算角度, LN 需要： 1. 求 mean; 2. 求 variance; 3. 做中心化. RMSNorm 只需要：1. 求平方均值; 2. 开根号; 3. 做缩放.

大语言模型有很多层，每一层都有 norm。即使 RMSNorm 每次只省一点点，叠加到几十上百层、长序列和大 batch 上，就会变成很明显的收益。所以从工程角度看，RMSNorm 是一种更划算的选择。

为什么不用 BN? BN 需要在一个 batch 上统计均值和方差。

但 NLP / LLM 中：
- 序列长度不同
- padding 很多
- token 分布差异很大
- batch 内样本语义高度异质

这会让 batch 统计变得噪声很大，也不稳定。

#### 3.1.3. Rotary Position Embedding (RoPE)

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

#### 3.1.4. Attention

$$
O=Attention(Q,K,V)=softmax\left(\text{Mask}(\frac{QK^T}{\sqrt{d_k}})\right)V
$$

1. **Causal Mask**: 对于单句输入 `I: (L, H)` (忽略 Batch 维度), 经过 attention 最终得到的输出矩阵 `O: (L, H)`; 我们在训练中期待 `O[i, :]` 这个 token 是 `I[:i, :]` 对应的 next token; 因此无论在推理还是训练时, 都需要一个下三角矩阵 mask 掉 `QK: (L, H)` 的 "未来" 部分: 比如对于 `QK` 的第一行, 只有第一个值有效, 这样 `O[0, :]` 就表示了给模型输入的第一个 token, 它预测出的是什么 token.
2. **Attention Mask**: 标注哪些部分用来计算 attention, 比如 pad token 在 attention mask 中就应该标记为 0.
3. **为什么除以 $\sqrt{d}$**: 因为点积 \(q \cdot k\) 的方差会随着维度 \(d\) 增大而变大，不除以 \(\sqrt{d}\) 会让 softmax 输入过大、容易饱和，导致梯度变小、训练不稳定，所以要用 \(\sqrt{d}\) 把尺度归一化。
4. **MHA** 是每个 query head 都有自己独立的 K/V head，**GQA** 是多个 query head 共享一组 K/V head，**MQA** 则是所有 query head 共用同一组 K/V head，所以它们本质上是在“KV 共享程度”和“效果与推理效率的权衡”上逐级加强。

#### 3.1.5. Sampling

LMHead 的输出是 `(B, L, V)`, `V` 是 vocab size, 后续要通过 sampling 过程选出具体是哪个 token:

- top-p sampling
- top-k sampling
- temperature sampling

**一般 temp 设置为 0.6-0.8 模型的效果最好**, 因为此时模型有更多的采样可能; 每次都只采纳最高 prob 的 token 会使效果变差.

### 3.2. 推理引擎和典型加速策略
#### 3.2.1. Model Parallel & Tensor Parallel & Pipeline Parallel

这三个并行策略主要争对 "大模型" (单卡放不下) 推理场景.

- MP: 一个 GPU 放不下一整个模型, 因此把模型**按顺序**切分放到不同 GPU 上**串行**执行; MP 是 transforers 库默认的并行策略; 缺点是串行执行不同 GPU 需要等待 (当前 GPU 需要等上一个 GPU 的数据传过来才能计算), 速度慢.
- TP: 把模型的一层 (可以认为是若干个矩阵计算操作, 例如一层是一个 Transformer Layer) 切分到不同 GPU **并行**执行 (例如, $A \times B$ 可以切分成 $A_1 \times B$ 和 $A_2 \times B$ 并行算); 速度快并且能降低单卡的显存占用, 但需要手动修改矩阵计算的代码; 目前工业界已经对 TP 做到了很大的支持, 推理引擎 (例如 vllm, sglang) 都已经默认支持 TP; 但是由于 TP 需要模型使用 "支持 TP 计算" 的模型层定义, 因此可以看到推理引擎内都会额外自己定义一遍模型.
- PP: 可以理解为高级的 MP, 把模型的不同部分切分到不同 GPU, 而相比 MP 进一步做流水线优化 --- 有些部分的计算不依赖于另一些部分的计算结果, 所以可以做到流水线并行; 目前对 PP 的支持都需要模型的开发者自己手动在推理引擎内部打补丁, 所以普及率远不如 TP.

#### 3.2.2. 常用推理引擎

- vllm: 老牌开源引擎, 提出 paged attention
- sglang: 社区更加活跃, 提出 radix attention (了解即可)
- tensorrtLLM: NV 亲自支持, 纯 C++ 实现

#### 3.2.3. 推理加速核心技术

1. PagedAttention: 解决多 request 不停申请 KV Cache 在 CUDA Memory 上的 Memory Fragmentation 问题.
2. Prefill-Decode Aggregation/Disaggregation: 是否 Prefill 和 Decode 一起算. 什么叫一起算: 当前 batch 同时包含 prefill request 和 decode request (推理时所有 input 一定时要组 batch 的, 比如通过对较短 input 填充 pad token id 进而组成 `(B, L)` 的 input batch, 这里 `B` 就是表示有 B 个 request) .
3. Continuous Batching: 批量 request 时的 decoding 阶段持续把新到请求动态插入同一批次 (假设设定死了 `max_batch_size=10`, 一个用户提交了 10 个 requests, 还没跑完时另一个用户又提交了 10 个 requests; 第一个用户提交的 10 个 requests 中有 8 个很快跑完了, 剩下 2 个一直跑不完; 这时候利用 Continuous Batching 技术能将第二个用户的 8 个 requets 和第一个用户的剩下 2 个 requets 组成一个新 batch 跑).
4. Chunked Prefill: 把长 prompt 的 prefill token ids 切开、让计算与其他请求交错执行以减少首 token 等待和显存峰值 (假设定死了 GPU 一次只能处理 `100` 个 token, 如果有个 prefill input 占 `90` 个 token, 那当前 batch 其他 request 没法跑了, 所以把 `100` 个 token 切开一点点算, 不要一次全算).
5. Prefix Caching: 如果多个 request 共享 prefill token ids, 比如问十个一样的问题, 就没必要重复计算这十个问题的 KV Cache 了 --- 我们为每个 prefill request 计算 hash value, 如果匹配上就利用之前计算过的 kv cache.

### 3.3. 训练
#### 3.3.1. Data Parallel & Distributed Data Parallel & Fully Sharded Data Parallel

这三个策略主要用于训练加速.

- DP: 每个 GPU 都有一份完整的模型副本, 每个 GPU 处理不同的数据 batch, 每个 GPU 计算完 loss 和梯度后, 需要把梯度同步到其他 GPU 上; DP 的优点是实现简单, 缺点是每个 GPU 都需要存储完整的模型副本, 显存占用大.
- DDP: 是 DP 的一种优化实现, 通过使用 NCCL 库进行高效的梯度同步 (RingAllReduce), 可以显著提升 DP 的效率; DDP 是 PyTorch 官方推荐的分布式训练方式. 使用 DDP 的方式是从 transformers 库中 import DDP 模块, 直接包裹模型; 另外 dataloader 需要一些额外配置 (例如 DistributedSampler) 来保证每个 GPU 处理不同的数据; 启动 DDP 我们一般会用 `torchrun` 命令来 launch 训练脚本, 如果你有 4 个 GPU, 在 launch 时就指定启动 4 个进程 (进程对应一个 GPU); DDP 的缺点是每个 GPU 仍然需要存储完整的模型副本, 显存占用大.
- FSDP: 是 DDP 的进一步优化, 通过把模型参数切分到不同 GPU 上 (Sharding), 可以显著降低每个 GPU 的显存占用; FSDP 的使用方式和 DDP 类似, 也是从 transformers 库中 import FSDP 模块, 直接包裹模型. FSDP 和 ZeRO 的三个 Stage 有千丝万缕的关系, 可以认为 FSDP 是 pytorch 官方实现的 ZeRO (详见下文 ZeRO).

#### 3.3.2. ZeRO

Zero 优化技术看这篇: {{<text url="https://zhuanlan.zhihu.com/p/694880795">}}Zero Stage 1, 2, 3{{</text>}} (至少了解训练中有哪些显存占用, 每个 stage 优化的点是什么).

#### 3.3.3. Gradient Checkpoint

前向传播时不保存某些层的中间激活，只保留少量“检查点”；反向传播走到这段时，再用检查点重新做一遍前向，把缺失的激活现算出来，再继续求梯度。

更具体一点，通常是这样实现的：

1. 把模型切成若干段: 例如一个很深的 Transformer，有 24 层，你可以每 4 层或每 6 层作为一个 segment。

2. 前向时只保存 segment 边界的张量: 正常训练里，每层输出激活都会留着给 backward 用。checkpoint 后，只保存每段输入/输出，段内各层的中间激活不保留。

3. 反向时按需重算该段前向: 当 backward 需要某一段内部某层的激活时，框架会拿这段的检查点输入，再临时跑一次这段 forward，恢复出中间激活，然后立刻计算梯度。

用“额外计算”换“更低显存”; 所以它本质上是：**显存下降，训练时间上升**。

常见情况是显存省很多，但计算开销增加大约 20% 到 40%，具体看切分方式。

#### 3.3.4. SFT

SFT 我一般用 trl 或者 unsloth 进行训练. trl 是 huggingface 的官方库, unsloth 是 trl 的优化版, 自己写了一些 kernel, 并且自动做 memory offload.

LoRA 的具体做法是创建两个矩阵 (LoRA Adapter): `B: (H, r)` `A: (r, H)`, (通常) 对 LLM 的每层 decoder 内的 `q_proj` 和 `v_proj` 做 LoRA:

原来: $Q = W_qX$, $V = W_vX$ ---> 现在: $Q = W_qX + BAX$, $V = W_vX + BAX$

一般把 B 初始化为 0, A 随机初始化, 这样能保证开始训练前模型输出和原始一致.

#### 3.3.5. PPO

#### 3.3.6. DPO

#### 3.3.7. GRPO

#### 3.3.8. DAPO

