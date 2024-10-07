---
title: "Dive into Paged Attention"
date: 2024-10-07T12:00:00+08:00
lastmod: 2024-10-07T23:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - vllm
    - continuous-batching
    - paged-attention
categories:
    - deeplearning
tags:
    - python
    - vllm
    - attention
description: Dive into the paged attention mechanism of VLLM.
summary: Dive into the paged attention mechanism of VLLM.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. 证明 Attention 的 \(O_i\) 只与 \(Q_i\) 有关

Attention 的公式如下:

\[
O=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
\]

假设 \(Q=\begin{bmatrix}Q_0\\Q_1\end{bmatrix}\), \(K=\begin{bmatrix}K_0\\K_1\end{bmatrix}\)

那么:

\[
O=softmax(\frac{\begin{bmatrix}Q_0K_0^T&Q_0K_1^T\\Q_1K_0^T&Q_1K_1^T\end{bmatrix}}{\sqrt{d_k}})V
\]

令:

\[
A=\begin{bmatrix}A_0\\A_1\end{bmatrix}=\begin{bmatrix}Q_0K_0^T&Q_0K_1^T\\Q_1K_0^T&Q_1K_1^T\end{bmatrix},f(x)=\frac{softmax(x)}{\sqrt{d_k}}
\]

此时, \(A_1\) 只和 \(Q_1\) 有关, 和 \(Q_0\) 无关, 那么:

\[
\begin{bmatrix}O_0\\O_1\end{bmatrix}=O=\begin{bmatrix}f(A_0)\\f(A_1)\end{bmatrix}V=\begin{bmatrix}f(A_0)V\\f(A_1)V\end{bmatrix}
\]

因此, \(O_i\) 只和  \(A_i\) 相关, 而根据 \(A\) 的设定, \(A_i\) 只和 \(Q_i\) 相关, 即:

Attention 矩阵的第 \(i\) 个输出只和第 \(i\) 个 \(Q\) 有关, 和之前的 \(Q\) 无关.

**总结**:

- 在预测下一个 token 时，只需对新 token 计算对应的 `Q_new`，并与之前已经缓存的 `K_cache` 和 `V_cache` 进行注意力计算。
- 新的 `K_new` 和 `V_new` 会被加入到缓存中，继续为下一个 token 生成提供基础。
- 整个过程避免了对所有历史 token 的重复计算，大幅提高了效率。

## 2. KV Cache 的增量过程
### 2.1. 初始输入（完整序列）计算：

- 对于初始的输入序列 `(seq_len, embed_dim)`，我们通过线性变换得到 `Q`、`K` 和 `V`，它们的形状都是 `(seq_len, embed_dim)`。
- 使用 `Q` 和 `K` 进行点积计算注意力分数，然后结合 `V` 计算得到输出 `(seq_len, embed_dim)`，这是第一次对初始序列的完整计算。

### 2.2. 预测下一个 token 时的增量计算：

在预测下一个 token 时，不需要对整个序列再进行完整的 `Q`、`K`、`V` 计算，而是只需对新生成的 token 进行一次增量计算。这时的操作流程如下：

1. **输入新的 token**：将已经生成的 token（其形状为 `(embed_dim,)`）作为输入，通过线性变换得到该 token 对应的 `Q_new`，形状为 `(embed_dim,)`。
2. **与之前缓存的 `K` 和 `V` 进行注意力计算**：
    - 使用 `Q_new` 与之前已经计算并缓存的 `K` 和 `V` 进行注意力计算。
    - 这里的 `K_cache` 和 `V_cache` 分别是之前每次生成 token 时得到的 `K` 和 `V`，它们的形状是 `(seq_len, embed_dim)`，即缓存了从最初输入序列到当前已经生成的所有 token 的 `K` 和 `V`。
    - `Q_new` 可以直接与 `K_cache` 进行点积，得到注意力分数，然后结合 `V_cache` 得到新的输出。
3. **更新 `KV Cache`**：
    - 新的 `K_new` 和 `V_new` 会通过线性变换得到（形状为 `(embed_dim,)`），并将它们添加到 `K_cache` 和 `V_cache` 的末尾，使得缓存的 `K_cache` 和 `V_cache` 不断增大，以备后续使用。
4. **输出**：通过注意力计算后的输出形状为 `(embed_dim,)`，即新生成的 token。

![vllm-prefill-and-decode](/imgs/blogs/dive-into-paged-attention/vllm-prefill-and-decode.png)

### 2.3. Python 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        k_dim=None,
        k_cache: torch.Tensor = None,  # (cache_len, num_heads, head_size)
        v_cache: torch.Tensor = None,  # (cache_len, num_heads, head_size)
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.k_dim = embed_dim if k_dim is None else k_dim
        assert (
            self.head_size * num_heads == embed_dim
        ), "embed_dim should be divisible by n_heads"
        self.wq = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))
        self.wk = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))
        self.wv = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.wout = nn.Parameter(torch.randn(self.embed_dim, self.k_dim))

    def forward(self, x):
        # x: (seq_len, embed_dim)
        seq_len, embed_dim = x.size()
        q = torch.matmul(x, self.wq)  # (seq_len, embed_dim)
        k = torch.matmul(x, self.wk)  # (seq_len, embed_dim)
        v = torch.matmul(x, self.wv)  # (seq_len, embed_dim)

        # q -> (n_heads, seq_len, head_size)
        q = q.view(seq_len, self.num_heads, self.head_size).transpose(0, 1)
        # k, v -> (seq_len, n_heads, head_size)
        k = k.view(seq_len, self.num_heads, self.head_size)
        v = v.view(seq_len, self.num_heads, self.head_size)
        if self.k_cache is not None and self.v_cache is not None:
            k = torch.cat([self.k_cache, k], dim=0)
            v = torch.cat([self.v_cache, v], dim=0)
            # New cache_len = seq_len + cache_len
        # k -> (n_heads, head_size, cache_len)
        k = k.transpose(0, 1)
        # v -> (n_heads, cache_len, head_size)
        v = v.transpose(0, 1)

        # scores: (n_heads, seq_len, cache_len)
        scores = (torch.matmul(q, k.transpose(-2, -1))) / (self.head_size**0.5)
        # attn_weights: (n_heads, seq_len, cache_len)
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_output: (n_heads, seq_len, head_size)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (seq_len, n_heads, head_size)
        attn_output = (
            attn_output.transpose(0, 1)  # (seq_len, n_heads, head_size)
            .contiguous()  # Make sure the memory is contiguous
            .view(seq_len, embed_dim)  # (seq_len, embed_dim)
        )
        # out: (seq_len, embed_dim)
        out = torch.matmul(attn_output, self.wout)
        return out

```

## 3. vllm 的在线推理 & 离线推理

离线推理: `vllm.LLM`
在线推理: `vllm.AsyncLLMEngine`

### 3.1. 离线推理

示例:

```python
# "examples/offline_inference.py"
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
     "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

`LLM.generate` 调用 `LLM._run_engine` 生成 outputs.

`_run_engine` 内部:

```python
# ...
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
            # ...
```


## 4. Continuous Batching

> 参考: https://blog.csdn.net/qq_27590277/article/details/135710435

### 4.1. BatchMaker: Low Latency RNN Inference with Cellular Batching

论文: [Low Latency RNN Inference with Cellular Batching](https://madsys.cs.tsinghua.edu.cn/publication/low-latency-rnn-inference-with-cellular-batching/EUROSYS2018-gao.pdf)

BatchMaker 是一个为 RNNs 设计的 serving 系统，它以 RNN Cell 为粒度进行调度和 Batching。RNN 使用相同权重对不同输入进行计算。当收到请求时，BatchMaker 将用于处理请求的数据流图分解为 RNN Cell（即一个iteration step），并以 Cell 的粒度进行执行调度，并批处理相同的单元执行。由于每个 RNN Cell 始终执行完全相同的计算，BatchMaker 可以无论单元的位置（即标记索引）如何，都以 Batching 方式执行多个 RNN Cell。通过这样做，BatchMaker 允许新到达的 RNN 请求加入（或已完成的请求离开）当前执行的批次，而无需等待批次完全完成。

![celluar-batching](/imgs/blogs/dive-into-paged-attention/CellularBatching.png)

### 4.2. ORCA：更适合 Transformer 的 Batching 方法

论文: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)

ORCA 借鉴 BatchMaker 方法，将它适配到 Transformer Decoder 生成过程。虽然 Transformer Decoder 和 RNN 在生成过程中都是逐个 token 地迭代生成，但它们之间存在一些本质区别。

1. 首先，Transformer Decoding 阶段每个迭代时，将当前 token 和之前生成的 token 序列拼接起来传入模型。尽管每次只生成一个 token，计算量近似，但每个迭代的 KVCache 的长度会逐渐增加。
2. 其次，Decoder 在进行解码时需要进行 Prefill 过程，这是 RNN 没有的。Prefill 计算是一堆 token 一起算，和 Decoding 阶段计算模式截然不同。前者是计算密集，后者是访存密集。

为了解决这些问题，OCRA 提出了两个设计思路：Iteration-level Batching 和 Selective Batching。 Iteration-level Batching 可以看作是对 BatchMaker Cell 粒度处理思想的一种致敬，而 Selective Batching 则是针对 Transformer 的独特处理，以支持在 batch size 和 input sequence 这两个维度动态变化对 Batching 执行的影响。

由于 Attention 机制和 FNN 的 Batching 方式不同。Linear 层可以将 batch size 和 seq_len 这两个维度融合为一个维度，类似于 Efficient Transformer 的思想，而 Attention 则不行。因此，一个 Transformer Layer 可以划分为 PreAttn、Attn 和 PostAttn 三个部分。从而支持 prefill 阶段和 decoding 一个 step 打成一个 batch 处理。如下图所示，QKV Linear 和 Attn Out Linear 打成一个batch size=7。Attn 的计算没有打 Batch，每个 request 单独处理。所以在 Attn 前后有 Split 和 Merge 操作。

![orca-transformer-execution](/imgs/blogs/dive-into-paged-attention/ORCA-transformer-execution.png)

**OCRA 还没考虑 KV Cache 内存管理优化，它每个序列预先分配 max token 数的作为 KV Cache 显存空间。OCRA 的实验都是按照 max token 来生成，不会考虑遇到 eos 的情况。**

### 4.3. 2023年更多 Continuous Batching 的变种

2023年 Continuous Batching 迎来了大发展，在 vLLM 推动下已成为推理框架事实标准。不同框架实现有差别，主要体现在对 prefill 处理的方式上。将 prefill 单独处理还是和 decoding 融合，以什么样的粒度融合，有一些讲究。

#### 4.3.1. vLLM (UC Berkeley)

SOSP 2023 的论文 vLLM，也是热门开源项目，其创新点paged attn（PA），减少内存碎片，增加memory efficiency，增大batch size从而增加吞吐。Batching策略是为PA设计服务的，所以没有照搬OCRA的实现。

和ORCA不同之处在于，vLLM Batching时候prefill和decoding是分开的，一个Batching step要么处理decoding要么处理prefill。这样实现比OCRA更简单了，prefill直接调用xformers处理计算密集的prefill attn计算；decoding手写CUDA PA处理访存密集的attn计算。

vLLM 之所以没有采用 OCRA 设计，是因为 vLLM 的 PA 是手写 CUDA Kernel 实现的，可以处理 sequence 长度不同的输入，Attn 的 Batching 方式可以和 Non-Attn 部分统一。因此，一个糙快猛方法是不采用 Selective Batching 的设计了，所 Decoding 整体一起处理一个 Batch 的 step 计算，prefill 不和 decoding step 融合。如果把 prefill 计算和一个 decoding step 融合，则还需要拆分 Attn 和 Non-Attn 了，Attn 实现也更更复杂了，不利于展示 PA 的思想。

不过因为Prefill过程会抢占decoding的step前进，如果输入prompt sequence length过长，所有decoding过程都需要等待，造成大家更长的延迟，因此留下了一些优化空间，这后来这也造成了和DeepSpeed的一段孽缘。

#### 4.3.2. FastGen (DeepSpeed)

微软 DeepSpeed 团队2023年11月在 MII 项目中提出了一种 Continous Batching 变种 SplitFuse，在发布时把 vLLM 当靶子打，vLLM 随后还击，逐渐演化成成为两个大门派的口水战。

SplitFuse 的想法是，对长 prompt request 被分解成更小的块，并在多个 forward step 中进行调度，只有最后一块的 forwar d完成后才开始这个 prompt request 的生成。对短 prompt request 将被组合以精确填充 step 的空隙。每个 step 的计算量基本相等，达到所有请求平均延迟更稳定的目的

## 5. Paged Attention

> References:
>   - [vLLM Paged Attention](https://docs.vllm.ai/en/latest/dev/kernel/paged_attention.html)
>   - [vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现](https://zhuanlan.zhihu.com/p/673284781)

### 5.1. Paged Attention 中 KV Cache 的变化形式

![paged-attention-animation](/imgs/blogs/dive-into-paged-attention/paged-attention-animation.webp)

1. **Page 的概念**：  
    - 在 Paged Attention 中，长序列被划分为若干小块（即 "pages"），每个 page 包含多个 tokens。每次 Attention 操作只在当前 page 内的 tokens 之间计算，也就是在有限的上下文窗口内计算 Attention，而不是在整个序列上计算。
    - 因此，Key 和 Value 缓存会按 page 进行管理，每个 page 对应一组 K 和 V 值。
2. **生成新 token 时的 KV Cache 变化**：  
    - 当生成新 token 时，新的 Key 和 Value 会根据这个新的 token 计算出来，并且会被添加到当前 page 的 **KCache** 和 **VCache** 中。
    - **KCache 和 VCache** 的尺寸只在当前 page 中变化。在生成到新 page 时，才会为新 page 分配新的缓存空间。
3. **KV Cache 的变化**：  
    - KV Cache 可以理解为 `(num_blocks, num_heads, head_size, block_size)`，其中:
        - `num_blocks` 是 page 的数量 (或者称为 block 数量，每个 block 对应一个 page).
        - `num_heads` 是 Attention 头的数量.
        - `head_size` 是每个 Attention 头的维度大小.
    - 随着新 token 的生成，每个 page 会增加新的 Key 和 Value，但它们被限制在当前 page 内部。
    - 当当前 page 填满时，系统会开启一个新的 page，新的 KV 值将被存入新 page 的缓存中，因此会在 `num_blocks` 维度上增加一个新的 block。

### 5.2. Paged Attention Kernel 计算流程

首先，按照 CUDA 编程模型对任务进行并行划分: 

- Grid 大小 `(num_heads, num_seqs)` .
- Grid 中每个 CUDA thread block 大小 `(NUM_THREADS)`，`NUM_THREADS` 是常量默认为128，也就说每个thread block包含128个线程，负责完成output矩阵一行（包含head_size个元素）结果的 attention 计算任务.
- Thread block 中的线程进一步划分若干个 WARP。众所周知，WARP 是 GPU 一个基本的执行单元，由32个线程组成，这些线程以 SMIT 方式在硬件上同时执行相同的指令，在不同的数据上进行操作.
- 在 PA 中比较特殊的是，WARP 内 32 个线程进一步划分为 `blk_size` 个 thread group ，这和 paged KVCache 设计 x 息息相关的，马上会细讲.

![paged-attention-compute-flow](/imgs/blogs/dive-into-paged-attention/paged-attention-compute-flow.webp)

在上图的左侧部分，我们看到了 Q 矩阵，这部分描述了从显存读取 Q 数据到共享内存的过程。在这个过程中: 

- 一个 CUDA thread block 会读取图中 Q 矩阵的一行（包含head_size个元素）并将其存入共享内存。(因此一共需要 `seqLen * numHeads` 个 CUDA block)
- 这个过程是通过一个循环来实现的，在每次迭代中，**每个 thread group 会读取 16 字节的 Q 数据**（例如，如果使用 float16，那么就是 8 个元素）。
- 由于一个 WARP 被分为 `BLOCK_SIZE` 个 thread group, 所以每个 WARP 会读取 `16 * blk_size` 字节的Q数据，这些数据对应于一个 sequence 的一个head，由 CUDA GRID 索引指定。
- 当循环访问结束后，共享内存存储 Q 行的一部分。如下图所示，绿色部分表示存储在一个线程读入共享内存中的数据。

![paged-attention-thread-group](/imgs/blogs/dive-into-paged-attention/paged-attention-thread-group.webp)

```cpp
// Load the query to registers.
// Each thread in a thread group has a different part of the query.
// For example, if the the thread group size is 4, then the first thread in
// the group has 0, 4, 8, ... th vectors of the query, and the second thread
// has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
// q is split from a qkv tensor, it may not be contiguous.
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
        i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
}
__syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs
```

图 1 中上面部分 K 矩阵部分描述了从显存读取 K Cache 到寄存器的过程。每个序列的 K Cache 包含 `cxt_length * num_kv_heads * head_size` 个元素，**但由于采用了页式内存管理，这些元素在内存中的存储并不连续**。

每个 CUDA thread block 只负责计算一个 sequence 一个 head 的 \(QK^T\)，因此只需要`ctx_length * head_size` 个 K Cache 元素。然而，由于 `ctx_length` 维度的存储是不连续的，并且以 `blk_size` 个 token 为粒度分布在不同的内存地址，我们需要根据 query 的 `head_idx` 和 `seq_idx` 访问 `block_table` 以找到 K Cache 的 `physical_block_num`。为了方便后续的描述，我们可以将 K Cache 视为 `(:, headSize)` 的形状，其中 `head_size` 个元素组成一行。

```cpp
// Iterate over the key blocks.
// Each warp fetches a block of keys for each iteration.
// Each thread group in a warp fetches a key from the block, and computes
// dot product with the query.
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
```

k_cache 的布局为 `(num_blocks, num_kv_heads, head_size/x, block_size, x)`，这是为了优化写入 shared memory 的操作。在 Q 和 K 矩阵的同一行元素被读入寄存器并进行点乘运算后，结果需要被存入 shared memory。

```cpp
const int physical_block_offset =
    (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
K_vec k_vecs[NUM_VECS_PER_THREAD];
```

如果一个 WARP 中所有线程都计算 Q、K 同一行数据，会导致写入 shared memory 的同一个位置，这将造成 WARP 内不同线程顺序地写入。因此，为了优化，WARP 的线程最好计算 Q 和 K 的不同行数据。因此，在设计 k_cache 布局时，我们将 `block_size` 放在比 `head_size` 更低的维度。

由于 `warp_size` 大于 `block_size`，我们需要将 `head_size` 拆分为 `head_size/x` 和 `x` 两个维度，借 `x` 到最低维度，以确保每个线程读入的数据量和计算量都足够大。最后，每个线程组派一个线程去写入 shared memory，这样一个 warp 有 `block_size` 个线程并行写入 shared memory，从而增加了 shared memory 的访问带宽。这种设计策略是为了实现高效的并行计算和内存访问，以提高整体的计算性能。

```cpp
if (thread_group_offset == 0) {
    // Store the partial reductions to shared memory.
    // NOTE(woosuk): It is required to zero out the masked logits.
    const bool mask = token_idx >= seq_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    // Update the max value.
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

在代码实现中，访问 K 矩阵需要一个循环，该循环使得 CUDA 线程块中的所有 WARP 依次访问 `num_block` 个页面。在每次循环迭代中，每个 WARP 负责访问连续的 `block_size` 个 K Cache 行，这涉及到的数据量为 `blk_size * head_size` 个元素。同时，每个 thread group 负责访问 K Cache 的一行，将 `head_size` 个元素加载到自己的寄存器中。接着，寄存器中的Q和K数据元素立即进行点乘运算，运算结果被写入 shared memory 中。因此，线程块的 shared memory 存储了一行 \(QK^T\) 的结果，包含 `ctx_length` 个元素。这种实现方式充分利用了 CUDA 的并行计算能力，以提高数据处理的效率。

然后，thread block 对 shared memory 中元素进行 max，sum 方式 reduction，然后计算得到 softmax 结果。

```cpp
// Get the sum of the exp values.
float exp_sum = 0.f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

// Compute softmax.
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
__syncthreads();
```
  
图 1 右边 V 矩阵部分描述从显存读 V Cache 到寄存器过程。和 K Cache 一样，CUDA thread block 依次访问 `num_blk` 个物理块到寄存器，每个 warp 负责 `blk_size` 个 token 的 page 内存，page 的真实物理地址同样需要进行索引。不过这里不需要以 thread group 为单位访问 16 字节，而是每个 thread 访问 16 字节的元素。访问完就可以与 shared memory 的 \(softmax(QK^T)\) 中间结果对应位置 16 字节的数据进行点乘，得到一个 float 结果，写到 output 对应位置中。  
  
为什么 V Cache 的 layout 是 `[num_blocks, num_kv_heads, head_size, block_size]`，和 K Cache layout 不一样？ 这是因为 V 要去做点乘的对象在 shared memory，只需要读，不涉及并行写的问题。

和 FlashAttention（FA）有什么不同？结合我的图和中间 FAv2 的流程图对比就一目了然了。FA 用了两层循环，每次写一个 Tile 的 output tensor，而 PA 一直只有一层循环，每次写一行 output tensor。因为每次都有整行的 \(QK^T\) 中间结果，不需要 online softmax 这种花哨技巧。

![flash-attention.webp](/imgs/blogs/dive-into-paged-attention/flash-attention.webp)

### 5.3. Paged Attention Kernel 代码精读

```cpp
// Grid: (num_heads, num_seqs, 1).
template<
typename scalar_t,
int HEAD_SIZE,
int BLOCK_SIZE,
int NUM_THREADS,
int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
... // Other side args.
const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
... // Other side args.
)
```

#### 5.2.1. 模板参数说明

- `scalar_t` 元素类型 (实际代码中还有 `cache_t` 表示 KV cache 的元素类型).
- `HEAD_SIZE` 每个 head 中元素数量.
- `BLOCK_SIZE` 每个 PA block 中的 token 数量.
>  KV cache 被存储在不同 Paged Attention blocks. 每个 PA block 存储一个 head 中 `BLOCK_SIZE` 个 token.   
>  例如, 若 `BLOCK_SIZE=16`, `HEAD_SIZE=128`, 则一个  PA block 能存储一个 head 的 `16 * 128 = 2048` 个元素. (这个例子暂时没看懂, token 和 `HEAD_SIZE` 有啥关系?)  
>  每个 PA block 可能只包含一部分的 context tokens.  
>  从 page 角度看, KV cache 是若干个 page 的集合; 
- `NUM_THREADS` 每个 CUDA thread block 中 thread 的数量.
- `PARTITION_SIZE` 参与 TP 的 GPU 数量, 默认 0 表示单卡. (以下都以单卡为例说明)

#### 5.2.2. 额外的 shape 参数说明

- `num_seqs`: 本次推理请求 sequence 数目.
 > 由于这个 kernel 只处理 decode 阶段单 query attention, 所以实际上每个 sequence 只有一个 query token. 因此 `num_seqs` 等于当前 batch 中总共正在处理的 token 数.
- `num_heads`: Q 的 head 数目
- `num_kv_heads`: KV 的 head 数目, 对于 MHA 其值和 `num_heads` 相同; 如果是 GQA, MQA 则 `num_kv_heads` 小于 `num_head`.
- `head_size`: 即 `HEAD_SIZE`
- `k_cache: (num_blocks, num_kv_heads, head_size/x, block_size, x)`, 其中 `x` 表示一个 **Vec** 的大小 (即: `VEC_SIZE`)，如 `float16 -> 16 / sizeof(float16) = 8`。
> **Vec**: The vec is a list of elements that are fetched and calculated together. For query and key data, the vec size (`VEC_SIZE`) is determined so that each thread group can fetch and calculate 16 bytes of data at a time. For value data, the vec size (`V_VEC_SIZE`) is determined so that each thread can fetch and calculate 16 bytes of data at a time. For example, if the `scalar_t` is FP16 (2 bytes) and `THREAD_GROUP_SIZE` is 2, the `VEC_SIZE` will be 4, while the `V_VEC_SIZE` will be 8.  

```cpp
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
```

Paged 内存管理相关的辅助数据结构：
- `blk_size`：也就是 `BLOCK_SIZE`，是 KVCache page 的最高维，KVCache 是若干个 page 的集合，每个 page 存 `(blk_size, num_head，head_size)` 个 K、V 的元素。
- `head_mapping`: `[num_heads]` 用于 MQA, GQA，确定用的 KV_head
- `block_tables`: `[num_seqs, max_num_blocks_per_seq]`, 映射表，表示每个 sequence 映射到哪几个 block 上
- `context_lens`: `[num_seqs]`, 用于变长


- **Sequence**: client 的请求. 由于 PA kernel 只在 decode 阶段被调用, 所以一条请求对应的输入 Q 只有一个 token. 多个 sequence 的输入组成一个 batch, 所以输入 Q 的 shape 表示为: `[num_seqs, num_heads, head_size]`, 其中 `[num_heas, head_size]` 对应一个  token.
- **Vec**: 被线程 fectch 和 calculate 的一串 elements. 
- **CUDA Thread Block**
    - 🤔Processes the calculation between one query token and key tokens of a whole context.
    - CUDA grid shape is `(num_heads, num_seqs, max_num_partitions)`. Therefore, each thread block only handles the calculation for one head, one sequence, and one partition.
- **Thread Group** 
    - Consists of `THREAD_GROUP_SIZE` threads
    - Fetches one query token data (`HEAD_SIZE` elements), while each thread itself only handles a part of one query token data (`NUM_ELEMENTS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE` elements). 
    - Within each warp, every thread group will fetch the same query token data, but will multiply it with different key token data.
    - Fetches and calculates `VEC_SIZE = 16 / THREAD_GROUP_SIZE / SIZE_OF_ELEM` elements at a time for Q and K. -> Each thread in a thread group fetches and calculates `VEC_SIZE` elements at a time.


    
