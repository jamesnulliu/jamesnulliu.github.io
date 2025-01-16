---
title: "APC, SD, and SF"
date: 2025-01-13T17:00:00+08:00
lastmod: 2025-01-16T19:20:00+08:00
draft: true
author: ["jamesnulliu"]
keywords: 
    - Automatic Prefix Caching (APC)
    - Speculative Decoding (SD)
    - Split Fuse (SF)
categories:
    - deeplearning
    - algorithm
tags:
    - c++
    - cuda
    - python
    - attention  
description: Explanation of Automatic Prefix Caching (APC), Speculative Decoding (SD), and Split Fuse (SF).
summary: Explanation of Automatic Prefix Caching (APC), Speculative Decoding (SD), and Split Fuse (SF).
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Automatic Prefix Caching

### 1.1. Introduction

Automatic Prefix Caching (APC) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

```python {linenos=true}
# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_prefix_caching=True
)
```

### 1.2. Example Workloads and Limitations

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. 

- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session.

APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).

### 1.3. Implementation


``` {linenos=true}
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

In the example above, the KV cache in the first block can be uniquely identified with the tokens “A gentle breeze stirred”. The third block can be uniquely identified with the tokens in the block “laughed in the distance”, along with the prefix tokens “A gentle breeze stirred the leaves as children”. Therefore, we can build the following one-to-one mapping:

```python {linenos=true}
hash(prefix tokens + block tokens) <--> KV Block
```

In "vllm/core/block_manager.py", hash calculation happens {{<href text="here" url="https://github.com/vllm-project/vllm/blob/ad34c0df0f1b26b303a590133685b29e3daad20e/vllm/core/block_manager.py#L156">}} right berfore allocating the sequnce to block table.

This design achieves automatic prefix caching without the need of maintaining a tree structure among the KV blocks. More specifically, all of the blocks are independent of each other and can be allocated and freed by itself, which enables us to manages the KV cache as ordinary caches in operating system.

## 2. Speculative Decoding

### 2.1. Introduction

Speculative decoding {{<href text="Leviathan et al., 2023" url="https://arxiv.org/abs/2211.17192">}} is a key technique in reducing latency during token generation in large language models (LLMs). This approach leverages smaller models to handle simpler token predictions while utilizing larger models to verify or adjust those predictions. By doing this, speculative decoding accelerates generation without sacrificing accuracy, making it a lossless yet highly efficient method for optimizing LLM performance.

Why can speculative decoding reduce latency? Traditionally, LLMs generate tokens one at a time in an autoregressive manner. For example, given a prompt, the model generates three tokens T1, T2, T3, each requiring a separate forward pass. Speculative decoding transforms this process by allowing multiple tokens to be proposed and verified in one forward pass.

Here's how the process works:

1. Draft Model: A smaller, more efficient model proposes tokens one by one.
2. Target Model Verification: The larger model verifies these tokens in a single forward pass. It confirms correct tokens and corrects any incorrect ones.
3. Multiple Tokens in One Pass: Instead of generating one token per pass, this method processes multiple tokens simultaneously, reducing latency.

By using this approach, speculative decoding speeds up token generation, making it an effective method for both small-scale and large-scale language model deployments.

{{<image 
src="/imgs/blogs/apc-sd-and-sf/sd-example.png"  
width="80%"
caption=`As shown in the picture above, the draft model proposes five tokens: ["I", "like", "cooking", "and", "traveling"]. These are then forwarded to the target model for parallel verification. In this example, the third token, "cooking" (should be "playing"), was proposed inaccurately. As a result, only the first three tokens, ["I", "like", "playing"], are generated in this step.`
>}}

By using this approach, speculative decoding speeds up token generation, making it an effective method for both small-scale and large-scale language model deployments.

### 2.2. How Speculative Decoding Works in vLLM

In vLLM, speculative decoding is integrated with the system’s continuous batching architecture, where different requests are processed together in a single batch, enabling higher throughput. vLLM uses two key components to implement this:

- Draft Runner: This runner is responsible for executing the smaller model to propose candidate tokens.
- Target Runner: The target runner verifies the tokens by running the larger model.

vLLM’s system is optimized to handle this process efficiently, allowing speculative decoding to work seamlessly with continuous batching, which increases the overall system performance.

{{<image
src="/imgs/blogs/apc-sd-and-sf/sd-in-vllm.png"
caption=`Diagram illustrating how the draft and target runners interact within the vLLM batching system.`
>}}

To implement speculative decoding in vLLM, two crucial components had to be modified:

- **Scheduler**: The scheduler was adjusted to handle multiple token slots within a single forward pass, enabling the simultaneous generation and verification of several tokens.
- **Memory Manager**: The memory manager now handles the KV cache for both the draft and target models, ensuring smooth processing during speculative decoding.

{{<image
src="/imgs/blogs/apc-sd-and-sf/vllm-sd-system-archi.png"
width="80%"
caption=`System architecture of speculative decoding in vLLM. `
>}}

### 2.3. Types of Speculative Decoding Supported in vLLM

#### 2.3.1. Draft Model-Based Speculative Decoding

{{<image
src="/imgs/blogs/apc-sd-and-sf/draft-model-based-sd.png"
width="80%"
>}}

This is the most commonly used form of speculative decoding, where a smaller model predicts the next tokens, and a larger model verifies them. A common example would be using a Llama 68M model to predict tokens for a Llama 2 70B model. This approach requires careful selection of the draft model to balance accuracy and overhead.

Choosing the correct draft model is essential for maximizing the efficiency of speculative decoding. The draft model needs to be small enough to avoid creating significant overhead but still accurate enough to provide a meaningful performance boost.

However, **selecting the right draft model** can be challenging. For example, in models like Llama 3, finding a suitable draft model is difficult due to differences in vocabulary size. Speculative decoding requires that the draft and target models **share the same vocabulary**, and in some cases, this can limit the use of speculative decoding. Therefore, in the following sections, we introduce several draft-model free speculative decoding methods.

#### 2.3.2. Prompt Lookup Decoding

{{<image
src="/imgs/blogs/apc-sd-and-sf/prompt-lookup-decoding.png"
width="80%"
caption=`An example of prompt lookup decoding. Given the prompt, we build all 2-grams as the lookup key. The values are the three tokens following the lookup key. During generation, we will check if the current 2-gram matches any key. If so, we will propose the following tokens with the value.`
>}}

Otherwise known as n-gram matching, this approach is effective for use cases like summarization and question-answering, where there is a significant overlap between the prompt and the answer. Instead of using a small model to propose tokens, the system speculates based on the information already available in the prompt. This works particularly well when the large model repeats parts of the prompt in its answers.

#### 2.3.3. Medusa/Eagle/MLPSpeculator

{{<image
src="/imgs/blogs/apc-sd-and-sf/medusa-eagle-mlpspeculator.png"
width="80%"
caption=`In the example, three heads are used to propose tokens for the following three positions. Head 1 is proposing ["is", "\'", "the"] for the first position. Head 2 is proposing ["difficult", "is", "\'"] for the second position. Head 3 is proposing ["not", "difficult", "a"] for the third position. All heads take the output of the last transformer block as the input. `
>}}

In this method, additional layers (or heads) are added to the large model itself, allowing it to predict multiple tokens in a single forward pass. This reduces the need for a separate draft model, instead leveraging the large model’s own capacity for parallel token generation. Though preliminary, this method shows promise for improving efficiency as more optimized kernels are developed.

#### 2.3.4. Speculative Decoding Performance Insights: Speedups and Trade-offs

Speculative decoding offers significant performance benefits in **low-QPS (queries per second)** environments. For example, in testing on the ShareGPT dataset, vLLM demonstrated up to a 1.5x speedup in token generation when using draft model-based speculative decoding. Similarly, prompt lookup decoding has shown speedups of up to 2.8x when applied to summarization datasets, such as CNN/DailyMail.

{{<image
src="/imgs/blogs/apc-sd-and-sf/sd-performance-low-qps.png"
caption=`Performance comparison showing spec decode delivering up to 1.5x Speedup at QPS=1 Llama3-70B on ShareGPT with 4xH100 using draft model (turboderp/Qwama-0.5B-Instruct) and up to 2.8x Speedup at QPS=1 Llama3-70B on CNN Dailymail with 4xH100 using n-grams.`
>}}

However, in **high-QPS environments**, speculative decoding may introduce performance trade-offs. The extra compute required to propose and verify tokens can sometimes slow down the system when it is already compute-bound, as seen when the number of requests per second increases. In such cases, the overhead of speculative decoding can outweigh its benefits, leading to reduced performance.

{{<image
src="/imgs/blogs/apc-sd-and-sf/sd-performance-high-qps.png"
caption=`As high QPS, we see 1.4x slowdown Llama3-70B on ShareGPT with 4xH100, 1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100`
>}}

#### 2.3.5 How to Use Speculative Decoding in vLLM

See: {{<href text="How to Use Speculative Decoding in vLLM" url="https://blog.vllm.ai/2024/10/17/spec-decode.html#how-to-use-speculative-decoding-in-vllm">}}.

---

## 🔗References

1. {{<href text="vLLM | Automatic Prefix Caching - Introduction" url="https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html">}}
1. {{<href text="vLLM | Automatic Prefix Caching - Implementation" url="https://docs.vllm.ai/en/stable/automatic_prefix_caching/details.html">}}
1. {{<href text="How Speculative Decoding Boosts vLLM Performance by up to 2.8x" url="https://blog.vllm.ai/2024/10/17/spec-decode.html">}}
1. {{<href text="PyTorch | A Hitchhiker's Guide to Speculative Decoding" url="https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/">}}