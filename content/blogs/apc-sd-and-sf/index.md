---
title: "APC, SD, and SF"
date: 2025-01-13T17:00:00+08:00
lastmod: 2025-01-13T17:00:00+08:00
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

> 🔗**References**
> 1. [vLLM | Automatic Prefix Caching - Introduction](https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html)
> 1. [vLLM | Automatic Prefix Caching - Implementation](https://docs.vllm.ai/en/stable/automatic_prefix_caching/details.html)
> 1. [How Speculative Decoding Boosts vLLM Performance by up to 2.8x](https://blog.vllm.ai/2024/10/17/spec-decode.html)
> 1. [A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

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

In "vllm/core/block_manager.py", hash calculation happens [here](https://github.com/vllm-project/vllm/blob/ad34c0df0f1b26b303a590133685b29e3daad20e/vllm/core/block_manager.py#L156) right berfore allocating the sequnce into block table.

This design achieves automatic prefix caching without the need of maintaining a tree structure among the KV blocks. More specifically, all of the blocks are independent of each other and can be allocated and freed by itself, which enables us to manages the KV cache as ordinary caches in operating system.

## 2. Speculative Decoding

### 2.1. Introduction

Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) is a key technique in reducing latency during token generation in large language models (LLMs). This approach leverages smaller models to handle simpler token predictions while utilizing larger models to verify or adjust those predictions. By doing this, speculative decoding accelerates generation without sacrificing accuracy, making it a lossless yet highly efficient method for optimizing LLM performance.

Why can speculative decoding reduce latency? Traditionally, LLMs generate tokens one at a time in an autoregressive manner. For example, given a prompt, the model generates three tokens T1, T2, T3, each requiring a separate forward pass. Speculative decoding transforms this process by allowing multiple tokens to be proposed and verified in one forward pass.

Here's how the process works:

1. Draft Model: A smaller, more efficient model proposes tokens one by one.
2. Target Model Verification: The larger model verifies these tokens in a single forward pass. It confirms correct tokens and corrects any incorrect ones.
3. Multiple Tokens in One Pass: Instead of generating one token per pass, this method processes multiple tokens simultaneously, reducing latency.

By using this approach, speculative decoding speeds up token generation, making it an effective method for both small-scale and large-scale language model deployments.

{{<image 
src="/imgs/blogs/apc-sd-and-sf/sd-example.png"  
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
caption=`System architecture of speculative decoding in vLLM. `
>}}