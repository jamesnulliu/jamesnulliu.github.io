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


---

## 🔗References

1. {{<href text="vLLM | Automatic Prefix Caching - Introduction" url="https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html">}}
1. {{<href text="vLLM | Automatic Prefix Caching - Implementation" url="https://docs.vllm.ai/en/stable/automatic_prefix_caching/details.html">}}
1. {{<href text="How Speculative Decoding Boosts vLLM Performance by up to 2.8x" url="https://blog.vllm.ai/2024/10/17/spec-decode.html">}}
1. {{<href text="vLLM | Speculative Decoding" url="https://docs.vllm.ai/en/latest/features/spec_decode.html">}}
1. {{<href text="PyTorch | A Hitchhiker's Guide to Speculative Decoding" url="https://pytorch.org/blog/hitchhikers-guide-speculative-decoding">}}
1. {{<href text="Accelerating Production LLMs with Combined Token/Embedding Speculators" url="https://arxiv.org/abs/2404.19124">}}