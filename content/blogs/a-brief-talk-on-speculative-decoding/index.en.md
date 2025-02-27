---
title: "A Brief Talk on Speculative Decoding"
date: 2025-02-21T01:14:06+08:00
lastmod: 2025-02-21T14:06:00+08:00
draft: true
author: ["jamesnulliu"]
keywords: 
    - speculative decoding
categories:
    - deeplearning
tags:
    - transformer
    - llm
    - vllm
description: A brief talk on speculative decoding in large language models.
summary: A brief talk on speculative decoding in large language models.
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---


{{<gdocs src="https://docs.google.com/spreadsheets/d/1F_KXAAow336r57Gw1JAh-wZe3EzR0e2s6f4PJBCLhLM/edit?usp=sharing">}}

## 1. Speculative Decoding

### 1.1. Introduction

Traditionally, LLMs generate tokens one at a time in an autoregressive manner. For example, given a prompt, the model generates three tokens T1, T2, T3, each requiring a separate forward pass. Speculative decoding transforms this process by allowing multiple tokens to be proposed and verified in one forward pass.

Here's how the process works:

1. **Draft Model**: A smaller, more efficient model proposes tokens one by one.
2. **Target Model Verification**: The larger model verifies these tokens in a single forward pass. It confirms correct tokens and corrects any incorrect ones.
3. **Multiple Tokens in One Pass**: Instead of generating one token per pass, this method processes multiple tokens simultaneously, reducing latency.

{{<image 
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/sd-example.png"  
width="80%"
caption=`As shown in the picture above, the draft model proposes five tokens: ["I", "like", "cooking", "and", "traveling"]. These are then forwarded to the target model for parallel verification. In this example, the third token, "cooking" (should be "playing"), was proposed inaccurately. As a result, only the first three tokens, ["I", "like", "playing"], are generated in this step.`
>}}

By using this approach, speculative decoding speeds up token generation, making it an effective method for both small-scale and large-scale language model deployments.

### 1.2. How Speculative Decoding Works in vLLM

In vLLM, speculative decoding is integrated with the system's continuous batching architecture, where different requests are processed together in a single batch, enabling higher throughput. vLLM uses two key components to implement this:

- Draft Runner: This runner is responsible for executing the smaller model to propose candidate tokens.
- Target Runner: The target runner verifies the tokens by running the larger model.

vLLM's system is optimized to handle this process efficiently, allowing speculative decoding to work seamlessly with continuous batching, which increases the overall system performance.

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/sd-in-vllm.png"
caption=`Diagram illustrating how the draft and target runners interact within the vLLM batching system.`
>}}

To implement speculative decoding in vLLM, two crucial components had to be modified:

- **Scheduler**: The scheduler was adjusted to handle multiple token slots within a single forward pass, enabling the simultaneous generation and verification of several tokens.
- **Memory Manager**: The memory manager now handles the KV cache for both the draft and target models, ensuring smooth processing during speculative decoding.

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/vllm-sd-system-archi.png"
width="80%"
caption=`System architecture of speculative decoding in vLLM. `
>}}

### 1.3. Types of Speculative Decoding Supported in vLLM

> **How to Use Speculative Decoding in vLLM**
> 
> - {{<href text="vLLM | Speculative Decoding" url="https://docs.vllm.ai/en/latest/features/spec_decode.html">}}
> - {{<href text="How to Use Speculative Decoding in vLLM" url="https://blog.vllm.ai/2024/10/17/spec-decode.html#how-to-use-speculative-decoding-in-vllm">}}.


#### 1.3.1. Draft Model-Based Speculative Decoding

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/draft-model-based-sd.png"
width="80%"
>}}

This is the most commonly used form of speculative decoding, where a smaller model predicts the next tokens, and a larger model verifies them. A common example would be using a Llama 68M model to predict tokens for a Llama 2 70B model. This approach requires careful selection of the draft model to balance accuracy and overhead.

Choosing the correct draft model is essential for maximizing the efficiency of speculative decoding. The draft model needs to be small enough to avoid creating significant overhead but still accurate enough to provide a meaningful performance boost.

However, **selecting the right draft model** can be challenging. For example, in models like Llama 3, finding a suitable draft model is difficult due to differences in vocabulary size. Speculative decoding requires that the draft and target models **share the same vocabulary**, and in some cases, this can limit the use of speculative decoding. Therefore, in the following sections, we introduce several draft-model free speculative decoding methods.

#### 1.3.2. Prompt Lookup Decoding

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/prompt-lookup-decoding.png"
width="80%"
caption=`An example of prompt lookup decoding. Given the prompt, we build all 2-grams as the lookup key. The values are the three tokens following the lookup key. During generation, we will check if the current 2-gram matches any key. If so, we will propose the following tokens with the value.`
>}}

Otherwise known as n-gram matching, this approach is effective for use cases like summarization and question-answering, where there is a significant overlap between the prompt and the answer. Instead of using a small model to propose tokens, the system speculates based on the information already available in the prompt. This works particularly well when the large model repeats parts of the prompt in its answers.

#### 1.3.3. MEDUSA

- **MEDUSA Heads**

MEDUSA heads are additional decoding heads appended to the last hidden states of the original model.

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/medusa.png"
width="70%"
caption=`Three heads are used to propose tokens for the following three positions. Head 1 is proposing ["is", "\'", "the"] for the first position. Head 2 is proposing ["difficult", "is", "\'"] for the second position. Head 3 is proposing ["not", "difficult", "a"] for the third position. NOTE: All heads take the output of the last transformer block as the input.`
>}}

Specifically, given the original model’s last hidden states $h_t$ at position $t$, we add $K$ decoding heads to $h_t$. The $k$-th head is used to predict the token in the $(t + k + 1)$-th position of the next tokens (the original language model head is used to predict the $(t + 1)$-th position).

$$
\begin{aligned}
p_{t}^{(k)} & =\mathrm{softmax}\left(W_{2}^{(k)}\cdot\left(\mathrm{SiLU}(W_{1}^{(k)}\cdot h_{t})+h_{t}\right)\right), \\
 & \mathrm{where~}W_{2}^{(k)}\in\mathbb{R}^{d\times V},W_{1}^{(k)}\in\mathbb{R}^{d\times d}.
\end{aligned}
$$

Unlike a draft model, MEDUSA heads are trained in conjunction with the original backbone model, which can remain frozen during training (MEDUSA-1) or be trained together (MEDUSA-2).

- **Tree Attention**

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/tree-attn.png"
width="70%"
>}}

The top-2 predictions from the first MEDUSA head and the top-3 from the second result in a total of $2 \times 3 = 6$ candidates. Each of these candidates corresponds to a distinct branch within the tree structure. 

To guarantee that each token only accesses its predecessors, an attention mask is devised that exclusively permits attention flow from the current token back to its antecedent tokens.

- **vLLM Implementation**

1. {{<href text="[ISSUE] Can vLLM support medusa head? #1023" url="https://github.com/vllm-project/vllm/issues/1023">}}  
1. {{<href text="[ISSUE] [Discussion] Will vLLM consider using Speculative Sampling to accelerating LLM decoding? #1171" url="https://github.com/vllm-project/vllm/issues/1171">}}
1. {{<href text="[PR] [Speculative Decoding] Medusa Implementation with Top-1 proposer #4978" url="https://github.com/vllm-project/vllm/pull/4978">}}

Medusa code: {{<href text="vllm/model_executor/models/medusa.py" url="https://github.com/vllm-project/vllm/blob/f90a375/vllm/model_executor/models/medusa.py">}}

{{<details title="Details of MEDUSA Forward">}}
```python {linenos=true}
class ResidualBlock(nn.Module):
    def __init__(self, config: VllmConfig, hidden_size: int,
                 num_layers: int) -> None:
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size,
                      hidden_size,
                      bias=getattr(config, "medusa_fc_bias", False))
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x

class Medusa(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.blocks = nn.ModuleList([
            ResidualBlock(config=config,
                          hidden_size=self.config.hidden_size,
                          num_layers=self.config.num_hidden_layers)
            for _ in range(self.config.num_heads)
        ])


    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        return [block(hidden_states) for block in self.blocks]
```
{{</details>}}

#### 1.3.4. EAGLE

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/eagle-compare.png"
width="80%"
caption=`A comparison of the methods for drafting the fourth and fifth tokens, t4 and t5. t (represented by blue blocks) denotes tokens, and f (orange blocks) signifies the features, with subscripts indicating their positions in the sequence.  The red border indicates the predictions of the draft model. For simplicity, the n in the n-gram for Lookahead, as shown in the figure, has been set to 2.`
>}}

### 1.4. Speculative Decoding Performance Insights: Speedups and Trade-offs

Speculative decoding offers significant performance benefits in **low-QPS (queries per second)** environments. For example, in testing on the ShareGPT dataset, vLLM demonstrated up to a 1.5x speedup in token generation when using draft model-based speculative decoding. Similarly, prompt lookup decoding has shown speedups of up to 2.8x when applied to summarization datasets, such as CNN/DailyMail.

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/sd-performance-low-qps.png"
caption=`Performance comparison showing spec decode delivering up to 1.5x Speedup at QPS=1 Llama3-70B on ShareGPT with 4xH100 using draft model (turboderp/Qwama-0.5B-Instruct) and up to 2.8x Speedup at QPS=1 Llama3-70B on CNN Dailymail with 4xH100 using n-grams.`
>}}

However, in **high-QPS environments**, speculative decoding may introduce performance trade-offs. The extra compute required to propose and verify tokens can sometimes slow down the system when it is already compute-bound, as seen when the number of requests per second increases. In such cases, the overhead of speculative decoding can outweigh its benefits, leading to reduced performance.

{{<image
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/sd-performance-high-qps.png"
caption=`As high QPS, we see 1.4x slowdown Llama3-70B on ShareGPT with 4xH100, 1.8x slowdown Llama3-70B on CNN Dailymail with 4xH100`
>}}


