---
title: "Arithmetic Intensity Estimation of Large Language Models"
date: 2025-03-13T17:38:00+08:00
lastmod: 2025-03-13T18:49:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - speculative decoding
categories:
    - deeplearning
tags:
    - transformer
    - llm
    - vllm
description: "This blog post discusses the arithmetic intensity of large language models and how it affects the performance of these models."
summary: "This blog post discusses the arithmetic intensity of large language models and how it affects the performance of these models."
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Estimating Total FLOPs

We only consider the FLOPs of Transformer layers, excluding the embedding layer and the output layer.

- **Attention**:
  - Each projection for Q, K and V is matmul of input `(B, S, H)` and weight `(H, H)`, yielding `(B, S, H)`:  
    $$
    \text{FLOPs} = 3 \times (2 \times B \times S \times H \times H) = 6 \times B \times S \times H^2
    $$
  - $S = QK^T$, matmul of $Q$ `(B, S, H)` and $K^T$ `(B, H, S)`, yielding `(B, S, S)`:  
    $$
    \text{FLOPs} = 2 \times B \times S \times S \times H = 2 \times B \times S^2 \times H
    $$ 
  - $L = S \cdot V$, matmul of $S$ `(B, S, S)` and $V$ `(B, S, H)`, yielding `(B, S, H)`:  
    $$
    \text{FLOPs} = 2 \times B \times S \times H \times S
    $$
  - $O = L \cdot W_O$, matmul of $L$ `(B, S, H)` and $W_O$ `(H, H)`, yielding `(B, S, H)`:  
    $$
    \text{FLOPs} = 2 \times B \times S \times H^2
    $$
  - Total attention FLOPs per Transformer layer:
    $$
    \text{FLOPs} = 8 \times B \times S \times H^2 + 4 \times B \times S^2 \times H
    $$

- **Feed-Forward Networek**  
  Typically 2 linear layers, one mapping `(B, S, H)` to `(B, S, 4H)` and the other mapping `(B, S, 4H)` to `(B, S, H)`:
  - Total FFN FLOPs per Transformer layer:
    $$
    \begin{align*}
    \text{FLOPs} &= 2 \times B \times S \times H \times (4 \times H) + 2 \times B \times S \times (4 \times H) \times H \\
                 &= 16 \times B \times S \times H^2
    \end{align*}
    $$

- **Total FLOPs: $N$ Layers of Transformer**  
  - Each Transformer layer consists of an attention mechanism and a feed-forward network:
    $$
    \begin{align*}
    \text{FLOPs}_\text{total} &= N \times [(8 \times B \times S \times H^2 + 4 \times B \times S^2 \times H) + (16 \times B \times S \times H^2)] \\
                 &= N \times (24 \times B \times S \times H^2 + 4 \times B \times S^2 \times H)
    \end{align*}
    $$

## 2. Estimating Total Bytes Transfered

In FP16, each parameter or activation element is 2 bytes. 

Data transferred includes **loading model weights** and **handling activations**.

Suppose we have a $Z$ B fp16 model and $N$ Transformer layers, each with input size `(B, S, H)`.

- **Model Weights**  
  A $Z$ B model has $Z \times 10^9$ fp16 parameters, each 2 bytes:

  $$
  \text{Weight}_\text{total} = Z \times 10^9 \times 2 ~ \text{Bytes} = 2 \times Z ~ \text{GBytes}
  $$

  <!-- - Per layer: 12 * hidden_size² parameters (4 * hidden_size² from attention, 8 * hidden_size² from FFN), so 40 * 12 * 5120² * 2 ≈ 25.17 GB, plus embeddings, totaling ~26 GB. -->

  In an optimized GPU inference, weights are typically loaded into high-bandwidth memory (HBM) once and reused, so we assume $2Z$ GB is read once per forward pass.

- **Activations**  
  - For each Transfomer layer, input and output activations are of shape `(B, S, H)`, and each element is 2 bytes in fp16:
      $$
      \text{Activation}_\text{layer} = 2 \times B \times S \times H ~ \text{Bytes}
      $$
  - For 40 layers, activations are computed sequentially. Since each layer’s output becomes the next layer’s input (read once, written once):  
    $$
    \begin{align*}
    \text{Activation}_\text{total} &= 2 \times N \times  \text{Activation}_\text{layer} ~ \text{Bytes} \\
                                   &= 4 \times N \times B \times S \times H ~ \text{Bytes}
    \end{align*}
    $$

- **Total Data Transferred**   
  
$$
\begin{align*}
\text{Bytes}_\text{total} &= \text{Weight}_\text{total} + \text{Activation}_\text{total} \\
                          &= 2 \times Z \times 10^9 + 4 \times N \times B \times S \times H ~ \text{B}
\end{align*}
$$

## 3. Arithmetic Intensity

When prefilling, there is no cached K and V, so the arithmetic intensity is:

$$
\begin{align*}
\text{Arithmetic Intensity} &= \text{FLOPs}_\text{total} / \text{Bytes}_\text{total} \\
                            &= \frac{N \times (24 \times B \times S \times H^2 + 4 \times B \times S^2 \times H)}{2 \times Z \times 10^9 + 4 \times N \times B \times S \times H}
\end{align*}
$$

When decoding, suppose cached sequence length is $S_c$ and the input sequence length is $S_i$ , then the arithmetic intensity is:

$$
\begin{align*}
\text{Arithmetic Intensity} &= \text{FLOPs}_\text{total} / \text{Bytes}_\text{total} \\
                            &= \frac{N \times (24 \times B \times S_i \times H^2 + 4 \times B \times S_i \times S_c \times H)}{2 \times Z \times 10^9 + 4 \times N \times B \times (S_i + S_c) \times H}
\end{align*}
$$

## 4. Roofline Model

{{<image
src="/imgs/blogs/arithmetic-intensity-estimation-of-large-language-models/roofline_model.png"
width="80%"
caption=`Roofline Model. If the arithmetic intensity is on the right side of the machine balance, the performance compute-bound. If it is on the left side, the performance is memory-bound.`
>}}

A100-80GB has the following hardware `specifications:

- **Peak FLOPs** ($\pi$): $312 \times 10^{12}$ FLOPs/s 
- **Memory Bandwidth** ($\beta$): $2039 \times 10^9$ B/s
- **Machine Balance** ($I_{max}$): $312 \times 10^{12} / (2039 \times 10^9) \approx 153$ FLOPs/Byte

Take LLaMA2-13B as an example, at prefilling stage, with `dtype=fp16`, `N=40` and `(B, S, H)=(2, 1024, 5120)`, the arithmetic intensity is:

$$
\begin{align*}
\text{Arithmetic Intensity} &= \text{FLOPs}_\text{total} / \text{Bytes}_\text{total} \\
                            &\approx 1.92 \times 10^3 ~ \text{FLOPs/Byte}
\end{align*}
$$

Since $1.92 \times 10^3 > 153$, the model is compute-bound.

At decoding stage, with `dtype=fp16`, `N=40`, `B=128`, `Si=2`, `Sc=512` and `H=5120`, the arithmetic intensity is about $42.7$ FLOPs/Byte. Since $42.7 < 153$, the model is memory-bound.
