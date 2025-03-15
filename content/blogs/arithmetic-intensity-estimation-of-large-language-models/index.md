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

## LLaMA2-13B

Parameters:

- `hidden_size` = 5120
- `num_layers` = 40
- `fnn_size` ≈ 4 * 5120 = 20480 (The FFN dimension is typically 4× the model's hidden dimension)

