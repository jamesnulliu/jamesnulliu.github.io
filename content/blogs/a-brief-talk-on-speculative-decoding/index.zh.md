---
title: "浅谈投机推理"
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
description: 大型语言模型中的投机推理简要介绍.
summary: 大型语言模型中的投机推理简要介绍.
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. 投机推理 (Speculative Decoding)

### 1.1. 简介

传统的 LLM 以自回归方式逐个生成 token. 例如, 给定一个提示 (prompt), 模型需要分别进行三次前向传播来生成三个 token T1, T2, T3. 推测解码 (Speculative Decoding) 通过允许一次性提议并验证多个 token，改变了这一生成过程.

其核心流程如下: 

1. **草稿模型提议**: 通过一个更轻量高效的模型逐个提议候选 token
2. **目标模型验证**: 将候选序列提交给大模型进行单次前向传播验证. 大模型会确认正确 token 并纠正错误提议
3. **单次处理多 token**: 与传统 "一次一 token" 模式不同, 该方法能并行处理多个 token, 显著降低生成延迟

By using this approach, speculative decoding speeds up token generation, making it an effective method for both small-scale and large-scale language model deployments.

{{<image 
src="/imgs/blogs/a-brief-talk-on-speculative-decoding/sd-example.png"  
width="80%"
caption=`如图所示, 草稿模型提议了五个token: ["I", "like", "cooking", "and", "traveling"]. 目标模型通过单次前向传播进行并行验证. 本例中第三个 token "cooking" (正确应为 "playing") 提议错误, 因此最终接受前三个有效 token["I", "like", "playing"]`
>}}

通过这种 "先推测后验证" 的机制, 推测解码实现了生成速度的飞跃. 该方法兼具通用性和高效性, 适用于不同规模的模型部署场景.