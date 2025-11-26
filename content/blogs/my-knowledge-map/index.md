---
title: "My Knowledge Map"
date: 2025-11-24T16:00:00-07:00
lastmod: 2025-11-24T16:06:00-07:00 
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

## 4. Concepts

### 4.1. NLP

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
