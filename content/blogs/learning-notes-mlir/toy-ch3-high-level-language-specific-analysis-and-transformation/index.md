---
title: "Toy Ch3 | High-Level Language Specific Analysis and Transformation"
date: 2024-07-31T11:11:11+08:00
lastmod: 2024-07-31T15:11:11+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
categories:
    - notes
tags:
    - mlir
description: My learning notes of MLIR.
summary: My learning notes of MLIR.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

> Reference: https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/  

**Note**: Check [Setup the Environment of MLIR](../setup-the-environment-of-mlir/) for the environment setup.

## 1. Run Example

Emit MLIR:

```bash
toyc-ch3 $MLIR_HOME/test/Examples/Toy/Ch3/codegen.toy -emit=mlir -opt
```

**Key Points:**
- Pattern-match and rewrite;
- Declarative, rule-based pattern-match and rewrite (DRR);