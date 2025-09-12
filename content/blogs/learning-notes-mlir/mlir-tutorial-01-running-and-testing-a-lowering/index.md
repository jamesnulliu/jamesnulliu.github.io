---
title: "MLIR Tutorial 01 | Running and Testing a Lowering"
date: 2024-08-29T11:05:00+08:00
lastmod: 2024-08-29T14:40:00+08:00
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

> Reference: https://www.jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/

**Note**: Check [Setup the Environment of MLIR](../setup-the-environment-of-mlir/) for the environment setup.

## 1. Implementing a Lowering

Create a file "ctlz.mlir":

```mlir
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

Lower the `math.ctlz` operation to the `llvm.ctlz` operation with `mlir-opt`:

```bash
mlir-opt --convert-math-to-funcs=convert-ctlz ./ctlz.mlir
```
