---
title: "02 | Setupt the Environment of MLIR"
date: 2024-07-30T10:45:00+08:00
lastmod: 2024-07-30T10:45:00+08:00
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

Follow the official guide: [Getting Started](https://mlir.llvm.org/getting_started/).

You can write a script to build the source code of MLIR in `/path/to/llvm-project/scripts/build-mlir.sh`:

```bash
#!/bin/bash
mkdir build
cd build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build . --target check-mlir -j $(nproc)
```

Then, you can run the script to build the MLIR:

```bash
chmod +x /path/to/llvm-project/scripts/build-mlir.sh

/path/to/llvm-project/scripts/build-mlir.sh
```