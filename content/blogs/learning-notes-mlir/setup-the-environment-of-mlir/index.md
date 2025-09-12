---
title: "Setup the Environment of MLIR"
date: 2024-07-30T10:45:00+08:00
lastmod: 2024-08-28T10:45:00+08:00
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

To build MLIR, follow the official guide: [Getting Started](https://mlir.llvm.org/getting_started/).

Set up some environment variables to make our life easier when working with MLIR:

```bash
export LLVM_PROJ_HOME="/path/to/llvm-project"
export MLIR_HOME="$LLVM_PROJ_HOME/mlir"
```

Write a script to help build MLIR from source:

```bash
# @file $LLVM_PROJ_HOME/scripts/build-mlir.sh

cd $LLVM_PROJ_HOME

CC=clang CXX=clang++ \
cmake -S . -B ./build -G Ninja ./llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_ENABLE_LLD=ON

cmake --build . --target check-mlir -j $(nproc)
```

Now we can run the script to build the MLIR easily:

```bash
bash $LLVM_PROJ_HOME/scripts/build-mlir.sh
```

The generated binary files are in `$LLVM_PROJ_HOME/build/bin`. 

It would be more convenient to add this directory to `PATH` when working with MLIR:

```bash
export PATH="$LLVM_PROJ_HOME/build/bin:$PATH"
```