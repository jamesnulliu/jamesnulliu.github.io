---
title: "Create A LibTorch Project"
date: 2024-12-23T01:00:00+08:00
lastmod: 2024-12-23T01:35:00+08:00
draft: true
author: ["jamesnulliu"]
keywords: 
    - pytorch
    - gtest
categories:
    - deeplearning
tags:
    - c++
    - python
    - pytorch
    - gtest
description: Step by step to create a LibTorch project.
summary: Step by step to create a LibTorch project.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

These days I am reading *Programming Massively Parallel Processors: A Hands-on Approach 4th Edition* by David B. Kirk, Wen-mei W. Hwu, and I have created a [project](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors) to store my notes and codes as I learn. 

One of the most important parts in the book is writing cuda kernels, so I decided to build all kernels into a shared library and test those implementations both in C++ and Python. I generated my project using [this](https://github.com/jamesnulliu/VSC-Python-Project-Template) template specifically tailored for the similar scenario, but still met some problems such as conflicts when linking libtorch and gtest.

So the purpose of this blog is to provide a step by step guide to: 

1. Build C++ librares/executables linking against libtorch.
2. Resolve problems when linking libtorch and gtest.
3. Develop custom operators with cuda kernels and register them to torch, calling them later in python. 

The main project structre is shown as follow:

```txt {linenos=true}
Learning-Programming-Massively-Parallel-Processors/
  |- csrc/
  |    |- cmake/
  |    |    |- ...  # Some useful functions for cmake
  |    |- include/pmpp/
  |    |    |- ...  # Header files for the project
  |    |- lib/
  |    |    |- CMakeLists.txt
  |    |    |- ...  # Library source code
  |    |- test/
  |    |    |- CMakeLists.txt
  |    |    |- ...  # Test source code; Use gtest here for testing
  |    |- CMakeLists.txt
  |- scripts/
  |    |- build.sh
  |    |- ...  # Other tool scripts
  |- src/pmpp/
  |    |- __init__.py
  |    |- ...  # Python package source code
  |- test/
  |    |- ...  # Test code for python package
  |- pyproject.toml
  |- setup.py
  |- ...  # Some configurates related to vscode, clangd and github
```