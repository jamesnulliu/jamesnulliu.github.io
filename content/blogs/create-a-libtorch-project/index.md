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

These days I am reading *Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition*, and created a [project](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors) to store my notes and codes as I learn. 

One of the most important parts in the book is writing cuda kernels, so I decided to build all kernels into a shared library and test those implementations both in C++ and Python. I generated my project using [this](https://github.com/jamesnulliu/VSC-Python-Project-Template) template specifically tailored for the similar scenario, but still met some problems such as conflicts when linking libtorch and gtest.

So the purpose of this blog is to provide a step by step guide to: 

1. Build a C++ & CUDA library, test it in a C++ gtest executable.
2. Load the library into torch, call the operates in Python.
3. Resolve problems when linking all the libraries.

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

This is a little sophisticated but let's start with something simple.

## 1. Environment

Managing python environment with miniconda is always a good choice. Check [this](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) website for installation guide.

After installation, create a new virtual environment named `PMPP`:

```bash {linenos=true}
conda create -n PMPP python=3.12
conda activate PMPP  # Activate this environment
```

Install pytorch following the steps on [this](https://pytorch.org/get-started/locally/#start-locally) website. In my case I installed torch-2.5.1 with cuda 12.4 both on Linux and Windows.

> 📓 **NOTE**  
> All the python packages you install can be found under the directory of `$CONDA_PREFIX/lib/python3.12/`.

Of course you also need to install cuda toolkit. In most cases, even if you installed *torch-2.5.1 with cuda 12.4*, using cuda 12.6 or cuda 12.1 can run torch in python without any mistakes. But in some cases you still have to use cuda 12.4 to exactly match the torch you have installed.

You can find all versions of cuda in [this](https://developer.nvidia.com/cuda-toolkit-archive) website. Installing and using multiple versions of cuda is possible by managing the `PATH` and `LD_LIBRARY_PATH` environment variables, and you can do this manually or refering to my method in [this](/blogs/environment-variable-management) blog.

To check your installation, examine the version and the path of `nvcc` using these commands:

``` {linenos=true}
nvcc --version

# bash
which nvcc
# powershell
where.exe nvcc
```