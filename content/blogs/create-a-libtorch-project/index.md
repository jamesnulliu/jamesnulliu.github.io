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
description: How to create a LibTorch project.
summary: How to create a LibTorch project.
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

So the purpose of this blog is to provide a guide to: 

1. Build a C++ & CUDA library, test it in a C++ gtest executable.
2. Load the library into torch, call the operates in Python.
3. Resolve problems when linking all the libraries.

## 1. Environment

Managing python environment with miniconda is always a good choice. Check [this website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) for an installation guide.

After installation, create a new virtual environment named `PMPP`:

```bash {linenos=true}
conda create -n PMPP python=3.12
conda activate PMPP  # Activate this environment
```

Install pytorch following the steps on [this website](https://pytorch.org/get-started/locally/#start-locally). In my case I installed torch-2.5.1 with cuda 12.4 both on Linux and Windows.

> 📓 **NOTE**  
> All the python packages you install can be found under the directory of `$CONDA_PREFIX/lib/python3.12/`.

Of course you also need to install cuda toolkit. Usually, even if you installed *torch-2.5.1 with cuda 12.4*, using cuda 12.6 or cuda 12.1 can run torch in python without any mistakes. But in some cases, you still have to use cuda 12.4 to exactly match the torch you have installed.

You can find all versions of cuda in [this website](https://developer.nvidia.com/cuda-toolkit-archive). Installing and using multiple versions of cuda is possible by managing the `PATH` and `LD_LIBRARY_PATH` environment variables on linux, and you can do this manually or refering to my methods in [this blog](/blogs/environment-variable-management).

> Or if you are a docker user, just pull the image that contains the cuda version you need.

## 2. Create a C++ and CUDA Project

I put all C++ and CUDA code in `./csrc/` and was going to build them with cmake. The intermediate files should be put in `./build/` and that is just about using some command-line arguments, see [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/scripts/build.sh#L39). 

Moreover, [vcpkg](https://vcpkg.io/en/) is used to manage the dependencies of the project. I am not going to teach you how to use vcpkg in this blog, but I will mention some traps I met when using it.

> 🧛‍♂️ I really enjoy building C++ projects with cmake and vcpkg.

**So what I am indicating here is that you should find some tutorials on how to use cmake and vcpkg before keep reading this blog, or you definitely will get lost.**

### 2.1. How to Link against LibTorch

Since you have installed pytorch in [1. Environment](#1-environment), now you actually have libtorch installed in your virtural environment. Try this command, and you will get the cmake prefix path of pytorch.

```bash
python -c "import torch;print(torch.utils.cmake_prefix_path)"
```

To integrate this into cmake, I created [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/utils/run-python.cmake) and [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/libraries/libtorch.cmake) to find torchlib in the current project and used them [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/CMakeLists.txt#L27).

Now you can link your targets against libtorch simply like what I did [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L19).


### 2.2. VCPKG Configuration

Currently, I am planning to use the packages listed in [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/vcpkg.json). I load the packages with [these lines](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/CMakeLists.txt#L30-L35) in `./csrc/CMakeLists.txt`. Then I link those packages to my targets [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L20-L21) and [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/test/CMakeLists.txt#L13-L14).

However, libtorch is compiled with `_GLIBCXX_USE_CXX11_ABI=0` to use legacy ABI before C++11 (I really hope they change this in later releases), which conflitcs with the packages managed by vcpkg by default. Consequentially, you have to create a custom vcpkg triplet to control the behaviors when vcpkg actually build the packages. The triplet file is [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/vcpkg-triplets/x64-linux-no-cxx11abi.cmake) and is enabled when building the project by [these lines](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/scripts/build.sh#L44-L45).

## 3. Create a Python Project (Package)

### 3.1. `pyproject.toml` and `setup.py`

### 3.2. Install the Package

If you are planning to build your oplib with dependencies listed [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/pyproject.toml#L26-L31) while installing the python project, I really don't suggest install it in an isolated python environment (default behavior of setuptools). This is because you will have to re-install all packages listed [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/pyproject.toml#L2) and in our cases you need to at least append `torch` to that list. 

Aternatively, try this command, which will directrly use the torch installed in current conda environment:

```bash
pip install --no-build-isolation -v .
```