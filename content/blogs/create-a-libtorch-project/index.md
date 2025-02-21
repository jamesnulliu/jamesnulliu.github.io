---
title: "Create A LibTorch Project"
date: 2024-12-23T01:00:00+08:00
lastmod: 2025-02-07T16:13:00+08:00
draft: false
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

## 0. Introduction

These days I am reading *Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition*, and created a {{<href text="project" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors">}} to store my notes as I learn. 

One of the most important parts in the book is writing **cuda kernels**, so I decided to build all kernels into shared libraries and test those implementations both in C++ and Python. 

I generated my project using {{<href text="this template" url="https://github.com/jamesnulliu/VSC-Python-Project-Template">}} specifically tailored for the similar scenario, but still met some problems such as conflicts when linking libtorch and gtest 🤯.

**So the purpose of this blog is to provide a concise guide to:** 

1. Build a C++, CUDA and LibTorch library, test it with gtest.
2. Load the library into torch, call the operaters in Python.
3. Resolve problems when linking all the libraries.

> ⚠️**WARNING**  
> Find some tutorials on how to use cmake and vcpkg before reading this blog.

## 1. Environment and Quick Start

Check {{<href text="README.md" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/tree/c648649/README.md">}} of the project repository.

## 2. Create a C++, CUDA and LibTorch Project

I put all C++ codes in "{{<href text="./csrc/" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/tree/96685ab/csrc">}}" and build them with cmake. The intermediate files should be generated in "./build/" and that is just about using some command-line arguments, see {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/scripts/build.sh#L42">}}. 

Vcpkg is used to manage the dependencies of the project. I am not going to teach you how to use vcpkg in this blog, but I will mention some pitfalls I met when using it.

> 😍️ I really enjoy building C++ projects with cmake and vcpkg. Have a try if you haven't used them before.

### 2.1. How to Link against LibTorch

Since you have installed pytorch in {{<href text="1. Environment" url="#1-environment" blank="false">}}, now you already have libtorch installed in your conda environment. Run this command, and you will get the cmake prefix path of libtorch:

```bash {linenos=true}
python -c "import torch;print(torch.utils.cmake_prefix_path)"
```

To integrate libtorch into cmake, I create {{<href text="this file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/cmake/utils/run-python.cmake">}} and {{<href text="this file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/cmake/libraries/libtorch.cmake">}} to find libtorch in the current project and use them {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/CMakeLists.txt#L27">}}.

Now you can link your targets against libtorch simply like what I do {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/CMakeLists.txt#L19">}}.

> 📝**NOTE**  
> When you link your target against `${TORCH_LIBRARIES}`, cuda libraries are being linked automatically, which means you don't have to find and link cuda using something like I write {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab6/csrc/cmake/libraries/libcuda.cmake">}}

### 2.2. CMake and VCPKG Configuration

Currently, I am planning to use the C/C++ packages listed in {{<href text="this file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/vcpkg.json">}}. I load the packages with {{<href text=`these lines in "./csrc/CMakeLists.txt"` url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/CMakeLists.txt#L30-L36">}} . Then I link those packages to my targets {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/CMakeLists.txt#L20-L21">}} and {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/test/CMakeLists.txt#L11-L13">}}.

> 📝**NOTE**  
> `libtorch < 2.6` is compiled with `_GLIBCXX_USE_CXX11_ABI=0` to use legacy ABI before C++11, which conflicts with the packages managed by vcpkg in default. Consequentially, you have to create a custom vcpkg triplet to control the behaviors when vcpkg actually build the packages. The triplet file is {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/cmake/vcpkg-triplets/x64-linux.cmake">}} and is enabled by {{<href text="these lines" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/scripts/build.sh#L47-L48">}} when building the C++ part.

I also set `CMAKE_CXX_SCAN_FOR_MODULES` to `OFF` on {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/cmake/compilers/cxx-compiler-configs.cmake#L15">}} because some compile errors occurs. This is a temporary solution but I am not planning to use modules from C++20 in this project, so just ignoring it.

### 2.3. Write and Register Custom Torch Operators

In order to register a custom torch **operator**, basically what you need to do next is to write a **function** that usually takes several `torch::Tensor` as input and returns a `torch::Tensor` as output, and then register this function to torch.

For example, I implement `pmpp::ops::cpu::launchVecAdd` in {{<href text="this cpp file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/vecAdd/op.cpp">}} and `pmpp::ops::cuda::launchVecAdd` in {{<href text="this cu file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/vecAdd/op.cu">}} and provide the corresponding torch implentations `pmpp::ops::cpu::vectorAddImpl` and `pmpp::ops::cuda::vectorAddImpl` in {{<href text="this file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/vecAdd/torch_impl.cpp">}}. 

> 🤔 I didn't add any of those function declarations in hpp files under "./include" because I don't think they should be exposed to the users of the library. For the testing part, I will get and test the functions using `torch::Dispatcher` which aligns with the operaters invoked in python.

To register these implementations as an operater into pytorch, see {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/torch_bind.cpp#L10">}}, {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/torch_bind.cpp#L22">}}, and {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/ops/torch_bind.cpp#L32">}}, where I:

1. Define a python function `vector_add` with signature: `vector_add(Tensor a, Tensor b) -> Tensor`.
2. Register the CPU implementation of the function.
3. Register the CUDA implementation of the function.

Now `vector_add` is a custom torch operator which can be called in both C++ and Python. All you need to do is to build these codes into a shared library like what I did {{<href text="here in cmake" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/CMakeLists.txt#L7">}}.

### 2.4. Test the Custom Torch Operators in C++

As long as a custom torch operator is registered, normally one or multiple shared libraries will be generated. For C++ users, you should link your executable target against libtorch and the generated shared libraries so that those registered operators can be called. 

Since I have linked `libPmppTorchOps` against libtorch as `PUBLIC` in {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/lib/CMakeLists.txt#L18">}}, the test target will link against libtorch automatically as long as it links against `libPmppTorchOps`, see {{<href text="this line" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/test/CMakeLists.txt#L10">}}.

> 📝**NOTE**  
> You may be confused about why `-Wl,--no-as-needed` is added before `${PROJECT_NAMESPACE}pmpp-torch-ops`. This is because the shared libraries are not directly used in the test target (an operator is register in the library but not called directly in the executable), and the linker will not link against them by default. This flag will force the linker to link against the shared libraries even if they are not directly used.

The registered operators can be dispatched in a not-so-intuitional way 🤣 based on the official documentation, see {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/csrc/test/OpTest/vecAdd.cpp#L14-L17">}}.

Now the only thing is to test the operators in C++ using gtest, but this is not the focus of this blog. So let's move on to the next part.


## 3. Create and Package a Python Project

### 3.1. `pyproject.toml` and `setup.py`

In modern python, pyproject.toml is a de-facto standard configuration file for packaging, and in this project, setuptools is used as the build backend because I believe it is the most popular one and is easy to cooperate with cmake. 

Particularly, "{{<href text=./pyproject.toml url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/pyproject.toml">}}" and "{{<href text=./setup.py url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/setup.py">}}" defines what will happen when you run `pip install .` in the root directory of the project. I created `CMakeExtention` and `CMakeBuild` ({{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/setup.py#L23-L68">}}) and pass them to `setup` function ({{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/setup.py#L92-L105">}}) so that the C++ library `libPmppTorchOps` (under "./csrc/") will be built and installed before installing the python package.

You can easily understand what I did by reading the source code of these two files, and there is one more thing I want to mention.

Based on [2. Create a C++, CUDA and LibTorch Project](#2-create-a-c-cuda-and-libtorch-project), you should find that the generated shared library is under `./build/lib` ending with `.so` on linux or `.dll` on windows. Additionally, I added an install procedure {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/setup.py#L62-L68">}} which will copy the shared libraries to "./src/pmpp/_torch_ops". 

> Note that "{{<href text="./src/pmpp" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/tree/96685ab/src/pmpp">}}" is already an existing directory being the root of the actual python package, and "./src/pmpp/_torch_ops" will be created automatically while installing the shared libraries.

The problem is, when packaging the python project, only the directory containing "\_\_init\_\_.py" will be considered as a package (or module), and I don't want to add this file to "./src/pmpp/_torch_ops" due to my mysophobia 😷. Therefore, I used `find_namespace_packages` instead of `find_packages` and specified `package_data` to include the shared libraries {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/setup.py#L106-L108">}}.

### 3.2. Install the Python Package

If you are planning to build your libraries with dependencies listed {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/pyproject.toml#L26-L31">}} while installing the python project, I don't really suggest installing it in an isolated python environment (which is the default behavior of setuptools). All packages listed {{<href text="here" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/pyproject.toml#L2">}} have to be re-installed and in our case you need to at least append `torch` to that list. 

Alternatively, try this command, which will directly use the torch installed in current conda environment:

```bash
pip install --no-build-isolation -v .
```

### 3.3. Test the Custom Torch Operators in Python

As long as you have the shared libraries built in {{<href text="2. Create a C++, CUDA and LibTorch Project" url="#2-create-a-c-cuda-and-libtorch-project" blank="false">}}, all you need to do is to use `torch.ops.load_library` to load the shared libraries and call the registered operators.

I write this process into "{{<href text="src/pmpp/__init__.py" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/src/pmpp/__init__.py">}}", so the time you import `pmpp` in python, your custom torch operators will be ready to use. See {{<href text="this file" url="https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/96685ab/test/test.py">}} for an example of testing the operators.
