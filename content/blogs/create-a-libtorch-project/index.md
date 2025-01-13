---
title: "Create A LibTorch Project"
date: 2024-12-23T01:00:00+08:00
lastmod: 2025-01-13T15:31:00+08:00
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

These days I am reading *Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition*, and created a [💻project](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors) to store my notes and codes as I learn. 

One of the most important parts in the book is writing **cuda kernels**, so I decided to build all kernels into shared libraries and test those implementations both in C++ and Python. 

I generated my project using [this template](https://github.com/jamesnulliu/VSC-Python-Project-Template) specifically tailored for the similar scenario, but still met some problems such as conflicts when linking libtorch and gtest 🤯.

**So the purpose of this blog is to provide a guide to:** 

1. Build a C++, CUDA and LibTorch library, test it with gtest.
2. Load the library into torch, call the operaters in Python.
3. Resolve problems when linking all the libraries.

> ⚠️**WARNING**  
> You should find some tutorials on how to use cmake and vcpkg before reading this blog.

## 1. Environment

Managing python environment with miniconda is always a good choice. Check [the official website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) for an installation guide.

After installation, create a new virtual environment named `PMPP` (or whatever you like) and activate it:

```bash {linenos=true}
conda create -n PMPP python=3.12
conda activate PMPP  # Activate this environment
```

Install pytorch following the steps on [the official website](https://pytorch.org/get-started/locally/#start-locally). In my case I installed **torch-2.5.1 with cuda 12.4** both on Linux (and Windows).

> 📝**NOTE**  
> All the python packages you installed can be found under the directory of `$CONDA_PREFIX/lib/python3.12/site-packages`.

Of course you also need to install **cuda toolkit** on your system. Usually, even if you installed **torch-2.5.1 with cuda 12.4**, using **cuda 12.6** or **cuda 12.1** can run torch in python without any mistakes. But in some cases, you still have to use **cuda 12.4** to exactly match the torch you chose.

You can find all versions of cuda in [this website](https://developer.nvidia.com/cuda-toolkit-archive). Installing and using multiple versions of cuda is possible by managing the `PATH` and `LD_LIBRARY_PATH` environment variables on linux, and you can do this manually or refering to my methods in [this blog](/blogs/environment-variable-management).

> Or if you are a docker user, just pull the image that contains the cuda version you need. Check [Docker Container with Nvidia GPU Support](/blogs/docker-container-with-nvidia-gpu-support) if you need any help.

## 2. Create a C++, CUDA and LibTorch Project

I put all C++ code in "./csrc/" and build them with cmake. The intermediate files should be generated in "./build/" and that is just about using some command-line arguments, see [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/scripts/build.sh#L39). 

Moreover, vcpkg is used to manage the dependencies of the project. I am not going to teach you how to use vcpkg in this blog, but I will mention some traps I met when using it.

> 😍️ I really enjoy building C++ projects with cmake and vcpkg. Have a try if you haven't used them before.

### 2.1. How to Link against LibTorch

Since you have installed pytorch in [1. Environment](#1-environment), now you actually already have libtorch installed in your virtural environment. Try this command, and you will get the cmake prefix path of pytorch.

```bash
python -c "import torch;print(torch.utils.cmake_prefix_path)"
```

To integrate this into cmake, I created [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/utils/run-python.cmake) and [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/libraries/libtorch.cmake) to find libtorch in the current project and used them [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/CMakeLists.txt#L27).

Now you can link your targets against libtorch simply like what I did [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L19).


### 2.2. CMake and VCPKG Configuration

Currently, I am planning to use the packages listed in [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/vcpkg.json). I load the packages with [these lines](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/CMakeLists.txt#L30-L35) in "./csrc/CMakeLists.txt". Then I link those packages to my targets [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L20-L21) and [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/test/CMakeLists.txt#L13-L14).

However, libtorch is compiled with `_GLIBCXX_USE_CXX11_ABI=0` to use legacy ABI before C++11 (I really hope they change this in future releases), which conflitcs with the packages managed by vcpkg in default. Consequentially, you have to create a custom vcpkg triplet to control the behaviors when vcpkg actually build the packages. The triplet file is [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/vcpkg-triplets/x64-linux-no-cxx11abi.cmake) and is enabled when building the project by [these lines](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/scripts/build.sh#L44-L45).

I also set `CMAKE_CXX_SCAN_FOR_MODULES` to `OFF` on [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/cmake/compilers/cxx-compiler-configs.cmake#L14) because some compile errors occurs showing conflicts between the compiler and the libraries. This is a temporary solution but I am not planning to use modules from C++20 in this project, so just ignoring it.

### 2.3. Write and Register Custom Torch Operators

In order to register a custom torch **operator**, basically what you need to do next is to write a **function** that usually takes several `torch::Tensor` as input and returns a `torch::Tensor` as output, and then register this function to torch.

For example, I implement `pmpp::ops::cpu::launchVecAdd` in [this cpp file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/vecAdd/op.cpp) and `pmpp::ops::cuda::launchVecAdd` in [this cu file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/vecAdd/op.cu) and provide the corresponding torch implentations `pmpp::ops::cpu::vectorAddImpl` and `pmpp::ops::cuda::vectorAddImpl` in [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/vecAdd/torch_impl.cpp). 

> 🤔 I didn't add any of those function declarations in hpp files under "./include" because I don't think they should be exposed to the users of the library. For the testing part, I will get and test the functions using `torch::Dispatcher` which aligns with the operaters invoked in python.

To register these implementations as an operater into pytorch, see [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/torch_bind.cpp#L10), [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/torch_bind.cpp#L19), and [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/ops/torch_bind.cpp#L25), where I:

1. Define a python function `vector_add` with signature: `vector_add(Tensor a, Tensor b) -> Tensor`.
2. Register the CPU implementation of the function.
3. Register the CUDA implementation of the function.

Now `vector_add` is a custom torch operator which can be called in both C++ and Python. All you need to do is to build these codes into a shared library like what I did [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L8) in cmake and wait for users to call it.

### 2.4. Test the Custom Torch Operators in C++

As long as a custom torch operator is registered, normally one or multiple shared libraries will be generated. For C++ users, you should link against libtorch and those shared libraries so that those registered operators can be called. Since I linked libtorch as `PUBLIC` in [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/lib/CMakeLists.txt#L18), the test target will link against libtorch automatically as long as it links against the shared libraries, see [this line](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/test/CMakeLists.txt#L12).

You may be confused about why `-Wl,--no-as-needed` is added before `${PROJECT_NAMESPACE}pmpp-torch-ops`. This is because the shared libraries are not directly used in the test target (an opearter is register in the library but not called directly in the executable), and the linker will not link against them by default. This flag will force the linker to link against the shared libraries even if they are not directly used.

The registered operators can be dispatched in a not-so-intuitional way 🤣 based on the official documentation, see [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/csrc/test/test_ops/vecAdd.cpp#L18-L21), but I think it is reasonable and can be understood with little mental burden.

Now the only thing is to test the operators in C++ using gtest, but this is not the focus of this blog. So let's move on to the next part.


## 3. Create and Package a Python Project

### 3.1. `pyproject.toml` and `setup.py`

In modern python, pyproject.toml is a de-facto standard configuration file for packaging, and in this project, setuptools is used as the build backend because I believe it is the most popular one and is easy to cooperate with cmake. 

Particularly, "[./pyproject.toml](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/pyproject.toml)" and "[./setup.py](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/setup.py)" defines what will happen when you run `pip install .` in the root directory of the project. I created `CMakeExtention` and `CMakeBuild` ([here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/setup.py#L23-L66)) and pass them to `setup` function ([here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/setup.py#L91-L99)) so that the C++ (under "./csrc/") will be built and installed before installing the python package.

You can easily understand what I did by reading the source code of these two files 👀, and there is only one thing I want to mention here.

Based on [2. Create a C++, CUDA and LibTorch Project](#2-create-a-c-cuda-and-libtorch-project), you should find that the generated shared library is under `./build/lib` ending with `.so` on linux or `.dll` on windows. Additionally, I added an install procedure [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/setup.py#L60-L66) which will copy the shared libraries to "./src/pmpp/_torch_ops". "[./src/pmpp](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/tree/cf690614d004aa647aefccb8db3eac83255cb99e/src/pmpp)" is already an existing directory being the root of the actual python package, and "./src/pmpp/_torch_ops" will be created automatically while installing the shared libraries.

The problem is that, when packaging the python project, only a directory containing "\_\_init\_\_.py" will be considered as a package (or module), and I don't want to add this file to "./src/pmpp/_torch_ops" due to my mysophobia 😷. That is why I used `find_namespace_packages` instead of `find_packages` and speficied `package_data` to include the shared libraries [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/setup.py#L100-L102).

### 3.2. Install the Python Package

If you are planning to build your libraries with dependencies listed [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/pyproject.toml#L26-L31) while installing the python project, I really don't suggest install it in an isolated python environment (which is the default behavior of setuptools).All packages listed [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/pyproject.toml#L2) have to be re-installed and in our case you need to at least append `torch` to that list. 

Aternatively, try this command, which will directrly use the torch installed in current conda environment:

```bash
pip install --no-build-isolation -v .
```

### 3.3. Test the Custom Torch Operators in Python

As long as you have the shared libraries built in [2. Create a C++, CUDA and LibTorch Project](#2-create-a-c-cuda-and-libtorch-project), all you need to do is to use `torch.ops.load_library` to load the shared libraries and call the registered operators.

I write this process into "src/pmpp/\_\_init\_\_.py" [here](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/src/pmpp/__init__.py), so the time you import `pmpp` in python, your custom torch operators will be ready to use. See [this file](https://github.com/jamesnulliu/Learning-Programming-Massively-Parallel-Processors/blob/cf690614d004aa647aefccb8db3eac83255cb99e/test/test.py) for an example of testing the operators.
