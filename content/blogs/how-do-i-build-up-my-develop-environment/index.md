---
title: "How do I Build up My Develop Environment?"
date: 2024-07-06T00:00:00+08:00
lastmod: 2024-07-06T00:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - environment
    - develop
categories:
    - software
tags:
    - cuda
    - c/c++
    - python
description: How do I build up my develop environment?
summary: How do I build up my develop environment?
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Rocky Linux 9

Rocky Linux 9 is a free and open-source enterprise Linux distribution. It is a 1:1 binary compatible fork of RHEL 9.

### 1.1. Install Rocky Linux 9

I choose to install `workstation` version of Rocky Linux 9.

### 1.2. Install Essential Tools

Install essential tools:

```bash
sudo dnf groupinstall "Development Tools" cmake
```

Enable devel repository and install gcc toolset 13:

```bash
sudo dnf install dnf-plugins-core
sudo dnf config-manager --set-enabled powertools
sudo dnf update
sudo dnf install gcc-toolset-13
```

> 💬 To enable gcc-13, use `scl enable gcc-toolset-13 bash`; To disable gcc-13, just exit the shell.

### 1.2. Install NVIDIA Driver and CUDA Toolkit

To disable the default nouveau driver:

```bash
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
# Update the kernel initramfs
sudo dracut --force
# Reboot
sudo reboot
```

Install epel-release and dkms:

```bash
sudo dnf install epel-release
sudo dnf install dkms
```

Download **THE LATEST** CUDA Toolkit from [NVIDIA official website](https://developer.nvidia.com/cuda-downloads) and install it (with driver).

> 💬 There is a bug of the compatibility of the new linux kernel and previous cuda derviers (less than 555). You could install other versions of cuda toolkit but keep the latest driver.

# 2. Ubuntu