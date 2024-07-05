---
title: "Install Nvidia Driver and CUDA Toolkit on Rocky 9"
date: 2024-07-06T00:00:00+08:00
lastmod: 2024-07-06T06:48:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - nvidia-driver
    - cuda
categories:
    - software
    - system
tags:
    - cuda
    - driver
    - rocky
description: How to install Nvidia driver and CUDA toolkit on Rocky 9.
summary: How to install Nvidia driver and CUDA toolkit on Rocky 9.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
draft: true
---

To stop and disable gdm service (which is the default display manager):

```bash
sudo systemctl disable gdm
sudo systemctl stop gdm
```

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

Download the installation **LOCAL RUN FILE** of **THE LATEST** CUDA Toolkit (>=12.5) from [NVIDIA official website](https://developer.nvidia.com/cuda-downloads) and install it (with driver).

> 💬 There is a bug of the compatibility of the new linux kernel and previous cuda derviers (less than 555). You could install other versions of cuda toolkit but keep the latest driver.

To enable and start gdm service:

```bash
sudo systemctl enable gdm
sudo systemctl start gdm
```