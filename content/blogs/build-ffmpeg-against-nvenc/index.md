---
title: "Build FFmpeg against NVENC"
date: 2024-06-29T00:00:00+08:00
lastmod: 2024-06-29T00:00:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - build
    - ffpmeg
    - nvenc
categories:
    - software
tags:
    - ffmpeg
    - nvenc
    - cuda
description: How to build FFmpeg against NVENC.
summary: How to build FFmpeg against NVENC.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

FFmpeg is a powerful tool for video processing. It supports a wide range of codecs and formats. In this post, I will show you how to build FFmpeg with NVENC support.

```bash {linenos=true}
# Install cuda 12.2 first

sudo apt update

sudo apt install autoconf automake build-essential cmake libass-dev \
    libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev \
    libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    pkg-config texinfo wget yasm zlib1g-dev

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git

cd nv-codec-headers && sudo make install

cd ..

git clone git@github.com:FFmpeg/FFmpeg.git

cd FFmpeg

./configure --prefix=/usr/local/ffmpeg --enable-nonfree \
    --enable-cuda-nvcc --disable-x86asm --nvcc=$CUDA_HOME/nvcc \
    --enable-gpl --enable-libass --enable-libfreetype --enable-libvorbis \
    --enable-libx265 --enable-cuvid --enable-nvenc --enable-libnpp \
    --extra-cflags=-I$CUDA_HOME/include --extra-ldflags=-L$CUDA_HOME/lib64

# Example: Following code encode a video with nvenc
ffmpeg -i input.mp4 -c:v hevc_nvenc -preset fast -rc:v vbr_hq \
    -cq:v 19 -b:v 0 -s 1280x720 output.mp4
```
