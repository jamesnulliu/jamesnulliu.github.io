---
title: 'Build FFmpeg with NVENC'
date: 2024-05-02
permalink: /posts/2024/05/build-ffmpeg-with-nvenc
tags:
  - ffmpeg
  - cuda
  - nvenc
---

```bash
export CUDA_HOME=/usr/local/cuda-12.2

sudo apt update

sudo apt install autoconf automake build-essential cmake libass-dev \
    libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev \
    libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    pkg-config texinfo wget yasm zlib1g-dev

git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git

cd nv-codec-headers

sudo make install

cd nv-codec-headers && sudo make install

cd FFmpeg

git clone git@github.com:FFmpeg/FFmpeg.git

./configure --prefix=/usr/local/ffmpeg --enable-nonfree \
    --enable-cuda-nvcc --disable-x86asm --nvcc=$CUDA_HOME/nvcc \
    --enable-gpl --enable-libass --enable-libfreetype --enable-libvorbis \
    --enable-libx265 --enable-cuvid --enable-nvenc --enable-libnpp \
    --extra-cflags=-I$CUDA_HOME/include --extra-ldflags=-L$CUDA_HOME/lib64

ffmpeg -i input.mp4 -c:v hevc_nvenc -preset fast -rc:v vbr_hq -cq:v 19 -b:v 0 -s 1280x720 output.mp4
```