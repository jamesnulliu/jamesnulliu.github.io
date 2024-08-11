---
title: "Docker Container with Nvidia GPU Support"
date: 2024-08-08T12:00:00+08:00
lastmod: 2024-08-08T15:15:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - docker
categories:
    - system
    - software
tags:
    - docker
    - cuda
    - nvidia
description: How to create a Docker container with Nvidia GPU support.
summary: How to create a Docker container with Nvidia GPU support.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

> Reference:  
> 1. [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
> 2. [Install Docker Engine on CentOS](https://docs.docker.com/engine/install/centos/)
> 3. [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

## 1. Installation
### 1.1. Uninstall Docker

For Ubuntu/Debian:

```bash
# Uninstall old versions
sudo apt-get remove docker.io docker-doc docker-compose docker-compose-v2 \
    podman-docker containerd runc

# Uninstall docker engine
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
    docker-compose-plugin docker-ce-rootless-extras
```

For CentOS/RHEL:

```bash
# Uninstall old versions
sudo yum remove docker docker-client docker-client-latest docker-common \
    docker-latest docker-latest-logrotate docker-logrotate docker-engine

# Uninstall docker engine
sudo yum remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
    docker-compose-plugin docker-ce-rootless-extras

sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

### 1.2. Install Docker

For Ubuntu/Debian:

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine:
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Enable and start the Docker service:
sudo systemctl enable docker
sudo systemctl start docker
```

For CentOS/RHEL:

```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

sudo yum install docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
    docker-compose-plugin

sudo systemctl enable docker
sudo systemctl start docker
```

### 1.3. Install Nvidia Container Toolkit

For Ubuntu/Debian:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Optionally, configure the repository to use experimental packages:
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
```

For CentOS/RHEL:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Optionally, configure the repository to use experimental packages:
sudo yum-config-manager --enable nvidia-container-toolkit-experimental

sudo yum install -y nvidia-container-toolkit

sudo systemctl restart docker
```

## 2. Create a Container

Choose a base image that supports Nvidia GPU in doker hub of [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/), run the following command to create a container:

```bash
docker run \
    -it  \
    --gpus all \
    --name <container_name> \
    -v $HOME/data:/root/data \
    -p <host_port>:<container_port> \
    --entrypoint /bin/bash \
    --shm-size <shm-size>G \
    <image_name>:<tag>
```