---
title: "WSL is All You Need"
date: 2025-01-10T13:25:00+08:00
lastmod: 2025-01-10T13:25:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - wsl
    - programming
    - game
categories:
    - system
    - software
tags:
    - windows
    - linux
    - wsl
    - environment
description: How do I work with WSL
summary: How do I work with WSL
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Motivation

Coding on Windows feels like a nightmare, but most games do not run on Linux, and I hate MacOS 🤢🤮 from many aspects.

After trying many ways, now I firmly believe that WSL (as a Docker launcher 🤣) is the best solution if:

- You are a programmer specifically relying on Unix environment, for example, Deep Learning, CUDA, C++, etc.
- You are a gamer who wants to play games on Windows.
- You want to switch between Linux and Windows with zero cost.

> 💡**NOTE**  
> **Windows Subsystem for Linux (WSL)** is a feature of Windows that allows you to run a Linux environment on your Windows machine, without the need for a separate virtual machine or dual booting. WSL is designed to provide a seamless and productive experience for developers who want to use both Windows and Linux at the same time.

## 2. Installation

> Official Doc: {{<href text="How to install Linux on Windows with WSL" url="https://learn.microsoft.com/en-us/windows/wsl/install">}}

> ⚠️**WARNING**  
> If you are using scoop on Windows, make sure uninstall it before the first-time installation of a WSL distribution. You can reinstall scoop after the installation.

List all available WSL distributions:

```
wsl --list --online
```

Install a specific WSL distribution:

```
wsl --install -d <distro-name>
```

Run a WSL distribution:

```
wsl -d <distro-name>
```

## 3. Change the Storage Location of a Distribution

List installed WSL distributions:

```
wsl -l -v
```

Shutdown WSL service:

```
wsl --shutdown
```

Export a WSL distribution:

```
wsl --export <distro-name> <path-to-exported-tar>.tar
```

Unregister the target WSL distribution:

```
wslconfig /u <distro-name>
```

Import a WSL distribution from tar and specify the storage location:

```
wsl --import <any-distro-name> <path-to-the-new-storage-dir> <path-to-exported-tar>.tar
```

## 4. Work with VSCode

Run a WSL distribution:

```
wsl -d <distro-name>
```

Open a directory in WSL with VSCode:

```
code <path-to-a-directory-in-wsl>
```

## 5. Install Docker in WSL

Docker Desktop for Windows is a piece of shit. Instead, You can install Docker in WSL as you do in a normal Linux system.

If you want to play with cuda and deep learning in your WSL, see this blog: {{<href text="Docker Container with Nvidia GPU Support" url="/blogs/docker-container-with-nvidia-gpu-support">}} 

If you need a concise mannual for docker images and containers, see this blog: {{<href text="Something about Docker" url="/blogs/something-about-docker">}}

To open a directory inside a running container with VSCode, install extension `ms-vscode-remote.remote-containers`, and:

1. Open a directory in a WSL (where you installed docker and ran containers) with VSCode following {{<href text="3. Work with VSCode" url="#3-work-with-vscode" blank="false">}}.
2. Press `ctrl` + `shift` + `p`, search for command "Dev Containers: Attach to Running Container...".
3. Choose and click the container you want to open.
4. That's it.

## 6. Networking

### 6.1. Proxy

Somethimes you may need to set the proxy for your subsystem.

From my experience, the easiest way is to turn on `System Proxy` and `TUN Mode` in clash or v2ray on windows, and your WSL and the running containers will automatically use the proxy.
