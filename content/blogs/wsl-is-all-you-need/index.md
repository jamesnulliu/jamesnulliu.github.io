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

Coding on Windows feels like a nightmare, but most games can't be played on Linux, and I hate MacOS from many aspects.

After trying many ways, now I firmly believe that WSL (as a Docker launcher 🤣) is the best solution if:

- You are a programmer specifically relying on Unix environment, for example, Deep Learning, CUDA, C++, etc.
- You are a gamer who wants to play games on Windows.
- You want to switch between Linux and Windows with zero cost.

> 💡**NOTE**  
> **Windows Subsystem for Linux (WSL)** is a feature of Windows that allows you to run a Linux environment on your Windows machine, without the need for a separate virtual machine or dual booting. WSL is designed to provide a seamless and productive experience for developers who want to use both Windows and Linux at the same time.

## 2. Installation

> Official Doc: [How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

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

Export a distribution:

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
code <path-to-the-directory>
```

## 5. Install Docker in WSL

Docker Desktop for Windows is a piece of shit. Instead, You can install Docker in WSL as you do in a normal Linux system.

If you want to play with cuda and deep learning in your WSL, see this blog: [Docker Container with Nvidia GPU Support](/blogs/docker-container-with-nvidia-gpu-support) 

If you need a concise mannual for docker images and containers, see this blog: [Something about Docker](/blogs/something-about-docker)

Install extension `ms-vscode-remote.remote-containers` in vscode to open a directory inside a running container with VSCode. All you need to do is:

1. Open a directory in WSL (where you installed docker and running containers as the host) with VSCode following [3. Work with VSCode](#3-work-with-vscode).
2. Press `ctrl` + `shift` + `p`, search for command "Dev Containers: Attach to Running Container...".
3. Choose and click the container you want to open.
4. That's it.