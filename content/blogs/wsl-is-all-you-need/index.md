---
title: "WSL is All You Need"
date: 2025-01-10T13:25:00+08:00
lastmod: 2025-08-22T14:54:00-07:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - wsl
    - programming
    - game
categories:
    - system
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

```powershell {linenos=true}
wsl --list --online
```

Install a specific WSL distribution:

```powershell {linenos=true}
wsl --install -d <distro-name>
```

Run a WSL distribution:

```powershell {linenos=true}
wsl -d <distro-name>
```

## 3. Change the Storage Location of a Distribution

List installed WSL distributions:

```powershell {linenos=true}
wsl -l -v
```

Shutdown WSL service:

```powershell {linenos=true}
wsl --shutdown
```

Export a WSL distribution:

```powershell {linenos=true}
wsl --export <distro-name> <path-to-exported-tar>.tar
```

Unregister the target WSL distribution:

```powershell {linenos=true}
wslconfig /u <distro-name>
```

Import a WSL distribution from tar and specify the storage location; Note that you can specify any new distro name here:

```powershell {linenos=true}
wsl --import <new-distro-name> <path-to-the-new-storage-dir> <path-to-exported-tar>.tar
```

After importing, the default user of `<new-distro-name>` would become `root`. You can set the default user for a WSL distribution by modifying the `/etc/wsl.conf` file. See {{<href text="7. Default User" url="#7-default-user" blank="false">}}.

## 4. Work with VSCode

Run a WSL distribution:

```powershell {linenos=true}
wsl -d <distro-name>
```

Open a directory in WSL with VSCode:

```powershell {linenos=true}
code <path-to-a-directory-in-wsl>
```

## 5. Install Docker in WSL

~I don't like Docker Desktop for Windows~. Instead, you can install Docker in WSL as you do in a normal Linux system.

If you want to play with cuda and deep learning in your WSL, see this blog: {{<href text="Docker Container with Nvidia GPU Support" url="/blogs/docker-container-with-nvidia-gpu-support">}} 

If you need a concise mannual for docker images and containers, see this blog: {{<href text="Something about Docker" url="/blogs/something-about-docker">}}

To open a directory inside a running container with VSCode, install extension `ms-vscode-remote.remote-containers`, and:

1. Open a directory in a WSL (where you installed docker and ran containers) with VSCode following {{<href text="4. Work with VSCode" url="#4-work-with-vscode" blank="false">}}.
2. Press `ctrl` + `shift` + `p`, search for command "Dev Containers: Attach to Running Container...".
3. Choose and click the container you want to open.
4. That's it.

## 6. Proxy 

Somethimes you may need to set the proxy for your subsystem.

From my experience, the easiest way is to turn on `System Proxy` and `TUN Mode` in clash or v2ray on windows, and your WSL and the running containers will automatically use the proxy.

## 7. Default User

You can set the default user for a WSL distribution by modifying the `/etc/wsl.conf` file.

First, login to a WSL distribution:

```powershell {linenos=true}
wsl -d <distro-name>
```

Then, create or modify the `/etc/wsl.conf` file:

```bash {linenos=true}
sudo vim /etc/wsl.conf
```

Add the following lines to the file:

```ini {linenos=true}
[user]
default=<your-username>
```

## 8. Hostname

You can set the hostname for a WSL distribution by modifying the `/etc/hostname` file.

First, login to a WSL distribution:

```powershell {linenos=true}
wsl -d <distro-name>
```

Then, create or modify the `/etc/wsl.conf` file:

```bash
sudo vim /etc/wsl.conf
```
Add the following lines to the file:

```ini
[network]
hostname = <your-hostname>
```

## 9. Compress the WSL File System

> Virtual disk files will grow automatically as you add files, but they will not shrink automatically when you delete files. This can lead to a large virtual disk file that takes up a lot of space on your hard drive.

First, search for a `.vdhx` file in your computer, which is the virtual disk file for your WSL distribution. The default location is "C:/Users/\<your-username\>/AppData/Local/wsl/\<hash-value\>/ext4.vhdx".

Then, run the following command in PowerShell:

```powershell {linenos=true}
wsl --shutdown
```

After that, run `diskpart`:

```powershell {linenos=true}
diskpart
```

In the pop-up window, run the following commands:

```powershell {linenos=true}
select vdisk file="<path-to-your-vhdx-file>"
compact vdisk
detach vdisk
```
