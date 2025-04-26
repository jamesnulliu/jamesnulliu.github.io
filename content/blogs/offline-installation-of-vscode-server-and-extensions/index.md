---
title: "Offline Installation of vscode-server and Extensions"
date: 2024-08-08T11:34:00+08:00
lastmod: 2025-04-26T18:38:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - vscode
    - vscode-server
categories:
    - software
tags:
    - vscode
description: How to install VSCode Server offline.
summary: How to install VSCode Server offline.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Introduction

Here is a common scenario: 

1. You have a target linux server without public network access, but you want to connect to it using VSCode Remote SSH. 
2. Or even further, your target server is running several Docker containers, and you want to attach to the containers on VSCode. 
3. You also need compatible extensions to be installed, so that you can actually work with your target server or containers.

What a pain! But don't worry, this article will show several methods to install VSCode Server and extensions offline, which can hopefully save you a lot of time and effort.

## 2. Method 1: Copy from Another Linux Server

> 😎 This is the easiest method! 

1. Connect to another Linux server (or WSL) which has access to the public network with VSCode Remote SSH on your local machine.
2. On the server, you would find the `~/.vscode-server` directory, which contains everything you need for SSH connection and all the extensions you have installed.
3. Copy the `~/.vscode-server` directory to your target server.
4. If you want to attach to a container on the server, copy the `~/.vscode-remote` directory to the container; For example: 
   ```bash {linenos=true}
   docker cp ~/.vscode-remote <container_id>:/root/.vscode-remote
   ```
5. Now you can connect to the target server from your local machine using VSCode Remote SSH, and you can also attach to the container after connecting to the server.

⚠️ Note that each time you update your local VSCode, you need to first connect to another linux server and then repeat the above steps to copy the `~/.vscode-server` directory to the target server or container.

## 3. Method 2: Install Manually

> 1. 😵‍💫 This is a relatively complex method, recommended only if you cannot use [Method 1](#2-method-1-copy-from-another-linux-server)! 
> 2. Moreover, this method does not support installing extentions in your target server or container. 
> 3. To use extentions, you will have to copy the `~/.vscode-server/extenstions` directory on another server to the target machine manually and then modify `~/.vscode-server/extensions/extensions.json`, replacing all the extention paths to a correct path based on your environment.

### 3.1. VSCode Version and Commit-ID

If your vscode binary is in `env:PATH`, you can get the version and commit-id by running the following command:

```bash {linenos=true}
code --version
```

Or if not, open vscode, click `Help` => `About`. Find the version and commit-id in the pop-up window:

{{<image
src="/imgs/blogs/offline-installation-of-vscode-server-and-extensions/commit-id.png"
width="90%"
caption=`Click Help => Click About => Find the version and commit-id in the pop-up window.`
>}}

### 3.2. Case A: If Your VSCode Version is Less than `1.19` (e.g, `1.18.1`)

Download `vscode-server-linux-x64` with the following link and send it to the target server:

```bash {linenos=true}
# 1. Download the vscode-server-linux-x64 with the commit-id
#    If your local machine is Linux:
wget https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
#    Or if your local machine is Windows:
curl -O https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable

# 2. Send "./stable" from host to "~" on server with scp and rename it to 
#    "~/vscode-server-linux-x64.tar.gz"
scp -P <port> ./stable <username>@<server-ip>:~/vscode-server-linux-x64.tar.gz
```

Now login to the the server with SSH on your local terminal:

```bash {linenos=true}
# 3. Create directory "~/.vscode-server/bin"
mkdir -p ~/.vscode-server/bin 
# 4. Extract "~/vscode-server-linux-x64.tar.gz" to "~/.vscode-server/bin"
tar -xf ~/vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin
# 5. Rename the extracted directory "vscode-server" to "<commit-id>"
mv ~/.vscode-server/bin/vscode-server ~/.vscode-server/bin/<commit-id>
# 6. Optional: Copy the ".vscode-server" directory to target container
docker cp ~/.vscode-server <container_id>:/root/.vscode-server
```

Finally, go back to your local machine and connect to your server with VSCode Remote SSH, and everything should be okay.

### 3.2. Case B: If Your VSCode Version is Greater than `1.19`

Download `vscode-cli` with the following link and send it to the target server:

```bash {linenos=true}
# 1. Download the vscode-cli with the commit-id
#    If your local machine is Linux:
wget https://update.code.visualstudio.com/commit:<commit-id>/cli-alpine-x64/stable
#    Or if your local machine is Windows:
curl -O https://update.code.visualstudio.com/commit:<commit-id>/cli-alpine-x64/stable

# 2. Send "./stable" from host to "~" on server and rename it to "~/vscode-cli.tar.gz"
scp -P <port> ./stable <username>@<server-ip>:~/vscode-cli.tar.gz
```

Now login to the the server with SSH on your local terminal:

```bash {linenos=true}
# 3. Create directory "~/.vscode-server/cli/servers/Stable-<commit-id>"
mkdir -p ~/.vscode-server/cli/servers/Stable-<commit-id>
# 4. Extract "~/vscode-cli.tar.gz" to "~/.vscode-server"
tar -xzf ~/vscode-cli.tar.gz -C ~/.vscode-server
# 5. Rename the extracted binary to "~/.vscode-server/code-<commit-id>"
mv ~/.vscode-server/code ~/.vscode-server/code-<commit-id>
# 6. Optional: Copy the ".vscode-server" directory to target container
docker cp ~/.vscode-server <container_id>:/root/.vscode-server
```

Finally, go back to your local machine and connect to your server with VSCode Remote SSH, and everything should be okay.