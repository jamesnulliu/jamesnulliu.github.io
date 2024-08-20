---
title: "Offline Installation of vscode-server and Extensions"
date: 2024-08-08T11:34:00+08:00
lastmod: 2024-08-20T17:28:00+08:00
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

**Host**: Your local machine;

**Server**: The server you want to install VSCode Server on.

**Background**: Host is connected to internet, with VSCode and extensions (Remote - SSH, Remote - Containers, etc.) installed. Server is not connected to internet, but can be accessed by Host through SSH.

## 1. Install vscode-server on the Server

Open vscode, click `Help` => `About`, find the commit id in the pop-up window:

![fig-1](/imgs/blogs/offline-installation-of-vscode-server-and-extensions/commit-id.png)

Download `vscode-server-linux-x64` with the following link and send it to the server:

```bash
wget https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
scp -P <port> ./stable <username>@<server-ip>:~/vscode-server
rm ./stable
```

Download `vscode-cli` with the following link and send it to the server:

```bash
wget https://update.code.visualstudio.com/commit:<commit-id>/cli-alpine-x64/stable
scp -P <port> ./stable <username>@<server-ip>:~/vscode-cli
rm ./stable
```

On the Server:

```bash
mkdir -p ~/.vscode-server/bin/
cd ~/.vscode-server/bin/
tar -xzf ~/vscode-server
mv ./vscode-server-linux-x64 ./<commit-id>

cd ~/.vscode-server
tar -xzf ~/vscode-cli
mv ./code ./code-<commit-id>
```

## 2. Install extensions

On the Host, archive the local extensions in `~/.vscode/extensions/` to be `extensions.zip`.

Send the file to the Server:

```bash
scp -P <port> extensions.zip <username>@<server-ip>:~/
```

On the Server, extract the file to `~/.vscode-server/extensions/`:

```bash
unzip extensions.zip -d ~/.vscode-server/extensions/
```

Modify `~/.vscode-server/extensions/extensions.json`, replace all the extention paths to `~/.vscode-server/extensions/`.

Reconnect to the server, and you can use VSCode Server with all the extensions installed.