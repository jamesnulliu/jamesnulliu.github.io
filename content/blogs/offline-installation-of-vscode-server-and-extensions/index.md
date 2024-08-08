---
title: "Offline Installation of vscode-server and Extensions"
date: 2024-08-08T11:34:00+08:00
lastmod: 2024-08-08T22:06:00+08:00
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

Open vscode, click `Help` => `About`, find the commit id in the pop-up window, e.g., `Commit: b1c0a14de1414fcdaa400695b4db1c0799bc3124`.

Download `vscode-server-linux-x64.tar.gz` with the following link:

```bash
https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
```

Send the downloaded file to the Server:

```bash
scp -P <port> vscode-server-linux-x64.tar.gz <username>@<server-ip>:~/
```

On the Server, create a foler with your commit id, and extract the file into it:

```bash
# Make directory
mkdir -p ~/.vscode-server/bin/<commit-id>/
# Extract the file
tar -xzf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/<commit-id>/ --strip 1
# Create a file to indicate the installation is done
touch ~/.vscode-server/bin/<commit-id>/0 
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