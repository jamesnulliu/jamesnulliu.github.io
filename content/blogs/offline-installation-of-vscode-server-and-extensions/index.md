---
title: "Offline Installation of vscode-server and Extensions"
date: 2024-08-08T11:34:00+08:00
lastmod: 2024-11-15T00:47:00+08:00
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

### 1.1. VSCode Version and Commit ID

If your vscode binary is in `env:PATH`, you can get the version and commit id by running the following command:

```bash
code --version
```

Or if not, open vscode, click `Help` => `About`, find the version and commit id in the pop-up window:

![fig-1](/imgs/blogs/offline-installation-of-vscode-server-and-extensions/commit-id.png)

### 1.2. If Your VSCode Version is Less than `1.19` (e.g, `1.18.1`)


Download `vscode-server-linux-x64` with the following link and send it to the server:

```bash
# If Linux
wget https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
# Or Windows
curl -O https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
# Send "./stable" from host to "~" on server and rename it to "~/vscode-server.tar.gz"
scp -P <port> ./stable <username>@<server-ip>:~/vscode-server.tar.gz
```

On the Server:

```bash
# Create directory "~/.vscode-server/bin"
mkdir -p ~/.vscode-server/bin 
# Extract "~/vscode-server.tar.gz" to "~/.vscode-server/bin"
cd ~/.vscode-server/bin/ && tar -xzf ~/vscode-server.tar.gz
# Rename the extracted directory to "~/.vscode-server/bin/<commit-id>"
mv ./vscode-server-linux-x64 ./<commit-id>
```

Go back to host and connect to your server again, and everything should be okay.

### 1.3. If Your VSCode Version is Greater than `1.19`

Download `vscode-server-linux-x64` with the following link and send it to the server:

```bash
# If Linux
wget https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
# Or Windows
curl -O https://update.code.visualstudio.com/commit:<commit-id>/server-linux-x64/stable
# Send "./stable" from host to "~" on server and rename it to "~/vscode-server.tar.gz"
scp -P <port> ./stable <username>@<server-ip>:~/vscode-server.tar.gz
```

Download `vscode-cli` with the following link and send it to the server:

```bash
# If Linux
wget https://update.code.visualstudio.com/commit:<commit-id>/cli-alpine-x64/stable
# Or Windows
curl -O https://update.code.visualstudio.com/commit:<commit-id>/cli-alpine-x64/stable
# Send "./stable" from host to "~" on server and rename it to "~/vscode-cli.tar.gz"
scp -P <port> ./stable <username>@<server-ip>:~/vscode-cli.tar.gz
```

On the Server:

```bash
# Create directory "~/.vscode-server/cli/servers/Stable-<commit-id>"
mkdir -p ~/.vscode-server/cli/servers/Stable-<commit-id>
# Extract "~/vscode-cli.tar.gz" to "~/.vscode-server"
cd ~/.vscode-server && tar -xzf ~/vscode-cli.tar.gz
# Rename the extracted binary to "~/.vscode-server/code-<commit-id>"
mv ./code ./code-<commit-id>
# Extract "~/vscode-server.tar.gz" to "~/.vscode-server/cli/servers/Stable-<commit-id>"
cd ~/.vscode-server/cli/servers/Stable-<commit-id> && tar -xzf ~/vscode-server.tar.gz
# Rename ".../vscode-server-linux-x64" to ".../server"
mv ./vscode-server-linux-x64 ./server
```

Go back to host and connect to your server again, and everything should be okay.

## 2. Install extensions

On the host, compress the local extension directory `~/.vscode/extensions` and send it to the server.

```bash
tar -cJf extensions.tar.xz ~/.vscode/extensions
scp -P <port> extensions.tar.xz <username>@<server-ip>:~/
```

On the Server, extract the archive to `~/.vscode-server/extensions/`:

```bash
tar -xf ~/extensions.tar.xz -C ~/.vscode-server/
```

Modify `~/.vscode-server/extensions/extensions.json`, replace all the extention paths to `~/.vscode-server/extensions/`.

Reconnect to the server, and you can use vscode server with all the extensions installed.