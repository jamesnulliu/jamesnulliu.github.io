---
title: "Archive a Docker Container and Import to a New Image"
date: 2024-08-07T22:00:00+08:00
lastmod: 2024-08-07T22:06:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - docker
categories:
    - system
tags:
    - docker
description: How to archive a Docker container and import to a new image.
summary: How to archive a Docker container and import to a new image.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

Start the container which you want to archive:

```bash
docker start <container_name>
```

Attach to the container:

```bash
docker exec -it <container_name> bash
```

Clean some temporary files and cache to reduce the size of the image.

```bash
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*
rm -rf ~/.cache/*
# ...
```

Start a tmux session to avoid the session being interrupted.

```bash
tmux new -s tar
```

Archive the container:

```bash
cd /
# Use `--exclude` to exclude some directories.
tar -cvf img.tar \
  --exclude=/proc \
  --exclude=/sys \
  --exclude=/tmp \
  --exclude=/var/log \
  --exclude=img.tar \
  /
# After finishing the archiving, exit the tmux session.
exit
```

Leave the container:

```bash
exit
```

Copy the archive file to the host:

```bash
docker cp <container_name>:/img.tar .
```

Import the archive file to a new image:

```bash
docker import img.tar <new_image_name>
cat img.tar | docker import - <new_image_name>:<tag>
```