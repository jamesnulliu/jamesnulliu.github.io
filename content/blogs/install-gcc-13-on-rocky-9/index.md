---
title: "Install GCC-13 on Rocky 9"
date: 2024-07-06T00:00:00+08:00
lastmod: 2024-07-06T06:43:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - gcc-13
    - rocky-9
categories:
    - software
tags:
    - gcc
    - c++
    - rocky
description: How to install GCC-13 in Rocky 9.
summary: How to install GCC-13 in Rocky 9.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

Install build essentials.

```bash {linenos=true}
sudo dnf groupinstall "Development Tools"  # gcc-11 is installed by default
```

Enable devel repository and install gcc toolset 13:

```bash {linenos=true}
sudo dnf config-manager --set-enabled devel
sudo dnf update
sudo dnf install gcc-toolset-13
```

To enable gcc-13:

```bash {linenos=true}
scl enable gcc-toolset-13 bash
```

To disable gcc-13, just exit the shell:

```bash {linenos=true}
exit
```