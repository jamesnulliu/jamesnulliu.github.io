---
title: "Close the Buzzer"
date: 2024-07-06T06:41:00+08:00
lastmod: 2024-07-06T06:41:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - buzzer
categories:
    - system
tags:
    - buzzer
    - linux
description: How to close the buzzer on Linux?
summary: How to close the buzzer on Linux?
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

The buzzer is a small speaker that emits a beep sound. It is used to notify the user of system events. However, sometimes the beep sound is annoying. To close the buzzer on Linux system, execute the following command:

```bash
sudo bash -c "echo blacklist pcspkr > /etc/modprobe.d/blacklist-pcspkr.conf"
```
