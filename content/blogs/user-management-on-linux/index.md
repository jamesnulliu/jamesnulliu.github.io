---
title: "User Management on Linux"
date: 2024-07-06T07:04:00+08:00
lastmod: 2024-07-06T07:12:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - user management
categories:
    - system
tags:
    - linux
description: Some useful commands for user management on Linux.
summary: Some useful commands for user management on Linux.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

## 1. Add/Remove a User

Add a new user:

```bash
sudo useradd <username>
```

Remove an existing user:

```bash
sudo userdel <username>
```

## 2. Change the Password of a User

```bash
sudo passwd <username>
```

## 3. Superuser

Grant write permission to `/etc/sudoers`:

```bash
chmode u+w /etc/sudoers
```

There are four ways to make a user a superuser:

1. Add `<username> ALL=(ALL:ALL) ALL` to the end of the file `/etc/sudoers`. This allows the user to execute any command with prefix `sudo` after entering the password.
2. Add `<username> ALL=(ALL:ALL) NOPASSWD: ALL` to the end of the file `/etc/sudoers`. This allows the user to execute any command with prefix `sudo` without entering the password.
3. Add `%<groupname> ALL=(ALL:ALL) ALL` to the end of the file `/etc/sudoers`. This allows all users in the group to execute any command with prefix `sudo` after entering the password.
4. Add `%<groupname> ALL=(ALL:ALL) NOPASSWD: ALL` to the end of the file `/etc/sudoers`. This allows all users in the group to execute any command with prefix `sudo` without entering the password.

Return the file `/etc/sudoers` to read-only mode:

```bash
chmode u-w /etc/sudoers
```

## 4. User Groups

Create a new group:

```bash
sudo groupadd <groupname>
```

Add a user to a group:

```bash
sudo usermod -aG <groupname> <username>
```
