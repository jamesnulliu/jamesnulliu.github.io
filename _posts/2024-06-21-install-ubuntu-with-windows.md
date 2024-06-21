---
title: 'Install Ubuntu with Windows'
date: 2024-05-02
permalink: /posts/2024/06/21/install-ubuntu-with-windows
tags:
  - system
  - architecture
  - linux
---

## 1. [Optional] Reset Windows

Reseting your current Windows installation is optional but recommended. It will free up disk space and make the installation process easier.

1. Open `Settings` by pressing `Win + I`.
2. Search for `Recovery`.
3. Click on `Reset this PC`.

## 2. Download Ubuntu Desktop

> We will use Ubuntu Desktop 24.04 LTS for this guide.

It is recommanded to download from mirror sites to get the best download speed.

## 3. Create a Bootable USB

Prepare a USB drive with at least 8GB of storage.

Download Rufus from [here](https://rufus.ie/en/).

Follow the instructions (it is pretty straightforward) to create a bootable USB drive.

## 4. Install Ubuntu

### 4.1. Check The Number of Disks on Your Machine

> We suppose ALL your disks are SSDs, and we will simply install everything (system and data) on SSDs. SSDs are way faster than HDDs, which means at least you should install systems on SSDs and use HDDs for storage. Mounting HDDs as data disks is quite easy but not covered in this guide. Leave a comment if you need help with that.

1. Open `Disk Management` by pressing `Win + X` and selecting `Disk Management`.
2. Check the number of disks on your machine.
3. You should also see one or more disks with a capacity of 128GB or more. These are your SSDs.
4. If you see a disk with a capacity of 8GB or more but less than 128GB, this might be your USB drive.