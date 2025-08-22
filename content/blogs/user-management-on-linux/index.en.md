---
title: "User Management on Linux"
date: 2024-07-06T07:04:00+08:00
lastmod: 2025-08-22T13:22:00-07:00
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

## 1. List, Add and Remove a User

List all users:

```bash {linenos=true}
cat /etc/passwd
```

Add a new user:

```bash {linenos=true}
useradd <username>
```

Remove an existing user:

```bash {linenos=true}
userdel <username>
```

## 2. Change the Password of a User

```bash {linenos=true}
passwd <username>
```

## 3. Superuser

Grant write permission to `/etc/sudoers`:

```bash {linenos=true}
chmode u+w /etc/sudoers
```

There are four ways to make a user a superuser:

1. Add `<username> ALL=(ALL:ALL) ALL` to the end of the file `/etc/sudoers`. This allows the user to execute any command with prefix `sudo` after entering the password.
2. Add `<username> ALL=(ALL:ALL) NOPASSWD: ALL` to the end of the file `/etc/sudoers`. This allows the user to execute any command with prefix `sudo` without entering the password.
3. Add `%<groupname> ALL=(ALL:ALL) ALL` to the end of the file `/etc/sudoers`. This allows all users in the group to execute any command with prefix `sudo` after entering the password.
4. Add `%<groupname> ALL=(ALL:ALL) NOPASSWD: ALL` to the end of the file `/etc/sudoers`. This allows all users in the group to execute any command with prefix `sudo` without entering the password.

Return the file `/etc/sudoers` to read-only mode:

```bash {linenos=true}
chmode u-w /etc/sudoers
```

## 4. User Groups

List all user groups:

```bash {linenos=true}
cat /etc/group
```

List the groups a user is in:

```bash {linenos=true}
groups <username>
```

Create a new group:

```bash {linenos=true}
groupadd <groupname>
```

Add a user to a group (logout and login again to take effect):

```bash {linenos=true}
usermod -aG <groupname> <username>
```

## 5. Onwership and Permission of Files and Directories

To check the owership and the permission of a file or directory:

```bash {linenos=true}
# File:
ls -l <filename>
# Directory:
ls -ld <dirname>
# List all files including the hidden ones
ls -la
```

Output example:

```bash {linenos=true}
# Permision|*|owner|group|bytes|   date    |file/dirname
drwxr-xr-x  2 james james 4096  Dec 2 11:02 example-dir/
# *: Number of subdirectories.
#    If file, usually starts at 1; Numbers higher than 1 indicate how many hard 
#    links point to this file.
#    If directory, the minimum value is 2 ("." and "..").
```

To break down `drwxr-xr-x`: 

```txt {linenos=true}
d | rwx | r-x | r-x
↓   ↓     ↓     ↓
|   |     |     └── Others permissions (last 3 chars), 101=5
|   |     └──────── Group permissions (middle 3), 101=5
|   └────────────── Owner permissions (first 3), 111=7
└────────────────── File type, d = directory; - = regular file; l = symbolic 
                    link; b = block device; c = character device
```


To change the ownership:

```bash {linenos=true}
chown [-R] <user>:<group> <filename/dirname>
chown [-R] :<group> <filename/dirname>
```

To change the permission using numeric mode:

```bash {linenos=true}
chmod [-R] 764 <filename/dirname>
```

Where:
- `7=0b100+0b010+0b001`, owner can Read Write Execute.
- `6=0b100+0b010+0b000`, group can Read Write.
- `4=0b100+0b000+0b000`, other can Read.

To change the permission using symbolic mode:

```bash {linenos=true}
chmod +r foldername       # Add read for everyone
chmod a+r foldername      # Add read for everyone
chmod u+r foldername      # Add read for owner only
chmod g+r foldername      # Add read for group only
chmod o+r foldername      # Add read for others only
chmod a-rwx file          # Remove all permissions from all
# ...
```

## 6. Shared Directory

To create a shared directory for all users in the same group being able to create, modify, execute, and delete files:

{{<details title="Click to see file: create-shared-dir">}}

```bash {linenos=true}
#!/bin/bash

set -e

# =============================================================================
# Script: create-shared-dir
# Description: Configures one or more directories to be shared with a specific
#              group.
#
# It performs the following actions on each target directory:
#   1. Creates the directory if it doesn't exist.
#   2. Recursively sets the group ownership.
#   3. Sets permissions (group: rwx, others: none).
#   4. Sets the 'setgid' bit on subdirectories to enforce group inheritance.
#   5. Uses Access Control Lists (ACLs) to enforce permissions for new items.
#
# Usage:
#   sudo create-shared-dir <shared_group> <dir1> [<dir2> ...]
#
# Example:
#   sudo create-shared-dir developers /srv/data /home/shared/project
#
# =============================================================================

# --- Argument Parsing ---
SHARED_GROUP="$1"
shift # Shift arguments so $@ contains only the directories
TARGET_DIRS=("$@")

# --- Input Validation ---
if [ ${#TARGET_DIRS[@]} -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]
then
    echo "Usage: $0 <shared_group> <dir1> [<dir2> ...]"
    echo "Example: $0 developers /var/www/project_a"
    exit 1
fi

# --- Pre-flight Checks ---
# 1. Check for root privileges.
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root (or with sudo)." >&2
    exit 1
fi

# 2. Check if the specified group exists.
if ! getent group "$SHARED_GROUP" > /dev/null; then
    echo "Error: Group '$SHARED_GROUP' does not exist." >&2
    exit 1
fi

# --- Main Logic ---
echo "Configuring shared directories for group '$SHARED_GROUP'..."

for target_dir in "${TARGET_DIRS[@]}"; do
    echo "--> Processing: $target_dir"
    mkdir -p "$target_dir"
    chgrp -R "$SHARED_GROUP" "$target_dir"
    chmod -R g=rwX,o-rwx "$target_dir"
    find "$target_dir" -type d -exec chmod g+s {} +
    setfacl -R -m "g:$SHARED_GROUP:rwX" "$target_dir"
    setfacl -R -d -m "g:$SHARED_GROUP:rwX" "$target_dir"
done

echo "Configuration complete."
exit 0
```

{{</details>}}

You may put the file to "/usr/local/bin/create-shared-dir", and then change its mode with command:

```bash {{linenos=true}}
chmod +x /usr/local/bin/create-shared-dir
```

Then you can run the script with the desired parameters, for example:

```bash {{linenos=true}}
sudo create-shared-dir developers /srv/data /home/shared/project
```

## Related Blogs

- [Environment Varialble Management on Linux](/blogs/environment-variable-management-on-linux)