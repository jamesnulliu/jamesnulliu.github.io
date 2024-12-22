---
title: "Environment Variable Management"
date: 2024-07-31T14:51:00+08:00
lastmod: 2024-07-31T14:51:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - environment-variable
categories:
    - system
    - software
tags:
    - linux
    - environment
description: An easy way to manage environment variables on Linux using load and unload.
summary: An easy way to manage environment variables on Linux using load and unload.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

Open `~/.bashrc` file.

Create 2 functions to load and unload environment variables:

```bash {linenos=true}
env_load() {
    local env_var=$1
    local path=$2
    if [[ ":${!env_var}:" != *":$path:"* ]]; then
        export $env_var="${!env_var}:$path"
    fi
}

env_unload() {
    local env_var=$1
    local path=$2
    local paths_array=(${!env_var//:/ })
    local new_paths=()
    for item in "${paths_array[@]}"; do
        if [[ "$item" != "$path" ]]; then
            new_paths+=("$item")
        fi
    done
    export $env_var=$(IFS=:; echo "${new_paths[*]}")
}
```

Now, you can use `env_load` and `env_unload` to manage environment variables.

For example, to manage CUDA environment, add these lines to `~/.bashrc`:

```bash {linenos=true}
export CUDA_HOME="/usr/local/cuda-12.1"
alias LOAD_CUDA="env_load PATH $CUDA_HOME/bin; env_load LD_LIBRARY_PATH $CUDA_HOME/lib64"
alias UNLOAD_CUDA="env_unload PATH $CUDA_HOME/bin; env_unload LD_LIBRARY_PATH $CUDA_HOME/lib64"
```

