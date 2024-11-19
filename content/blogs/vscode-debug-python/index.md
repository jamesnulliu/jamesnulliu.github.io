---
title: "VSCode: Debug Python"
date: 2024-10-09T10:40:00+08:00
lastmod: 2024-10-12T10:45:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - debugging
    - python
    - vscode
categories:
    - programming
tags:
    - vscode
    - python
    - debugging
description: How to configure launch.json in VSCode for debugging Python
summary: This post shows how to configure launch.json in VSCode for debugging Python.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

> Here is my template repository of building a Python project (with Pytorch and cutomized CUDA kernels): [VSC-Pytorch-Project-Template](https://github.com/jamesnulliu/VSC-Python-Project-Template) !

First, add the following code to "./.vscode/launch.json" (create the file if it does not exist):

```json
{
    "version": "0.2.0",
    "configurations": [
        // Other configurations...,
        {
            "name": "DebugPy: Current File",
            "type": "debugpy",
            "request": "launch",
            "console": "integratedTerminal",
            // Whether to jump to external code when debugging
            "justMyCode": true,
            // Path to the Python file to debug; If set to "${file}", it will 
            // use the currently opened file
            "program": "${file}",
            // Arguments to pass to the program
            "args": [
                "<arg1>",
                "<arg2>",
                // ...
            ],
            // Environment variables
            "env": {
                "<YOUR_ENV_VAR>": "<VALUE>"
            },
        },
        // Other configurations...,
    ]
}
```

Next, click on the "Run and Debug" icon on the left sidebar, choose the configuration with the name you specified in "launch.json", then click on the green play button to start debugging.