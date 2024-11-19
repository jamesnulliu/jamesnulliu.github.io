---
title: "VSCode: Debug C++"
date: 2024-10-09T11:35:00+08:00
lastmod: 2024-10-09T12:41:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - debugging
    - c++
    - vscode
categories:
    - programming
tags:
    - vscode
    - c++
    - debugging
description: How to configure launch.json in VSCode for debugging C++
summary: This post shows how to configure launch.json in VSCode for debugging C++.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

> Here is my template repository of building a CMake-CXX project (with CUDA): [VSC-CMake-CXX-Project-Template](https://github.com/jamesnulliu/VSC-CMake-CXX-Project-Template) !

Suppose that you are managing your project with CMake. To build an executable, first write all your build commands in a bash script. For example, create a new file "./scripts/build.sh":

```bash
build_type=$1
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=$build_type
cmake --build ./build -j $(nproc)
```

Second, add the following code to "./.vscode/tasks.json" (create the file if it does not exist):

```json
{
    "version": "2.0.0",
    "tasks": [
        // Other tasks...,
        {
            // Task name, anything you want, must match the preLaunchTask in 
            // launch.json
            "label": "Build: Debug 01",  
            "type": "shell",
            // Command: bash <script> <args...>
            "command": "bash",
            "args": [
                // Your build script path
                "${workspaceFolder}/scripts/build.sh",
                // Build script arguments
                "Debug"
            ],
            "group": "build"
        },
        // Other tasks...
    ]
}
```


Next, add the following code to "./.vscode/launch.json" (create the file if it does not exist):

```json
{
    "version": "0.2.0",
    "configurations": [
        // Other configurations...,
        {
            // Launch configuration name, anything you want
            "name": "Launch: Debug 01",
            "type": "cppdbg",
            "request": "launch",
            // Path to the generated executable
            "program": "${workspaceFolder}/<path-to-generated-executable>",
            // Arguments to pass to the program
            "args": [
                "arg1",
                "arg2",
                // Other arguments...
            ],
            "externalConsole": false,
            "stopAtEntry": false,
            // Working directory
            "cwd": "${workspaceFolder}",
            // MIMode should be "gdb" for gdb, "lldb" for lldb
            "MIMode": "gdb",
            // Path to the gdb executable
            // Change this to lldb path if you are using lldb
            "miDebuggerPath": "/usr/bin/gdb",
            // Pre-launch task, make sure it matches the task label in 
            // tasks.json
            "preLaunchTask": "Build: Debug 01",
            // Environment variables
            "environment": [
                // This is an example of adding a path to the PATH environment 
                // variable
                {
                    "name": "PATH",
                    "value": "<some-path>:${env:PATH}"
                }
            ],
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb/lldb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
        },
        // Other configurations...,
    ]
}
```

Finally, click on the "Run and Debug" icon on the left sidebar, choose the configuration with the name you specified in "launch.json", then click on the green play button to start debugging.