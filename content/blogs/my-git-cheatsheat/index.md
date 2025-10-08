---
title: "My Git Cheatsheet"
date: 2025-08-23T15:00:00-07:00
lastmod: 2025-08-23T20:06:00-07:00 
draft: false
author: ["jamesnulliu"]
keywords: 
    - git
    - github
categories:
    - software
tags:
    - git
description: Some commands I often forget when using git.
summary: Some commands I often forget when using git.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

There are some git commands I often forget. Fuck that. I will write a cheatsheet to remember them.

## 1. Create a Upstream Remote

Here is the case. You have your local repository `LA`, connected to your remote repository `RA`, which is forked from `RB`, the community upstream repository.

But now you want to connect `LA` to `RB` with your local git, so that pulling from `RB` will be possible.

First, check existing remotes:

```bash {linenos=true}
git remote -v
# Expected output:
# origin  https://github.com/YourUsername/RA.git (fetch)
# origin  https://github.com/YourUsername/RA.git (push)
```

Next, add `RB` as a remote, naming it, for example, `upstream`:

```bash {linenos=true}
git remote add upstream <URL_OF_RB>
```

Verify that the new remote has been added:

```bash {linenos=true}
git remote -v
# Expected output:
# origin  https://github.com/YourUsername/RA.git (fetch)
# origin  https://github.com/YourUsername/RA.git (push)
# upstream  https://github.com/Community/RB.git (fetch)
# upstream  https://github.com/Community/RB.git (push)
```

Then, fetch the branches and commits from `upstream`:

```bash {linenos=true}
git fetch upstream
```

Create a local branch `upstream-dev` to track `upstream/dev`:

```bash {linenos=true}
git switch -c upstream-dev upstream/dev
# or
git checkout -b upstream-dev upstream/dev
```

Create the corresponding remote branch in your `RA`:

```bash {linenos=true}
git push -u origin upstream-dev:upstream-dev
```

Next time when you want to pull the changes from `RB/dev` to `RA/upstream-dev`, you can do:

```bash {linenos=true}
git switch upstream-dev
git pull upstream dev
```


## 2. I Forgot to Clone a Branch

This is very disgusting. 

If you clone a repository with `--branch <branchname>`, git will only clone that branch, and you cannot see other branches with `git branch -a`.

Check your remote config:

```bash {linenos=true}
git config --get remote.origin.fetch
# Expected output:
# +refs/heads/main:refs/remotes/origin/main
```

This means that git will only fetch `main` branch from remote `origin`.

To fix this, run:

```bash {linenos=true}
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch origin
```

Then you can see all branches with `git branch -a`.

Switch to your target branch:

```bash {linenos=true}
git switch <branchname>
# or
git checkout <branchname>
```