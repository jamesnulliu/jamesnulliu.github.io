---
title: "Render Mathematics in Hugo"
date: 2024-10-08T13:39:00+08:00
lastmod: 2024-10-09T13:03:00+08:00
draft: false
author: ["jamesnulliu"]
keywords: 
    - mathematics
    - latex
    - hugo
categories:
    - web
tags:
    - hugo
description: How to render mathematics in Hugo
summary: This post shows how to render mathematics in Hugo.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

> This blog should be a complete guide to render mathematics in Hugo. However, if you have problems reproducing this blog, note that the **official documentation** is always the best place to start:
>
> 1. [HUGO - Mathematics in Markdown](https://gohugo.io/content-management/mathematics/)
> 2. [Cloudfare - Use a Newer HUGO Verseion](https://developers.cloudflare.com/pages/framework-guides/deploy-a-hugo-site/#use-a-specific-or-newer-hugo-version)

First, add following code to `hugo.yaml`:

```yaml
params:
  # ...
  math: true
  # ...

markup:
  goldmark:
    extensions:
      passthrough:
        delimiters:
          block:
          - - \[
            - \]
          - - $$
            - $$
          inline:
          - - \(
            - \)
          - - $
            - $
        enable: true
```

Second, create a new file `layouts/partials/extend_head.html`:

```html
{{ if .Param "math" }}
  {{ partialCached "math.html" . }}
{{ end }}
```

> Note that name and path of the created file is based on your theme configuration.  
>
> For example, in theme PaperMode (which I use), there is a file "[extend_head.html](https://github.com/adityatelange/hugo-PaperMod/blob/master/layouts/partials/extend_head.html#L3)" indicating that, to extend the head, I can create a file named `extend_head.html` in `./layouts/partials` (so-called **global layouts**, without modifying layouts inside theme, which commonly being a git submodule).  
>
> In other words, if your theme does not support this feature, you may need to copy the `head.html` from the theme to global layouts and modify it, or simply modify the theme directly (but rememenber that modifications in git submodules will not be committed and pushed to the remote repository).

Next, create a new file `layouts/partials/math.html`:

```html
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<script>
  MathJax = {
    tex: {
      displayMath: [['\\[', '\\]'], ['$$', '$$']],  // block
      inlineMath: [['\\(', '\\)'], ['$', '$']]      // inline
    }
  };
</script>
``` 

Now, you can render both block and inline mathematics in your content files. For example, the following code renders the equation $O=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$:

```markdown
$$
O=Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
```

> Test it locally by running `hugo server -D .` to see if the equation rendered correctly.

Finally, there is one more thing to note before deploying your website.

You should always use **the newest version of hugo** to support all features (in goldmark). 

> Why? In previous versions, the `passthrough` extension is not supported, and consequently, `\\` inside the equation will be rendered as `\` instead of a new line. 

Based on the [official doc](https://developers.cloudflare.com/pages/framework-guides/deploy-a-hugo-site/#use-a-specific-or-newer-hugo-version), if you are using **Cloudflare** to deploy your website:

1. Go to **Workers & Pages** -> Your Porject -> **Settings** -> **Variables and Secrets**;
2. Add a new variable `HUGO_VERSION` with value `0.135.0` at least.

This will specify the version of Hugo used by Cloudflare to build your website.

