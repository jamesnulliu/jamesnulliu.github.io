---
title: "Mathematics in Hugo"
date: 2024-10-08T13:39:00+08:00
lastmod: 2024-10-08T14:06:00+08:00
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

Note that the file name is based on your theme configuration. For example, in theme PaperMode (which I use), there is a file "[extend_head.html](https://github.com/adityatelange/hugo-PaperMod/blob/master/layouts/partials/extend_head.html#L3)" indicating that I can create a file named `extend_head.html` in `./layouts/partials` (so-called **global layouts**, without modifying layouts in theme) to extend the head.

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

Test it locally by running `hugo server -D .` . If you see the equation rendered correctly, you can deploy your website to the server.

However, here's one more trap. 

You should always use the newest version of hugo to support all features (in goldmark). With previous versions, the `passthrough` extension is not supported, and consequently, `\\` inside the equation will be rendered as `\` instead of a new line. 

If you are using [cloudflare](https://dash.cloudflare.com) to deploy your website, go to **Workers & Pages** -> Your Porject -> **Settings** -> **Variables and Secrets**; Add a new variable `HUGO_VERSION` with value `0.135.0` at least.

