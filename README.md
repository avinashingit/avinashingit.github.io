# avinashingit.github.io

Personal portfolio and writing site for Avinash Kadimisetty, published with GitHub Pages and Jekyll.

## Publish a new post

1. Add a Markdown file to `_posts`.
2. Name it `YYYY-MM-DD-short-title.md`.
3. Start with this front matter:

```yaml
---
layout: post
title: "Your post title"
description: "A one-sentence summary for the writing page and link previews."
date: 2026-07-29 09:00:00 -0700
tags:
  - machine learning
  - engineering
---
```

Write the post below the closing `---`, commit the file, and push it to the `master` branch. GitHub Pages will publish it automatically.

## Update the portfolio

- Homepage content: `index.html`
- Styling: `assets/css/style.css`
- Social links: `_includes/footer.html`
- Site title and metadata: `_config.yml`
- Portrait and project images: `images/`
