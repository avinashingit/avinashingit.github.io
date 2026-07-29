# avinashingit.github.io

Personal portfolio and writing site for Avinash Kadimisetty, published with GitHub Pages and Jekyll.

## Publish a new post

1. Copy the `_article-template` folder into `writing`.
2. Rename the copied folder to a short URL-friendly name, such as `building-a-search-system`.
3. Edit its `index.md` file and replace the sample front matter:

```yaml
---
layout: post
article: true
title: "Your post title"
description: "A one-sentence summary for the writing page and link previews."
date: 2026-07-29 09:00:00 -0700
tags:
  - machine learning
  - engineering
---
```

Write the article below the closing `---`. Put images in the article folder's `images` directory and use them in Markdown with a relative path:

```markdown
![A useful description of the image](images/diagram.png)
```

The finished structure should look like this:

```text
writing/
  building-a-search-system/
    index.md
    images/
      diagram.png
```

Commit the new folder and push it to the `master` branch. The article will automatically appear at the top of the Writing page, and its title will link to the rendered article.

## Update the portfolio

- Homepage content: `index.html`
- Styling: `assets/css/style.css`
- Social links: `_includes/footer.html`
- Site title and metadata: `_config.yml`
- Portrait and project images: `images/`
