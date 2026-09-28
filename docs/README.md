# Writing chapters

Each chapter is one Markdown file. Add a new file under the relevant part directory, then add it to the navigation configuration when the site renderer is connected.

Recommended path pattern:

```text
docs/
  part-1-tokenization/
    chapter-1-llm-fundamentals.md
  part-2-transformer/
    chapter-1.md
```

Use the front matter at the top of each file for the title, part, and chapter order. Write the body with normal Markdown. Code fences, tables, blockquotes, and LaTeX math are supported by the planned MDX renderer.
