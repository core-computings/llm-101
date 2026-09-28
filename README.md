# LLM 101

LLM 101 is a practical, visual tutorial for understanding large language models—from tokenization and Transformer fundamentals to inference systems.

## Links

- **Repository:** [github.com/core-computings/llm-101](https://github.com/core-computings/llm-101)
- **Website:** [core-computings.github.io/llm-101](https://core-computings.github.io/llm-101/)

## Contents

Each chapter is a standalone Markdown file under `docs/`:

```text
docs/
├── part-1-tokenization/
│   └── chapter-1-llm-fundamentals.md
└── part-2-transformer/
    └── chapter-1.md
```

The site is built with [Docusaurus](https://docusaurus.io/) and published through GitHub Pages. The navigation is defined in `sidebars.js` and the visual style lives in `src/css/custom.css`.

## Write a chapter

Create or edit a Markdown file in `docs/`, then push your changes:

```bash
npm install
npm run start
```

When changes are pushed to `main`, GitHub Actions builds and publishes the website automatically.
