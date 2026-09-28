---
title: Chapter 6 · Regex and Pre-tokenization
sidebar_position: 6
---

# Chapter 6: Regex and Pre-tokenization

Pure byte-level BPE can merge across words, punctuation, and whitespace. Modern GPT-style tokenizers first split text into sensible categories, then run BPE inside each piece.

## Why split first?

Pre-tokenization makes merges more predictable. A rule can keep letters together, isolate punctuation, and handle whitespace consistently. The exact pattern is part of the tokenizer specification, not a cosmetic detail.

```python
import re

chunks = re.findall(r"\w+|[^\w\s]+|\s+", "Hello, world!")
print(chunks)  # ['Hello', ',', ' ', 'world', '!']
```

Real GPT patterns are more careful about contractions, digits, and spaces. They also differ between model generations, so copying a regex from one tokenizer into another can change every token boundary.

## Exercise

Extend the example to cover `can't`, decimal numbers, repeated spaces, and a newline. Explain which pieces should be allowed to merge.
