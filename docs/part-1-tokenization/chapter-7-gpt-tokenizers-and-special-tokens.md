---
title: Chapter 7 · GPT Tokenizers and Special Tokens
sidebar_position: 7
---

# Chapter 7: GPT Tokenizers and Special Tokens

OpenAI's `tiktoken` packages the vocabulary, merge ranks, regex rules, and special-token policy needed for fast inference. GPT-2 and GPT-4-style tokenizers share the BPE idea but differ in details such as regex behavior and whitespace handling.

## Special tokens

Tokens such as an end-of-text marker are control symbols. They are not ordinary text fragments and should be handled deliberately: either allow a known special token or treat its spelling as normal user text.

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello world")
print(ids, enc.decode(ids))
```

Never assume that token IDs are interchangeable across model families. A token ID is meaningful only with its vocabulary and merge table.

## Exercise

Compare the tokenization of the same prompt with two encodings. Record differences in spaces, numbers, punctuation, and special-token handling.
