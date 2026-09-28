---
title: Chapter 2 · Strings, Unicode, and Byte Encodings
sidebar_position: 2
---

# Chapter 2: Strings, Unicode, and Byte Encodings

Tokenizers must work for every piece of text a user can submit. That means understanding the difference between characters, Unicode code points, and bytes.

## Code points are not bytes

Unicode assigns a code point to each abstract character. An encoding then represents that code point as bytes. UTF-8 uses one to four bytes per code point, while UTF-16 and UTF-32 use different storage strategies.

```python
text = "café 🚀"
print(list(text))
print(text.encode("utf-8"))
```

The visible emoji is one Unicode character, but its UTF-8 representation occupies four bytes. A tokenizer operating directly on bytes can represent any input without an unknown-character escape hatch.

## Why GPT tokenizers use bytes

Byte-level tokenization starts with the 256 possible byte values. BPE can merge common byte sequences into larger tokens. Rare words, symbols, and scripts remain representable because every input eventually falls back to bytes.

## A useful mental model

```text
Python string → Unicode code points → UTF-8 bytes → learned token IDs
```

Do not confuse a byte with a character, and do not assume one token equals one visible symbol.

## Exercise

Inspect the UTF-8 bytes for an accented word, an emoji, and a Chinese sentence. Count the bytes and explain why byte-level fallback improves coverage.
