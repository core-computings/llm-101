---
title: Chapter 5 · Encoding and Decoding
sidebar_position: 5
---

# Chapter 5: Encoding and Decoding

After training, a tokenizer exposes two core operations: `encode` maps text to IDs, and `decode` maps IDs back to text.

## Encoding

Encoding begins with UTF-8 bytes and repeatedly applies the learned merges. A practical implementation must apply merges in the same priority order used during training. Otherwise, the same string can produce different IDs.

## Decoding

Decoding looks up each ID in the vocabulary, concatenates the byte strings, and decodes the result as UTF-8.

```python
ids = tokenizer.encode("hello, world")
text = tokenizer.decode(ids)
assert text == "hello, world"
```

Round-trip tests are essential. Test punctuation, repeated spaces, newlines, Unicode, and an empty string—not only ordinary English sentences.

## The model boundary

The Transformer sees only integer IDs. It does not receive the original string or the tokenizer's merge rules. Changing a tokenizer therefore changes the model interface and normally requires retraining or a compatible vocabulary.

## Exercise

Implement `encode` and `decode` for your toy BPE vocabulary. Add tests that assert `decode(encode(text)) == text` for at least five edge cases.
