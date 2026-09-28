---
title: Chapter 3 · Efficient Context Mixing
sidebar_position: 3
---

# Chapter 3: Efficient Context Mixing

The Bigram model ignores all earlier context. A first improvement is to let each position average information from the tokens before it.

## Causal averaging

For a sequence of length `T`, construct a lower-triangular matrix. Row `t` can read positions `0...t`, but never positions after `t`.

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)
out = wei @ x
```

This is a communication mechanism, not yet learned attention: every allowed connection has a fixed weight. Matrix multiplication performs all the weighted sums at once, replacing a slow Python loop.

## The limitation

Every token uses the same averaging pattern. A vowel cannot request a nearby consonant, and a closing bracket cannot selectively retrieve its opening bracket. We need data-dependent weights.

## Exercise

Visualize the triangular matrix and calculate the output for a three-token sequence by hand. What information is unavailable to the first token?
