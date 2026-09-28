---
title: Chapter 4 · Self-Attention
sidebar_position: 4
---

# Chapter 4: Self-Attention

Self-attention lets every token decide which earlier tokens are relevant. Each position produces a **query** (what am I looking for), a **key** (what do I contain), and a **value** (what information should I send).

## Scaled dot-product attention

For queries `Q`, keys `K`, and values `V`, the core operation is:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The lower-triangular mask sets future affinities to negative infinity before softmax. The scale by `√d_k` keeps logits in a useful range at initialization.

```python
k = key(x); q = query(x); v = value(x)
wei = q @ k.transpose(-2, -1) * head_size ** -0.5
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v
```

Attention is communication over a directed graph. Self-attention means Q, K, and V come from the same sequence; cross-attention reads K and V from a separate sequence.

## Exercise

Remove the causal mask and observe the information leak. Then restore it and inspect the attention weights for a single position.
