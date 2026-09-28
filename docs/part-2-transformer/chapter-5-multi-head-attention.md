---
title: Chapter 5 · Multi-Head Attention
sidebar_position: 5
---

# Chapter 5: Multi-Head Attention

One attention head learns one communication pattern. Multi-head attention runs several smaller heads in parallel, allowing different heads to specialize in different relationships.

```python
heads = [Head(head_size) for _ in range(num_heads)]
x = torch.cat([h(x) for h in heads], dim=-1)
x = self.proj(x)
```

The projection mixes the concatenated head outputs back into the model's embedding size. The computation remains parallel over the batch, sequence positions, and heads.

## Position matters

Attention alone operates on a set of vectors and has no inherent notion of order. Add token embeddings to learned positional embeddings so the same token at two positions can receive different representations.

$$
x_{t}=E_{\text{token}}(\text{token}_t)+E_{\text{position}}(t)
$$

## Exercise

Change the number of heads while keeping the total embedding size fixed. Explain why each head receives a smaller `head_size` and why concatenation restores the original width.
