---
title: Chapter 6 · The Transformer Block
sidebar_position: 6
---

# Chapter 6: The Transformer Block

Attention moves information between positions. A feed-forward network transforms each position independently after that communication step.

## Communication plus computation

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The additions are residual connections: each sublayer learns an update to an existing representation instead of rebuilding it from zero. Layer normalization stabilizes the activations before each sublayer (the pre-norm arrangement).

The feed-forward network usually expands the channel dimension, applies a nonlinearity such as GELU, and projects back. Stacking blocks lets information and transformations accumulate over depth.

## Decoder-only structure

GPT uses causal self-attention in every block. Future positions are masked, so the same network can train on all positions in parallel while generation remains autoregressive.

## Exercise

Implement one block with a single head, then replace it with multi-head attention. Track tensor shapes through each residual connection.
