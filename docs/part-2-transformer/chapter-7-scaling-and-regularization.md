---
title: Chapter 7 · Scaling and Regularization
sidebar_position: 7
---

# Chapter 7: Scaling and Regularization

Once the architecture works, improve it systematically: increase the number of layers, embedding width, heads, and context length while keeping optimization stable.

## Dropout

Dropout randomly removes a fraction of activations during training. It discourages brittle reliance on a single pathway and is disabled during evaluation and generation.

## The language-model head

After the final Transformer block, apply layer normalization and a linear projection from the embedding dimension to the vocabulary size. The resulting logits are converted to probabilities for the next token.

```python
x = self.ln_f(x)
logits = self.lm_head(x)
```

Crop the context to `block_size` during generation; positional embeddings cannot represent positions outside the configured window.

## Exercise

Run a small sweep over learning rate and dropout. Record training and validation loss, then choose a configuration based on both curves rather than training loss alone.
