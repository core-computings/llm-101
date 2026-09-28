---
title: Chapter 2 · The Bigram Baseline
sidebar_position: 2
---

# Chapter 2: The Bigram Baseline

Before adding attention, establish a model that only asks: “given this token, what token tends to follow it?”

## 1. Bigram model

The Bigram model uses a trainable lookup table with shape `(vocab_size, vocab_size)`. Each character ID selects one row: a fixed-length vector with one score (a logit) for every possible next character. In `forward`, lookup returns one such vector for each input position. Softmax turns those scores into a probability distribution over the vocabulary; during training, cross-entropy compares the scores with the actual next-character targets.

![Bigram model mechanism: a character ID selects its trainable embedding row, producing vocabulary-sized logits that softmax converts to next-character probabilities.](./assets/chapter-2/bigram-embedding.svg)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)
```

```text
torch.Size([32, 62])
tensor(4.7232, grad_fn=<NllLossBackward0>)
```
