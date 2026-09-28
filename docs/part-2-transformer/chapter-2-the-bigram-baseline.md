---
title: Chapter 2 · The Bigram Baseline
sidebar_position: 2
---

# Chapter 2: The Bigram Baseline

Before adding attention, establish a model that only asks: “given this token, what token tends to follow it?”

## A lookup table as a model

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
```

Each row of the embedding table is a distribution of next-token scores. Cross-entropy compares those scores with the target IDs. Sampling from the final logits generates one token at a time.

## Why keep the weak baseline?

The Bigram model gives us a measurable loss, a generation loop, and a reference point. Every later improvement should lower validation loss or produce more coherent samples instead of adding complexity without evidence.

## Exercise

Train the model for a few hundred steps. Compare greedy decoding with sampling and record how far the generated text remains plausible.
