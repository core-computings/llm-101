---
title: Chapter 1 · Building the Dataset
part: Part 2 · Transformer
sidebar_position: 1
---

# Chapter 1: Building the Dataset

We will build a small GPT-like language model from the ground up. The training text is a single corpus, but the model sees it as many overlapping examples of context and next-token targets.

## From text to integers

For this first implementation, use a character vocabulary. Create a sorted list of unique characters, then map each character to an integer ID.

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)
```

Later, Part 1's subword tokenizer can replace this character encoder without changing the Transformer interface.

## Context windows and batches

Choose a `block_size` and sample contiguous chunks. For every input sequence `x`, the target `y` is the same sequence shifted one position to the left. A batch adds an independent dimension so many examples can be processed in parallel.

> The model must never see a future target while predicting the current position. This causal constraint is the defining data-flow rule of a decoder-only language model.

## Train and validation splits

Hold out a validation slice before training. Training loss measures memorization of seen examples; validation loss is a rough signal for generalization.

## Exercise

Print one batch, its shifted targets, and the decoded text. Verify the shapes `(batch_size, block_size)` and explain why the final target is not a new sample.
