---
title: Chapter 4 · Training a Tokenizer from Scratch
sidebar_position: 4
---

# Chapter 4: Training a Tokenizer from Scratch

This chapter turns the BPE recipe into a small training program. The goal is not production speed; it is to make every state transition inspectable.

## Counting and merging

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
```

At each step, choose the most common pair, assign it a new ID, and replace all non-overlapping occurrences. Save the pair-to-ID mapping so encoding can reproduce the same result later.

## Measuring compression

The compression ratio is the original byte count divided by the resulting token count. Higher is not automatically better: very large tokens can reduce sequence length while making the vocabulary expensive and less flexible.

## Exercise

Train a toy vocabulary on a paragraph. Print the top pair at every step and plot token count against vocabulary size. Which patterns become single tokens first?
