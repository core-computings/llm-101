---
title: Chapter 3 · Byte Pair Encoding
sidebar_position: 3
---

# Chapter 3: Byte Pair Encoding

Byte Pair Encoding (BPE) builds a vocabulary by repeatedly merging the most frequent adjacent pair. It began as a compression idea and became a simple, effective way to learn subword units.

## The algorithm

1. Start with a sequence of bytes.
2. Count every adjacent pair.
3. Find the most frequent pair.
4. Replace that pair with a new token.
5. Repeat until the vocabulary reaches its target size.

For the training sequence `aaabdaaabac`, the pair `aa` may be frequent enough to merge first. Later merges can represent common words or word fragments with a single ID.

## Training versus inference

Training learns a deterministic list of merge rules. Inference applies those rules in the learned order; it does not invent new vocabulary entries. This separation lets a tokenizer run quickly and consistently in production.

> The vocabulary size is a design choice: larger vocabularies shorten sequences but increase the embedding and output matrices.

## Exercise

Write down the pair counts for a short string, perform one merge by hand, and repeat twice. Record the merge order; the order is part of the tokenizer's learned state.
