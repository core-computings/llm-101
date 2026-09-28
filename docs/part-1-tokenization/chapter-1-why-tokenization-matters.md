---
title: Chapter 1 · Why Tokenization Matters
sidebar_position: 1
slug: /
---

# Chapter 1: Why Tokenization Matters

Before a Transformer can process text, a tokenizer turns a string into a sequence of integer IDs. Those IDs are the interface between human-readable text and the model's numerical computation.

## Learning goals

- explain where tokenization sits in the LLM pipeline;
- compare character, word, and subword units;
- recognize common tokenization failure modes.

## The first stage of the pipeline

```text
text → tokenizer → token IDs → embeddings → Transformer → token IDs → text
```

A token is not necessarily a word. Depending on the vocabulary, `tokenization` might be one token, several subword tokens, or a sequence of bytes. The model learns patterns over these units, so the tokenizer affects context length, cost, multilingual behavior, and even spelling and arithmetic.

## Why not use words or characters?

Word vocabularies become enormous and cannot contain every new name or typo. Character vocabularies have a small dictionary, but sequences become very long. Subword tokenization is a practical compromise: frequent pieces get their own IDs while rare strings can still be represented by smaller pieces.

> **Key idea:** tokenization is a separate, trained stage. A model cannot fix a poor split simply by being larger.

## Try it yourself

Tokenize the same sentence with a few different libraries. Compare the number of tokens, the treatment of spaces, and the treatment of an emoji or a non-English word. Keep these observations for the next chapters.
