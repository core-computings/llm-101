---
title: Chapter 8 · From nanoGPT to ChatGPT
sidebar_position: 8
---

# Chapter 8: From nanoGPT to ChatGPT

The model built in this Part is a compact decoder-only Transformer. It captures the central mechanism, but production systems add scale, distributed training, data pipelines, evaluation, and alignment stages.

## Three training stages

1. **Pretraining** learns broad language patterns by predicting the next token over a large corpus.
2. **Supervised fine-tuning** teaches the pretrained model to follow instructions using curated examples.
3. **Preference optimization or RLHF** shifts behavior toward responses people rate as useful, safe, and well-formed.

The architecture can remain largely the same while the data and objective change. This is why a small character model is useful pedagogically: it exposes the mechanics without hiding them behind a library.

## Reading nanoGPT

Map the concepts in this Part to a production-style codebase: the data loader creates batches, the model implements repeated Transformer blocks, and the training script handles optimization and checkpoints. The details scale; the causal next-token objective remains recognizable.

## Capstone exercise

Train the smallest model that produces recognizable samples, then write a one-page report covering data, tokenizer, context length, model size, loss curves, and failure cases.
