---
title: Chapter 1 · LLM Fundamentals
part: Part 1 · Tokenization
sidebar_position: 1
slug: /
---

# Chapter 1: LLM Fundamentals

A large language model is not a giant database that memorizes answers. It is a probabilistic model that repeatedly predicts the next token from its context. This chapter builds that core intuition, then turns it into an engineering process you can inspect.

## Learning goals

By the end of this chapter, you will be able to:

- explain a language model through next-token prediction;
- distinguish pretraining, instruction tuning, and inference;
- read a minimal generation loop and explain temperature.

## 1. What is a language model?

Given text that has already appeared, a language model assigns probabilities to candidates for the next token. After the phrase “Paris is the capital of,” “France” receives a high probability while “banana” receives a low one. The model does not first need to retrieve the answer to the entire sentence; it repeatedly makes this local prediction well.

$$
P(x_1, x_2, \ldots, x_n) = \prod_i P(x_i \mid x_1, \ldots, x_{i-1})
$$

This factorization is called **autoregression**: token *i* depends only on the tokens before it. Append a predicted token to the context, predict once more, and text emerges one step at a time.

> **An easy detail to miss:** models learn sequences of tokens, not characters, words, or sentences. A token can be a whole word, part of a word, punctuation, or a common sequence of characters.

## 2. How does an LLM become useful?

Turning a language model into a capable assistant typically involves three stages with different goals. They all update or use the same Transformer, but the data, objective, and cost differ.

| Stage | Main input | What it solves |
| --- | --- | --- |
| Pretraining | Large-scale unlabeled text | Learns language, facts, and patterns while predicting the next token. |
| Instruction tuning | High-quality task and Q&A examples | Teaches the model to follow instructions and structure useful responses. |
| Inference | A user prompt and context | Keeps parameters fixed and generates an answer one token at a time. |

Pretraining resembles broad reading: the model compresses patterns from enormous amounts of text. Instruction tuning resembles worked examples: the same knowledge is organized into conversational behavior through prompt-and-response pairs. At inference time, the model is not acquiring new knowledge; it is generating from its current parameters and context.

## 3. The minimal generation loop

The pseudocode below captures the core of inference. A production framework uses batching, KV cache, and GPU kernels to make this fast, but the logic is unchanged.

```python
# the tokenizer has encoded the prompt as token ids
tokens = tokenizer.encode("Paris is the capital of")

for _ in range(max_new_tokens):
    logits = model(tokens)               # raw score for every candidate token
    probs = softmax(logits[-1] / temperature)
    next_token = sample(probs)           # or choose the most likely token
    tokens.append(next_token)
    if next_token == EOS:
        break

text = tokenizer.decode(tokens)
```

### Temperature: controlling randomness

Temperature scales logits before softmax. A lower value sharpens the probability distribution and makes output more stable; a higher value gives lower-probability candidates more opportunity, making output more varied. It does not add knowledge to the model. It only changes how the existing distribution is sampled.

> **Practical starting point:** use 0–0.3 for factual questions; try about 0.7 for creative writing or brainstorming. The result also depends on the model, top-p, and your prompt.

## 4. Capabilities—and their limits

At sufficient scale, next-token prediction can produce translation, summarization, code completion, and basic reasoning-like behavior. But the output is always text that is statistically plausible given its training and context, not direct access to truth. Fluent does not mean correct. For dependable conclusions, provide sources, call tools, or verify externally.

## Try it yourself

1. Use “Machine learning changes” as a prompt. List three plausible next tokens and explain your reasoning.
2. Why might moving temperature from 0.1 to 1.2 make an answer more creative but less stable?
3. Distinguish knowledge encoded in model parameters from context supplied in the current prompt.
