---
title: Chapter 1 · Reproducing GPT-2
sidebar_position: 1
---

# Chapter 1: Reproducing GPT-2

## 1. Load GPT-2 model

The original GPT-2 model uses a vocabulary of **50,257 tokens**. Each token is mapped to one row of the token-embedding table, so `transformer.wte.weight` has shape `[50257, 768]`: there is one 768-dimensional embedding vector for every token ID.

GPT-2 also has a **context window of 1,024 tokens**. This is the maximum number of token positions the model can process in one sequence, which is why the positional-embedding table `transformer.wpe.weight` has shape `[1024, 768]`. The first dimension enumerates the 1,024 possible positions; the second stores the 768-dimensional positional embedding for each position.

```python
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)
```

```text
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias torch.Size([2304])
transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias torch.Size([768])
transformer.h.0.ln_2.weight torch.Size([768])
...
```

## 2. Generate text

Once the checkpoint is loaded, GPT-2 can generate text one token at a time. The `pipeline` helper bundles the GPT-2 model with its tokenizer: it converts the prompt into token IDs, runs the model, samples a next token, appends that token to the context, and repeats the process until the requested number of new tokens has been produced.

Here `max_new_tokens=30` limits each completion to 30 newly generated tokens, while `num_return_sequences=3` asks for three independent continuations of the same prompt. Setting the random seed makes the sampling choices reproducible for this example; different library versions or generation settings may still produce slightly different text.

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_new_tokens=30, num_return_sequences=3)
```

```text
[{'generated_text': "Hello, I'm a language model, I work in front-end development. I wanted to keep writing code for the game, so I think the game needs to have a lot of logic"},
 {'generated_text': "Hello, I'm a language model, and I know what it is like for the human mind. So I'll say this: I want to see a language that looks like our world,"},
 {'generated_text': "Hello, I'm a language model, so I'm going to take it a step further, and go from a single-strategy (a model) to a more complex model, from"}]
```
