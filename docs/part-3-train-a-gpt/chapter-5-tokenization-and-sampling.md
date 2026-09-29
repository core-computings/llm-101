---
title: Chapter 5 · Tokenization and Sampling
sidebar_position: 5
---

# Chapter 5: Tokenization and Sampling

## 1. Encode the prompt

GPT-2 does not receive raw text directly. It uses its GPT-2 byte-pair encoding (BPE) tokenizer to convert text into vocabulary IDs. Here, `tiktoken.get_encoding("gpt2")` encodes the prompt `"hello, i am"` as four IDs: `[31373, 11, 1312, 716]`. These IDs refer to tokens in the fixed GPT-2 vocabulary of 50,257 entries; they are not character codes.

The encoded prompt is first converted to a tensor with shape `[T]`. `unsqueeze(0)` adds a batch dimension, and `repeat(num_return_sequences, 1)` makes a batch with shape `[B, T]`. Every row starts from the same prompt, but the random sampling step will let each row produce a different continuation.

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("hello, i am")
print(tokens)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
print(tokens)
```

```text
[31373, 11, 1312, 716]
tensor([[31373,    11,  1312,   716],
        [31373,    11,  1312,   716],
        [31373,    11,  1312,   716],
        [31373,    11,  1312,   716],
        [31373,    11,  1312,   716]])
```

## 2. Generate one token at a time

The sampling loop is autoregressive: at each iteration the model reads the current sequence `x`, predicts a distribution for the next token, appends the selected token, and then repeats. `model.eval()` switches the model to inference behavior, while `torch.no_grad()` ensures that PyTorch does not retain gradients during generation.

`model(x)` returns logits for every position with shape `[B, T, vocab_size]`, but only the final position is relevant for continuing the sequence. The expression `logits[:, -1, :]` extracts these next-token scores as `[B, vocab_size]`. Applying softmax converts the scores into probabilities. The code keeps only the 50 most likely candidates with `topk`, samples one candidate with `torch.multinomial`, maps that sampled position back to its real vocabulary ID with `torch.gather`, and concatenates it onto `x`.

```python
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
model.eval()

x = tokens

torch.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]
        # get probabilities
        probs = F.softmax(logits, dim=-1)
        # get topk results
        topk_probs, topk_indicies = torch.topk(probs, k=50, dim=-1) # (B, k)
        # pick next token from topk results
        idx = torch.multinomial(topk_probs, num_samples=1)
        # gather the mapping index
        next_tokens = torch.gather(topk_indicies, -1, idx)
        # append to the sequence
        x = torch.cat((x, next_tokens), dim=1)

# print the decode text
for i in range(num_return_sequences):
    output = x[i, :max_length].tolist()
    decode_output = enc.decode(output)
    print(">", decode_output)
```

```text
> hello, i am a c-cup, so i am going to change that. Asking for permission is a bad idea. And that really makes
> hello, i am on a break after reading your post on her new position. She has been fired for her efforts so i took this opportunity to take
> hello, i am still trying to figure out how it all works. Anyway, I've been having such terrible luck with the app lately.
> hello, i am a woman. I am a woman not of any male lineage. My life is all my own.
> hello, i am very sure you have done the right thing, and we need your help to fix it. Our new website has tons of great content
```

The seed makes this particular run reproducible. The five outputs differ because they independently sample from the same top-50 candidate distribution at every step. Once the sequence reaches `max_length`, `enc.decode` converts the token IDs back into readable text.
