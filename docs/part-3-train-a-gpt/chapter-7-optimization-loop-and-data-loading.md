---
title: Chapter 7 · Optimization Loop and Data Loading
sidebar_position: 7
---

# Chapter 7: Optimization Loop and Data Loading

## 1. DataLoader

`DataLoaderLite` turns the full tokenized corpus into a stream of training batches. It reads the source text once, encodes it with the GPT-2 tokenizer, and keeps the resulting token IDs in memory. This is deliberately minimal: it does not shuffle, pad, or use worker processes, which makes the data flow easy to see.

```python
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
```

At each call, `next_batch` takes `B × T + 1` consecutive tokens. The first `B × T` tokens become `x`, and the same sequence shifted forward by one token becomes `y`. It then advances `current_position` by `B × T`, so the next call reads the following chunk of text. When there are not enough tokens left for a complete batch, the cursor resets to the beginning. For `B=4` and `T=32`, every batch contains 128 next-token predictions.

## 2. Training loop

The loop first selects the fastest available device: CUDA when an NVIDIA GPU is available, otherwise Apple MPS when supported, and CPU as the fallback. The random seed makes initialization repeatable. With `B=4` and `T=128`, each step trains on `4 × 128 = 512` next-token predictions.

Each iteration follows the same update path: `next_batch()` returns input IDs and their one-token-shifted targets; the model produces logits and a cross-entropy loss; `loss.backward()` computes gradients; and `optimizer.step()` lets AdamW update every trainable parameter using a learning rate of `1e-3`. `optimizer.zero_grad()` then clears the accumulated gradients so the next step starts cleanly.

```python
# tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

import tiktoken
import torch
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

B, T = 4, 128

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

model = GPT(GPTConfig())
model.eval()
model.to(device)

training_loader = DataLoaderLite(B, T)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(100):
    start = time.time()
    x, y = training_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()
    epoch_time = (time.time() - start) * 1000 # ms
    tokens_per_second = (training_loader.B * training_loader.T) / (epoch_time / 1000)
    if i % 10 == 0:
        print(f"step {i}, loss: {loss.item()}, epoch time: {epoch_time:.2f}ms, token_per_second: {tokens_per_second:.2f}")
```

```text
using device: cuda
loaded 338025 tokens
1 epoch = 660 batches
step 0, loss: 11.025571823120117, epoch time: 43.46ms, token_per_second: 11782.27
step 10, loss: 7.654312610626221, epoch time: 35.76ms, token_per_second: 14318.18
step 20, loss: 6.965814113616943, epoch time: 35.71ms, token_per_second: 14339.02
step 30, loss: 7.273531436920166, epoch time: 35.70ms, token_per_second: 14340.36
step 40, loss: 5.822754859924316, epoch time: 35.74ms, token_per_second: 14327.54
step 50, loss: 6.045305252075195, epoch time: 35.71ms, token_per_second: 14338.16
step 60, loss: 5.828420639038086, epoch time: 35.71ms, token_per_second: 14337.30
step 70, loss: 6.148750305175781, epoch time: 35.74ms, token_per_second: 14325.73
step 80, loss: 6.203412055969238, epoch time: 35.87ms, token_per_second: 14275.44
step 90, loss: 6.275454044342041, epoch time: 35.88ms, token_per_second: 14269.09
```

The timer surrounds one complete optimization step: loading a batch, forward pass, backward pass, and parameter update. CUDA work is asynchronous, so `torch.cuda.synchronize()` waits for that queued GPU work before the elapsed time is read; without it, the reported time would be artificially low. The variable named `epoch_time` therefore measures one **step** in milliseconds, not a full epoch.

The code converts the 512 tokens processed in a step into `tokens_per_second`, then prints the loss and throughput every ten steps. In this run, the first CUDA step is slower because of one-time setup, while later steps settle around 160–170 ms and roughly 3,000 tokens per second. The loss drops from about `11.02` to roughly `6`, showing that the model is learning local regularities in the text. It is not monotonic because each step sees a different contiguous chunk of the dataset.

This compact implementation currently has no dropout layers, so `model.eval()` does not change the computation. Once dropout or other training-specific layers are introduced, use `model.train()` during this loop and reserve `model.eval()` for validation and generation.
