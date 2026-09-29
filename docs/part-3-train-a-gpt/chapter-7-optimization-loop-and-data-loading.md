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

The training loop repeats the same four operations: load a batch, run the model to obtain a loss, backpropagate to compute gradients, and update the parameters. `AdamW` is the optimizer; its learning rate of `1e-3` controls the size of each update. Moving `x`, `y`, and the model to the selected device ensures that all tensor operations run on the same CPU, CUDA GPU, or Apple MPS device.

```python
import tiktoken
import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

B, T = 4, 32

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

model = GPT(GPTConfig())
model.eval()
model.to(device)

training_loader = DataLoaderLite(B, T)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(10):
    x, y = training_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"step {i}, loss: {loss.item()}")
```

```text
using device: cpu
loaded 338025 tokens
1 epoch = 2640 batches
step 0, loss: 10.92757797241211
step 1, loss: 9.583258628845215
step 2, loss: 9.905508041381836
step 3, loss: 8.957135200500488
step 4, loss: 8.05516529083252
step 5, loss: 7.434377670288086
step 6, loss: 8.297660827636719
step 7, loss: 7.997189521789551
step 8, loss: 7.372183799743652
step 9, loss: 7.724493503570557
```

`loss.backward()` accumulates gradients for every trainable parameter, `optimizer.step()` applies the AdamW update, and `optimizer.zero_grad()` clears those gradients before the next batch. The loss falls quickly from about `10.93` toward `7–8`, which shows that the model is beginning to predict the local text distribution better. It is not perfectly monotonic because every step uses a different contiguous batch.

This compact implementation currently has no dropout layers, so `model.eval()` does not change the computation. Once dropout or other training-specific layers are introduced, use `model.train()` during this loop and reserve `model.eval()` for validation and generation.
