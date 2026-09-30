---
title: Chapter 12 · Hardware-Friendly Model Shapes
sidebar_position: 12
---

# Chapter 12: Hardware-Friendly Model Shapes

GPT-2's tokenizer has 50,257 token IDs. Here we pad the model's `vocab_size` to 50,304, the next multiple of 128 (`50,304 = 393 × 128`). The tokenizer and training text stay the same; the extra 47 rows only enlarge the embedding table and output logits.

The expensive output projection multiplies hidden states by a weight matrix with one row per vocabulary entry. With 50,257 rows, its vocabulary dimension ends in a partial GPU tile. With 50,304 rows, that dimension is aligned, which can let CUDA matrix-multiplication kernels use their tiles and Tensor Cores more efficiently. The relevant pattern is **matrix-dimension alignment**, rather than the number of CUDA kernels. NVIDIA's [fully connected layer guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#step-1-padding-the-vocabulary-size) discusses this exact vocabulary-padding optimization.

```python
import tiktoken
import torch
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

B, T = 16, 1024

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

training_loader = DataLoaderLite(B, T)

# for GPUs supporting TF32
torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(100):
    start = time.time()
    x, y = training_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
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

before
```text
using device: cuda
loaded 338025 tokens
1 epoch = 20 batches
step 0, loss: 10.999507904052734, epoch time: 26101.93ms, token_per_second: 627.69
step 10, loss: 6.684609413146973, epoch time: 90.84ms, token_per_second: 180366.55
step 20, loss: 6.7426605224609375, epoch time: 90.79ms, token_per_second: 180454.65
step 30, loss: 6.446186065673828, epoch time: 91.26ms, token_per_second: 179526.41
step 40, loss: 6.540253639221191, epoch time: 92.08ms, token_per_second: 177937.54
step 50, loss: 6.383149147033691, epoch time: 91.39ms, token_per_second: 179279.11
step 60, loss: 6.407015800476074, epoch time: 91.15ms, token_per_second: 179743.82
step 70, loss: 6.131112098693848, epoch time: 91.81ms, token_per_second: 178461.55
step 80, loss: 6.2237548828125, epoch time: 91.17ms, token_per_second: 179706.21
step 90, loss: 5.996917724609375, epoch time: 92.02ms, token_per_second: 178041.26
```

after
```text
using device: cuda
loaded 338025 tokens
1 epoch = 20 batches
step 0, loss: 10.934228897094727, epoch time: 21180.06ms, token_per_second: 773.56
step 10, loss: 6.683698654174805, epoch time: 88.09ms, token_per_second: 185996.25
step 20, loss: 6.737393379211426, epoch time: 87.56ms, token_per_second: 187109.89
step 30, loss: 6.438484191894531, epoch time: 87.22ms, token_per_second: 187844.36
step 40, loss: 6.532954216003418, epoch time: 87.26ms, token_per_second: 187758.13
step 50, loss: 6.365345001220703, epoch time: 87.78ms, token_per_second: 186646.41
step 60, loss: 6.367377281188965, epoch time: 88.03ms, token_per_second: 186113.12
step 70, loss: 6.115792274475098, epoch time: 89.16ms, token_per_second: 183755.67
step 80, loss: 6.233377456665039, epoch time: 88.21ms, token_per_second: 185730.83
step 90, loss: 6.005648136138916, epoch time: 88.24ms, token_per_second: 185667.60
```

After the compilation warm-up, step time falls from roughly 91 ms to 88 ms, and throughput rises from about 180K to 186K tokens/s. Padding adds a little arithmetic, but this measurement is consistent with the aligned projection running more efficiently. The exact gain depends on the GPU and the kernels PyTorch selects; a profiler would be needed to attribute the entire difference to a specific kernel. The 47 padded IDs are not real tokenizer tokens, so generation should exclude them, and a pretrained GPT-2 checkpoint with 50,257-row embeddings cannot be loaded into this model without handling the shape difference.
