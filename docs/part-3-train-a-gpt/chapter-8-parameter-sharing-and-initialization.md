---
title: Chapter 8 · Parameter Sharing and Initialization
sidebar_position: 8
---

# Chapter 8: Parameter Sharing and Initialization

## 1. Parameter sharing

GPT-2 uses **weight tying**: the input token-embedding table (`transformer.wte.weight`) and the output language-model head (`lm_head.weight`) are the same learnable matrix. At the input, a token ID selects one row from this table to create a 768-dimensional embedding. At the output, the final hidden state is multiplied by the same matrix to produce one logit for each vocabulary token.

Both tensors have shape `[50257, 768]`. If they were stored independently, GPT-2 small would need an additional `50,257 × 768 = 38,597,376` parameters just for the output projection. Sharing them reduces the parameter count and keeps the representation used to read a token aligned with the representation used to score that token as an output candidate.

```python
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

print(sd_hf["lm_head.weight"].shape)
print(sd_hf["transformer.wte.weight"].shape)

print((sd_hf["lm_head.weight"] == sd_hf["transformer.wte.weight"]).all())

print(sd_hf["lm_head.weight"].data_ptr())
print(sd_hf["transformer.wte.weight"].data_ptr())
```

```text
torch.Size([50257, 768])
torch.Size([50257, 768])
tensor(True)
133296384950227
133296384950227
```

The equality check returns `True`, showing that the two tensors contain the same values. More importantly, the two `data_ptr()` calls return the same memory address. This proves that the checkpoint does not merely store two identical copies: `lm_head.weight` and `transformer.wte.weight` point to the same underlying parameter, so an update through either path changes both uses simultaneously.

The checkpoint demonstrates the desired behavior; our own `GPT` implementation must explicitly create it. The two modules are constructed first because both need a weight with shape `[vocab_size, n_embd]`. We then replace the embedding module's independently initialized weight with the language-model head's weight.

```python
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight
```

After this assignment, `transformer.wte.weight` and `lm_head.weight` reference the same `nn.Parameter`, not two tensors that merely start with the same values. Gradients from input embedding lookups and from next-token prediction both accumulate into that shared matrix, and the optimizer updates it only once. The assignment happens after both layers are created, so the model still exposes the two familiar GPT-2 checkpoint names while storing a single shared set of weights underneath.

## 2. Weights initialization

Most linear layers and embeddings use the standard GPT-2 normal initialization with `std=0.02`. The exception is a linear layer marked with `NANOGPT_SCALE_INIT`: its standard deviation is scaled down by `1 / √N`, where `N` is the total number of residual branches in the model.

Each Transformer block has two residual additions—one after causal self-attention and one after the feed-forward MLP—so GPT-2 has `N = 2 × n_layer` residual branches. For GPT-2 small, `n_layer=12`, therefore `N=24` and the residual output projections use:

$$
\operatorname{std}_{\text{residual}} = \frac{0.02}{\sqrt{2 \times n_{\text{layer}}}} = \frac{0.02}{\sqrt{24}} \approx 0.00408.
$$

```python
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

The marker is attached to the output projections that are added back into the residual stream: `attn.c_proj` and `mlp.c_proj`. It does not scale every linear layer—only the layers whose outputs repeatedly pass through skip connections. The small example below makes the reason for this scaling visible.

```python
x = torch.zeros(768)
y = torch.zeros(768)
n = 100
for i in range(n):
    x += torch.randn(768)
    y += (n ** -0.5) * torch.randn(768) 
print("std of x: ", x.std())
print("std of y: ", y.std())
```

```text
std of x:  tensor(9.8059)
std of y:  tensor(1.0004)
```

The vector `x` adds 100 independent random updates with standard deviation 1. Variances add, so its variance grows from roughly 1 to roughly 100; its standard deviation therefore grows to `√100 ≈ 10`, matching the printed value `9.8059`. This is the same accumulation effect that can occur when every residual branch adds an unscaled update to the Transformer residual stream.

For `y`, each update is first multiplied by `n ** -0.5`, which is `1 / √100`. Each individual update now has variance `1 / 100`; after summing 100 of them, the total variance remains about 1, and the output standard deviation stays near 1. The printed `1.0004` demonstrates this directly.

GPT-2 applies the same idea to its residual output projections. A model with `n_layer` blocks has two residual branches per block—attention and MLP—so `N = 2 × n_layer`. `NANOGPT_SCALE_INIT` changes their initialization from `0.02` to `0.02 / √N`. This prevents the variance of those repeated residual additions from growing with depth and keeps the final residual stream at an order-one standard deviation at initialization.
