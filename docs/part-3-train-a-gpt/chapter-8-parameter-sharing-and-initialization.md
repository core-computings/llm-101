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
