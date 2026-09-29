---
title: Chapter 3 · Loading the GPT-2 Checkpoint
sidebar_position: 3
---

# Chapter 3: Loading the GPT-2 Checkpoint

The `from_pretrained` class method connects our hand-written `GPT` implementation to an existing Hugging Face checkpoint. It supports all four GPT-2 sizes by selecting the corresponding number of Transformer blocks, attention heads, and embedding dimensions. The vocabulary size (`50257`) and context length (`1024`) stay fixed across these GPT-2 checkpoints.

The method first creates a randomly initialized model with the selected configuration and obtains its `state_dict`. It then loads the matching Hugging Face model and obtains a second `state_dict`. The two key lists are compared after removing attention-mask buffers: these masks are implementation details, not learned parameters, so they should not be copied like weights.

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

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        print("load pretrained model successfully")
        return model
```

For most tensors, loading is a direct copy. The important exception is the four matrices whose names end in `attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight`, or `mlp.c_proj.weight`. Hugging Face represents these layers with a GPT-2-style `Conv1D` layout, while our implementation uses standard `nn.Linear` layers. Their dimensions are therefore reversed, so the code transposes the pretrained matrix before copying it.

The shape assertions protect the conversion from silent errors: every ordinary parameter must have the same shape, and every transposed parameter must match after reversing its dimensions. The copies are performed inside `torch.no_grad()` because loading a checkpoint is initialization, not a differentiable training operation. Once all tensors have been copied, the returned model has the same learned parameters as the selected pretrained GPT-2 checkpoint.

```python
model = GPT.from_pretrained("gpt2")
```

```text
loading weights from pretrained gpt: gpt2
load pretrained model successfully
```
