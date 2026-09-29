---
title: Chapter 4 · Forward Pass and Logits
sidebar_position: 4
---

# Chapter 4: Forward Pass and Logits

The new `forward` function defines what happens when the GPT model receives a batch of token IDs. Its input, `idx`, has shape `[B, T]`, where `B` is the batch size and `T` is the sequence length. The first assertion ensures that the sequence fits inside GPT-2's context window: `T` cannot exceed `block_size` (1,024 for the GPT-2 checkpoint used here).

For every position from `0` to `T - 1`, the function looks up a positional embedding with shape `[T, n_embd]`. It also looks up the token embedding for every token ID, producing `[B, T, n_embd]`. Adding the two creates `x`, the initial representation of each token together with its position. PyTorch broadcasts the positional embeddings across the batch dimension, so the same position indices are used for every sequence in the batch.

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
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        ...
```

The tensor `x` then passes through all 12 Transformer blocks in order. Each block preserves the shape `[B, T, n_embd]`, but updates every token representation using causal attention and its feed-forward network. After the final `LayerNorm`, `lm_head` projects the hidden width `n_embd=768` to the vocabulary size, producing `logits` with shape `[B, T, 50257]`.

These logits are raw scores, not probabilities. At position `t`, the vector `logits[:, t, :]` contains one score for each possible next token. A later generation step will turn that vector into a probability distribution with softmax and select a token. When `targets` are supplied during training, the function also flattens the batch and sequence dimensions and computes cross-entropy loss across all `B × T` next-token predictions; when `targets` is omitted, it simply returns the logits for inference.
