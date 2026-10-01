---
title: Chapter 6 · The Transformer Block
sidebar_position: 6
---

# Chapter 6: The Transformer Block

## 1. The original encoder-decoder Transformer

The first Transformer used an encoder stack on the input side and a decoder stack on the output side. GPT keeps the decoder-style, causally masked path, but omits the separate encoder stack.

![Transformer encoder-decoder architecture, showing the encoder and decoder stacks, attention sublayers, embeddings, and output probabilities.](./assets/chapter-6/transformer-block.png)

> **Source:** Figure 1 from Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762) (2017).

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

The block applies two kinds of processing in sequence. **Multi-head self-attention** lets tokens exchange information with earlier positions; the **feedforward network** then processes the resulting representation at each position.

## 2. Feedforward network: computation at each position

`FeedFoward` is a small multilayer network applied independently to every token vector. The first linear layer expands each vector from `n_embd` features to `4 * n_embd`. `ReLU` adds a nonlinearity, allowing the network to build richer feature combinations. The second linear layer projects the result back to `n_embd`, so the block's representation width stays unchanged. Dropout is applied afterward during training to reduce over-reliance on particular activations.

The original Transformer paper writes this position-wise feedforward network as:

$$
\operatorname{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2
$$

Here, `x` is the vector at one token position, and `max(0, ·)` applies ReLU element by element. `W₁, b₁` expand the vector to the inner dimension `d_ff`; `W₂, b₂` map it back to `d_model`. The parameters are shared across positions, but each position is transformed independently. In this code, `d_model = n_embd` and `d_ff = 4 * n_embd`; the final dropout is an additional regularization step not shown in the paper's equation. ([Vaswani et al., 2017, Section 3.3](https://arxiv.org/abs/1706.03762))

Unlike self-attention, this network does not mix information between token positions: the same layers are applied to each position separately. Attention handles communication across the sequence; the feedforward network transforms what each position has learned.

## 3. Skip connections: keep and refine the representation

In `x = x + self.sa(self.ln1(x))`, the attention sublayer receives a normalized copy of `x`, while the original `x` travels along a direct path and is added back to the attention result. The next line does the same around the feedforward network: `x = x + self.ffwd(self.ln2(x))`.

These additions are **skip connections** (also called residual connections). Each sublayer can focus on learning an update to the existing representation instead of having to rebuild it from scratch. The direct path also helps information and gradients move through a deep stack of blocks.

## 4. LayerNorm placement: original paper and this code

The LayerNorm placement differs between the original paper's diagram and this implementation. In the paper, each sublayer is followed by the residual addition and then LayerNorm: `LayerNorm(x + Sublayer(x))` (post-LN). Here, LayerNorm comes first—once before multi-head attention and once before the feedforward network—followed by the residual addition: `x + Sublayer(LayerNorm(x))` (pre-LN). The sublayers and skip connections serve the same roles; only the placement of LayerNorm changes.
