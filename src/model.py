"""Minimal decoder-only Transformer.

`attn_type` is a deliberate seam: Stage 2 of the roadmap swaps this block
for sparse/linear attention or a selective-state block, and everything
else (embeddings, MLP, training loop, benchmark harness) stays fixed so
comparisons are apples-to-apples.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class LocalCausalSelfAttention(nn.Module):
    """Causal attention restricted to a fixed-size trailing window.

    Query at position i may attend to keys in [i - window_size + 1, i]
    (still causal — no peeking ahead). Reduces the attention term from
    O(T) per token to O(window_size) per token, independent of sequence
    length once T > window_size. Implemented as an explicit boolean mask
    over full QK^T rather than a blocked/sparse kernel — correct and easy
    to verify, but on CPU this doesn't skip the masked-out multiplications,
    so wall-clock latency won't show the saving that analytical FLOPs do.
    A real speedup needs a chunked/blocked implementation (candidate
    follow-up, see ROADMAP.md).
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.window_size = cfg["window_size"]
        self.qkv = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])

        block_size = cfg["block_size"]
        pos = torch.arange(block_size)
        distance = pos[:, None] - pos[None, :]  # i - j
        allowed = (distance >= 0) & (distance < self.window_size)  # causal + within window
        self.register_buffer("attn_mask", allowed, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        mask = self.attn_mask[:T, :T]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


def build_attention(cfg):
    if cfg["attn_type"] == "full":
        return CausalSelfAttention(cfg)
    if cfg["attn_type"] == "local":
        return LocalCausalSelfAttention(cfg)
    raise NotImplementedError(
        f"attn_type={cfg['attn_type']!r} not implemented yet — Stage 2 work"
    )


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = build_attention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg["block_size"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)

        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
