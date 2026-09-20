"""GPT-2-style decoder: pre-LN, GELU, learned absolute positions, tied embeddings, no dropout."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, T, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        # PAD sits at the end of every row, so a causal mask is all that is needed.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, n_layers: int = 10, d_model: int = 512,
                 n_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2 style residual scaling
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return F.linear(x, self.tok_emb.weight)  # tied input/output embeddings

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb}

    def param_groups(self, weight_decay: float):
        """Decay applies to 2-D weights of Linear layers only (not LN, biases, embeddings)."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if p.ndim >= 2 and "emb" not in n:
                decay.append(p)
            else:
                no_decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def lm_loss(logits, targets, pad_id: int = 0):
    """Mean next-token cross-entropy over non-PAD target positions."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                           ignore_index=pad_id)
