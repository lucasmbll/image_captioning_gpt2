import torch
import torch.nn as nn
from torch.nn import functional as F

class QFormerConfig:
    def __init__(self, n_queries=32, n_layer=4, n_head=8, n_embd=512, vi_embd=768, dropout=0.1):
        self.n_queries = n_queries
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.vi_embd = vi_embd
        self.dropout = dropout

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.o = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, kv_dim):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(kv_dim, n_embd)
        self.v = nn.Linear(kv_dim, n_embd)
        self.o = nn.Linear(n_embd, n_embd)

    def forward(self, q_tokens, kv_ctx):
        B, Tq, C = q_tokens.shape
        Tk = kv_ctx.shape[1]
        q = self.q(q_tokens).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(kv_ctx).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(kv_ctx).view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.o(y)

class QFormerBlock(nn.Module):
    def __init__(self, cfg: QFormerConfig):
        super().__init__()
        self.ln_q = nn.LayerNorm(cfg.n_embd)
        self.self_attn = SelfAttention(cfg.n_embd, cfg.n_head)
        self.ln_cross = nn.LayerNorm(cfg.n_embd)
        self.cross_attn = CrossAttention(cfg.n_embd, cfg.n_head, cfg.vi_embd)
        self.ln_mlp = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd),
            nn.GELU(approximate='tanh'),
            nn.Dropout(cfg.dropout),
            nn.Linear(4*cfg.n_embd, cfg.n_embd),
        )

    def forward(self, q_tokens, vi_ctx):
        x = q_tokens + self.self_attn(self.ln_q(q_tokens))
        x = x + self.cross_attn(self.ln_cross(x), vi_ctx)
        x = x + self.mlp(self.ln_mlp(x))
        return x

class QFormer(nn.Module):
    def __init__(self, cfg: QFormerConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.n_queries, cfg.n_embd) * 0.02)
        self.blocks = nn.ModuleList([QFormerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.to_gpt = nn.Linear(cfg.n_embd, out_dim)

    def forward(self, vi_ctx):
        B = vi_ctx.size(0)
        q = self.query_tokens.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, vi_ctx)
        q = self.ln_out(q)  # (B, n_queries, n_embd)
        return self.to_gpt(q)