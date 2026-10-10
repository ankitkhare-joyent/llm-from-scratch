"""A small decoder-only transformer with the design choices current open models use.

Compared with the 2019 GPT-2 recipe, this follows what Llama, Qwen, and Karpathy's
nanochat converged on:
  * RMSNorm (here parameter-free) in pre-norm position, plus a final norm
  * rotary position embeddings (RoPE) instead of a learned position table
  * QK-norm: normalise queries and keys before the dot product for stable logits
  * grouped-query attention (GQA): fewer key/value heads than query heads
  * ReLU^2 MLP, no bias terms anywhere
  * untied input embedding and output head, with a tanh soft cap on the logits
  * zero-initialised output projections so every block starts as the identity

Tensor shape legend used in comments: B batch, T time (tokens), C channels (n_embd),
H heads, D head_dim, V vocab.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyllm.config import ModelConfig


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without a learnable gain: x / sqrt(mean(x^2) + eps)."""
    return F.rms_norm(x, (x.size(-1),))


def rope_tables(seq_len: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for rotary embeddings, shape (T, D/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by a position-dependent angle. x: (B, H, T, D)."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)


class KVCache:
    """Keys and values from earlier positions, so generation costs O(T) per token, not O(T^2)."""

    def __init__(self, cfg: ModelConfig, batch_size: int, device: torch.device, dtype: torch.dtype):
        shape = (cfg.n_layer, batch_size, cfg.n_kv_head, cfg.seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0  # number of positions already written
        self.capacity = cfg.seq_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.size(2)
        if self.pos + t > self.capacity:
            raise ValueError("KV cache is full; reset it and re-prefill a shorter context")
        self.k[layer, :, :, self.pos : self.pos + t] = k
        self.v[layer, :, :, self.pos : self.pos + t] = v
        end = self.pos + t
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, t: int) -> None:
        self.pos += t


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.c_k = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.c_v = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k = norm(q), norm(k)  # QK-norm

        prefix = 0
        if kv_cache is not None:
            prefix = kv_cache.pos
            k, v = kv_cache.update(self.layer_idx, k, v)

        if self.n_kv_head != self.n_head:  # GQA: each K/V head serves a group of Q heads
            repeat = self.n_head // self.n_kv_head
            k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

        if prefix == 0:
            # Plain causal mask; SDPA dispatches to a fused flash/memory-efficient kernel.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # New queries sit at positions prefix..prefix+T-1 and may see every cached key
            # plus the new keys at or before their own position.
            q_pos = torch.arange(prefix, prefix + T, device=x.device)[:, None]
            k_pos = torch.arange(k.size(2), device=x.device)[None, :]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=k_pos <= q_pos)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    """Pre-norm residual block: x + Attn(norm(x)), then x + MLP(norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.attn(norm(x), cos, sin, kv_cache)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base)
        # Buffers move with .to(device) but are not trained or saved in state_dict.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.wte.weight, std=1.0)
        # A tiny head means the first prediction is ~uniform: loss starts at ln(V).
        nn.init.normal_(self.lm_head.weight, std=0.001)
        bound = math.sqrt(3.0 / self.cfg.n_embd)  # uniform with std 1/sqrt(C)
        for block in self.blocks:
            for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.c_fc):
                nn.init.uniform_(lin.weight, -bound, bound)
            # Zero output projections: every block is the identity at step 0, so the
            # residual stream starts clean and deep stacks train stably.
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _B, T = idx.shape
        start = kv_cache.pos if kv_cache is not None else 0
        if start + T > self.cfg.seq_len:
            raise ValueError(f"sequence of {start + T} exceeds context of {self.cfg.seq_len}")
        cos, sin = self.cos[start : start + T], self.sin[start : start + T]

        x = norm(self.wte(idx))  # (B, T, C)
        for block in self.blocks:
            x = block(x, cos, sin, kv_cache)
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = self.lm_head(norm(x)).float()  # (B, T, V); fp32 for a stable softmax
        if self.cfg.logit_softcap:
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
