"""Shared building blocks: RMSNorm, attention (with optional QASSMax query
scaling), feed-forward, and Fourier value features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class FourierFeatures(nn.Module):
    """[v] -> [v, sin(2^k * v), cos(2^k * v)] over a fixed frequency ladder,
    giving resolution across scales without hand-tuned binning."""

    def __init__(self, n_freqs: int = 6):
        super().__init__()
        freqs = 2.0 ** torch.arange(-1, n_freqs - 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs * math.pi)

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.freqs.numel()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (...,) -> (..., out_dim)
        ang = v.unsqueeze(-1) * self.freqs
        return torch.cat([v.unsqueeze(-1), torch.sin(ang), torch.cos(ang)], dim=-1)


class Attention(nn.Module):
    """Multi-head attention. Supports cross-attention (kv != q source), an
    optional boolean mask (True = may attend), and optional QASSMax: queries
    are rescaled by s_h * log(n_context) per head so attention stays sharp at
    context lengths beyond those seen in pretraining."""

    def __init__(self, dim: int, n_heads: int, qassmax: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.qass_scale = nn.Parameter(torch.full((n_heads,), 0.15)) if qassmax else None

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        context_len: int | None = None,
    ) -> torch.Tensor:
        # q: (B, Nq, D); kv: (B, Nk, D); attn_mask: broadcastable to (B, h, Nq, Nk)
        kv = q if kv is None else kv
        B, Nq, D = q.shape
        Nk = kv.shape[1]
        qh = self.q_proj(q).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(kv).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(kv).view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        if self.qass_scale is not None:
            n_ctx = float(context_len if context_len is not None else Nk)
            scale = self.qass_scale * math.log(max(n_ctx, 2.0))
            qh = qh * scale.view(1, -1, 1, 1).to(qh.dtype)
        # CUDA grid dims cap at 65535: per-row stages (batch = tasks x rows) can
        # exceed it and fail with "invalid argument" — chunk the batch axis.
        if B > 32768:
            out = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        qh[i : i + 32768], kh[i : i + 32768], vh[i : i + 32768],
                        attn_mask=attn_mask,
                    )
                    for i in range(0, B, 32768)
                ]
            )
        else:
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, Nq, D))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttnBlock(nn.Module):
    """Pre-RMSNorm self-attention block."""

    def __init__(self, dim: int, n_heads: int, qassmax: bool = False, ff_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, qassmax=qassmax)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        context_len: int | None = None,
        kv_len: int | None = None,
    ) -> torch.Tensor:
        """kv_len: restrict keys/values to the first kv_len positions (train
        rows). Equivalent to a keys-are-train mask but stays on the fast
        unmasked SDPA path — this is the train/test information-flow
        constraint, expressed structurally instead of via a mask."""
        h = self.norm1(x)
        kv = h if kv_len is None else h[:, :kv_len]
        x = x + self.attn(h, kv, attn_mask=attn_mask, context_len=context_len)
        return x + self.ff(self.norm2(x))


class CrossAttnBlock(nn.Module):
    """Pre-RMSNorm cross-attention block (queries read from a separate kv set)."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_mult)

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = q + self.attn(self.norm_q(q), self.norm_kv(kv), attn_mask=attn_mask)
        return q + self.ff(self.norm2(q))
