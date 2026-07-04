"""Output heads.

Classification: attention-based retrieval decoder — test embeddings query
train embeddings; class probabilities are the softmax-attention-weighted
average of one-hot train labels. Non-parametric in class count: works for 2
classes or 200 with the same weights.

Regression: bar distribution (discretized CDF) over fixed bins in
standardized-target space, with half-open tail bins. One forward pass yields
the full distribution; point estimates and arbitrary quantiles are decoded
from it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RetrievalClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q_proj = nn.Linear(cfg.d_model, cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.head_dim, bias=False)
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.head_dim = cfg.head_dim

    def forward(
        self,
        test_emb: torch.Tensor,  # (B, Nt, d_model)
        train_emb: torch.Tensor,  # (B, Ntr, d_model)
        y_train: torch.Tensor,  # (B, Ntr) long
        n_classes: int,
        chunk: int = 2048,
    ) -> torch.Tensor:
        """Returns class probabilities (B, Nt, n_classes).

        The (Nt, Ntr) attention matrix is materialized per test-row chunk, not
        all at once — at large Nt x Ntr a single full matrix can exceed system
        memory on unified-memory hardware (this froze the GB10 twice)."""
        q = self.q_proj(test_emb)
        k = self.k_proj(train_emb)
        onehot = F.one_hot(y_train, n_classes).to(q.dtype)  # (B, Ntr, C)
        temp = self.log_temp.exp()
        outs = []
        for i in range(0, q.shape[1], chunk):
            logits = q[:, i : i + chunk] @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            outs.append(torch.softmax(logits * temp, dim=-1) @ onehot)
        return torch.cat(outs, dim=1)

    def top_neighbors(
        self,
        test_emb: torch.Tensor,  # (B, Nt, d_model)
        train_emb: torch.Tensor,  # (B, Ntr, d_model)
        top_k: int = 5,
        chunk: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per test row, the top_k most-attended train rows: (weights, indices),
        each (B, Nt, top_k). These are the exact attention weights the forward
        pass votes with — a faithful case-based explanation, not a post-hoc one."""
        q = self.q_proj(test_emb)
        k = self.k_proj(train_emb)
        temp = self.log_temp.exp()
        top_k = min(top_k, k.shape[1])
        ws, idxs = [], []
        for i in range(0, q.shape[1], chunk):
            logits = q[:, i : i + chunk] @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            att = torch.softmax(logits * temp, dim=-1)
            w, idx = att.topk(top_k, dim=-1)
            ws.append(w)
            idxs.append(idx)
        return torch.cat(ws, dim=1), torch.cat(idxs, dim=1)


class BarDistributionRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_bins = cfg.n_reg_bins
        edges = torch.linspace(-cfg.reg_support, cfg.reg_support, cfg.n_reg_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_reg_bins)
        )

    def forward(self, test_emb: torch.Tensor) -> torch.Tensor:
        """(B, Nt, d_model) -> bin logits (B, Nt, n_bins) in standardized-y space."""
        return self.proj(test_emb)

    def nll(self, logits: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the bin containing y (targets clamped into the
        outermost bins, which therefore act as half-open tails)."""
        idx = torch.bucketize(y_std, self.edges[1:-1])  # (B, Nt) in [0, n_bins-1]
        return F.cross_entropy(logits.reshape(-1, self.n_bins), idx.reshape(-1))

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1) @ self.centers

    def quantile(self, logits: torch.Tensor, q: float) -> torch.Tensor:
        """Piecewise-linear inverse CDF at level q, in standardized-y space."""
        p = torch.softmax(logits, dim=-1)
        cdf = p.cumsum(-1)
        idx = torch.searchsorted(cdf, torch.full_like(cdf[..., :1], q)).clamp(max=self.n_bins - 1)
        cdf_lo = torch.where(idx > 0, cdf.gather(-1, (idx - 1).clamp(min=0)), torch.zeros_like(idx, dtype=p.dtype))
        p_bin = p.gather(-1, idx).clamp(min=1e-12)
        lo = self.edges[:-1][idx.squeeze(-1)]
        width = (self.edges[1:] - self.edges[:-1])[idx.squeeze(-1)]
        frac = ((q - cdf_lo) / p_bin).clamp(0, 1).squeeze(-1)
        return lo + frac * width
