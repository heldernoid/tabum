"""Stage 1 — feature-distribution embedding (column-wise, over rows).

Each feature-group column is embedded independently. Cell encoding is
[Fourier(value), nan_flag] per feature, concatenated across the triplet, plus
(train rows only) a label embedding. Inducing-point attention (Set-Transformer
style) then computes column-level statistics: K learned inducing vectors
attend to the column's rows, and every row attends back to the K summaries.

Leakage constraint: the inducing vectors attend to TRAIN rows only. If they
attended to all rows, one test row's features would flow through the column
summary into another test row's embedding, violating the "test rows are
independent of each other" guarantee that Phase 2's tests enforce. Test rows
still *read* the summaries — they just don't write into them. This mirrors
how Stage 0 fits scalers on train rows only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import Attention, CrossAttnBlock, FeedForward, FourierFeatures, RMSNorm


class InducingBlock(nn.Module):
    """inducing -> rows (train-only keys), then rows -> inducing."""

    def __init__(self, dim: int, n_heads: int, n_inducing: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim) * 0.02)
        self.summarize = CrossAttnBlock(dim, n_heads)
        self.broadcast = CrossAttnBlock(dim, n_heads)

    def forward(self, x: torch.Tensor, train_size: int) -> torch.Tensor:
        # x: (B*, N, dim) where B* = batch * n_groups
        b = x.shape[0]
        ind = self.inducing.unsqueeze(0).expand(b, -1, -1)
        # keys sliced to train rows (structural equivalent of a keys-are-train
        # mask, but stays on the fast unmasked SDPA path)
        summaries = self.summarize(ind, x[:, :train_size])
        return self.broadcast(x, summaries)


class Stage1(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fourier = FourierFeatures(cfg.n_fourier_freqs)
        d_cell = self.fourier.out_dim + 1  # + nan flag
        self.d_cell = d_cell
        self.absent = nn.Parameter(torch.zeros(d_cell))  # padded "absent feature" slot
        self.cell_proj = nn.Linear(3 * d_cell, cfg.d_stage1)
        self.cls_label_emb = nn.Embedding(cfg.max_classes, cfg.d_stage1)
        self.reg_label_proj = nn.Linear(self.fourier.out_dim, cfg.d_stage1)
        self.blocks = nn.ModuleList(
            InducingBlock(cfg.d_stage1, cfg.stage1_heads, cfg.n_inducing)
            for _ in range(cfg.stage1_blocks)
        )
        self.norm = RMSNorm(cfg.d_stage1)

    def forward(
        self,
        z: torch.Tensor,  # (B, N, F) standardized values, NaN already zeroed
        nan_mask: torch.Tensor,  # (B, N, F) bool
        groups: torch.Tensor,  # (G, 3) long, -1 = absent slot
        y_emb_input: torch.Tensor,  # (B, N) — std. y (regression) or class idx (classification)
        train_mask: torch.Tensor,  # (B, N) bool — label embedding applied ONLY where True
        task_type: str,
        train_size: int,
    ) -> torch.Tensor:
        B, N, F_ = z.shape
        G = groups.shape[0]

        cells = torch.cat([self.fourier(z), nan_mask.unsqueeze(-1).to(z.dtype)], dim=-1)
        # gather into groups; absent slots (-1) get the learned absent vector
        safe_idx = groups.clamp(min=0)  # (G, 3)
        gathered = cells[:, :, safe_idx.reshape(-1), :].view(B, N, G, 3, self.d_cell)
        absent = (groups < 0).view(1, 1, G, 3, 1)
        gathered = torch.where(absent, self.absent.view(1, 1, 1, 1, -1), gathered)
        h = self.cell_proj(gathered.reshape(B, N, G, 3 * self.d_cell))  # (B, N, G, d1)

        # target-aware embedding, train rows only (label leakage guard: the
        # train_mask gate here is what test 4 in tests/test_invariance.py checks)
        if task_type == "classification":
            lab = self.cls_label_emb(y_emb_input.long().clamp(min=0))
        else:
            lab = self.reg_label_proj(self.fourier(y_emb_input))
        h = h + (lab * train_mask.unsqueeze(-1).to(lab.dtype)).unsqueeze(2)

        h = h.transpose(1, 2).reshape(B * G, N, -1)  # attention over rows, per group
        for block in self.blocks:
            h = block(h, train_size)
        return self.norm(h).view(B, G, N, -1).transpose(1, 2)  # (B, N, G, d1)
