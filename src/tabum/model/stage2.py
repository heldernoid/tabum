"""Stage 2 — feature aggregation (row-wise, within a single row).

A few learned CLS tokens attend over one row's feature-group embeddings; their
concatenated final states become the row vector. Everything here is strictly
per-row (rows are processed as independent batch elements), so this stage can
never move information between rows. No positional encoding: the feature
groups are an unordered set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import CrossAttnBlock, RMSNorm, SelfAttnBlock


class Stage2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(cfg.n_cls_tokens, cfg.d_stage1) * 0.02)
        self.read_blocks = nn.ModuleList(
            CrossAttnBlock(cfg.d_stage1, cfg.stage2_heads) for _ in range(cfg.stage2_blocks)
        )
        self.mix_blocks = nn.ModuleList(
            SelfAttnBlock(cfg.d_stage1, cfg.stage2_heads) for _ in range(cfg.stage2_blocks)
        )
        self.norm = RMSNorm(cfg.d_stage1 * cfg.n_cls_tokens)
        self.out_proj = nn.Linear(cfg.d_stage1 * cfg.n_cls_tokens, cfg.d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, G, d1) -> row vectors (B, N, d_model)
        B, N, G, d1 = h.shape
        groups = h.reshape(B * N, G, d1)
        cls = self.cls.unsqueeze(0).expand(B * N, -1, -1)
        for read, mix in zip(self.read_blocks, self.mix_blocks):
            cls = read(cls, groups)
            cls = mix(cls)
        row = self.norm(cls.reshape(B * N, -1))
        return self.out_proj(row).view(B, N, -1)
