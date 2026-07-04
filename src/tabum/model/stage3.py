"""Stage 3 — in-context learning across rows.

The train/test information-flow constraint (not a left-to-right causal mask):
- train rows attend to all train rows (bidirectional: the train set is an
  exchangeable context, order carries no meaning)
- test rows attend to train rows ONLY — never to other test rows and never to
  themselves (their own features arrive via the residual stream), so each test
  prediction is independent of every other test row.

Implemented structurally rather than with an attention mask: every query
attends to keys/values sliced to the train prefix (kv_len). One unmasked SDPA
call per block — flash-kernel eligible, and leakage-free by construction.

QASSMax query scaling uses the train-context length so attention entropy stays
controlled when inference contexts exceed pretraining sizes. RMSNorm
throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, SelfAttnBlock


class Stage3(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            SelfAttnBlock(cfg.d_model, cfg.stage3_heads, qassmax=True)
            for _ in range(cfg.stage3_blocks)
        )
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, rows: torch.Tensor, train_size: int) -> torch.Tensor:
        # rows: (B, N, d_model), train rows first
        for block in self.blocks:
            rows = block(rows, kv_len=train_size, context_len=train_size)
        return self.norm(rows)
