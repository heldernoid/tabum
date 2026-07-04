"""TabUM — full three-stage tabular ICL model.

One forward pass encodes the training rows and decodes predictions for the
test rows; there is no gradient step at inference. Rows [0:train_size] of the
input are the training set, the rest are test rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .heads import BarDistributionRegressor, RetrievalClassifier
from .preprocessing import build_groups, standardize, standardize_target
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3


class TabUM(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.stage1 = Stage1(self.cfg)
        self.stage2 = Stage2(self.cfg)
        self.stage3 = Stage3(self.cfg)
        self.cls_head = RetrievalClassifier(self.cfg)
        self.reg_head = BarDistributionRegressor(self.cfg)

    @classmethod
    def from_pretrained(cls, export_dir, device: str = "cpu") -> "TabUM":
        """Load a released model directory (model.safetensors + config.json),
        as produced by scripts/export_safetensors.py."""
        import json
        from pathlib import Path

        from safetensors.torch import load_file

        d = Path(export_dir)
        cfg = ModelConfig(**json.loads((d / "config.json").read_text())["model_config"])
        model = cls(cfg)
        model.load_state_dict(load_file(d / "model.safetensors"))
        return model.to(device).eval()

    def embed_rows(
        self,
        x: torch.Tensor,  # (B, N, F) raw values, NaN = missing, train rows first
        y_full: torch.Tensor,  # (B, N) labels; entries at test positions are IGNORED
        train_mask: torch.Tensor,  # (B, N) bool, True for rows [0:train_size]
        train_size: int,
        task_type: str,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stages 0-3. Exposed separately (rather than inlined in forward) so the
        label-leakage test can inject garbage at test positions of y_full and
        assert the output is unchanged."""
        if groups is None:
            groups = build_groups(x.shape[2])
        groups = groups.to(x.device)

        z, nan_mask = standardize(x, train_size)  # Stage 0
        if task_type == "regression":
            y_std, _, _ = standardize_target(y_full, train_size)
            y_emb_input = torch.where(train_mask, y_std, 0.0)  # keep test slots finite
        else:
            y_emb_input = torch.where(train_mask, y_full, 0.0)

        h = self.stage1(z, nan_mask, groups, y_emb_input, train_mask, task_type, train_size)
        rows = self.stage2(h)
        return self.stage3(rows, train_size)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F), train rows first
        y_train: torch.Tensor,  # (B, train_size) — test labels are never an input
        train_size: int,
        task_type: str,
        n_classes: int | None = None,
        groups: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        assert y_train.shape[1] == train_size
        train_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        train_mask[:, :train_size] = True
        y_full = torch.zeros(B, N, dtype=y_train.dtype, device=x.device)
        y_full[:, :train_size] = y_train

        emb = self.embed_rows(x, y_full.to(x.dtype) if task_type == "regression" else y_full,
                              train_mask, train_size, task_type, groups)
        train_emb, test_emb = emb[:, :train_size], emb[:, train_size:]

        if task_type == "classification":
            assert n_classes is not None
            probs = self.cls_head(test_emb, train_emb, y_train.long(), n_classes)
            return {"probs": probs}
        _, y_mean, y_std = standardize_target(y_train.to(x.dtype), train_size)
        logits = self.reg_head(test_emb)
        return {"logits": logits, "y_mean": y_mean, "y_std": y_std}

    def predict_mean(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Destandardized regression point estimate."""
        return self.reg_head.mean(out["logits"]) * out["y_std"] + out["y_mean"]

    def predict_quantile(self, out: dict[str, torch.Tensor], q: float) -> torch.Tensor:
        return self.reg_head.quantile(out["logits"], q) * out["y_std"] + out["y_mean"]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
