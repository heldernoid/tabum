"""Stage 0 — in-graph preprocessing.

All statistics (per-column mean/std, target mean/std) are computed from TRAIN
rows only and applied to all rows. Computing them over test rows would leak
test-set information across rows and break the no-test-leakage guarantee.

Feature grouping is represented explicitly as an index tensor groups (G, 3)
into the column axis, with -1 marking padded "absent feature" slots. Passing
the grouping around (rather than hard-coding it) is what lets the column-
permutation-invariance test permute columns and remap the grouping
consistently, and lets training randomize the grouping seed.
"""

from __future__ import annotations

import torch


def build_groups(n_features: int, seed: int | None = None) -> torch.Tensor:
    """Partition columns into ceil(F/3) triplets; -1 pads the last group.

    Assumption vs spec: ARCHITECTURE.md describes a "cyclic-shift" assignment
    (i, i+1, i+3 mod n), but read literally that produces overlapping triplets
    (3x MORE tokens, not 3x fewer). We implement what the stated purpose
    requires — a partition into triplets — with an optional seeded shuffle so
    training sees varied groupings. Flagged in STATUS.md for human review.
    """
    idx = torch.arange(n_features)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        idx = idx[torch.randperm(n_features, generator=g)]
    pad = (-n_features) % 3
    if pad:
        idx = torch.cat([idx, torch.full((pad,), -1, dtype=torch.long)])
    return idx.view(-1, 3)


def standardize(
    x: torch.Tensor, train_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B, N, F) raw values with NaN for missing.

    Returns (values standardized by train-row stats with NaN->0, nan_mask).
    """
    nan_mask = torch.isnan(x)
    x_train = x[:, :train_size]
    train_nan = nan_mask[:, :train_size]
    cnt = (~train_nan).sum(dim=1).clamp(min=1)  # (B, F)
    mean = torch.where(train_nan, 0.0, x_train).sum(dim=1) / cnt
    var = (torch.where(train_nan, 0.0, (x_train - mean.unsqueeze(1)) ** 2)).sum(dim=1) / cnt
    std = var.sqrt().clamp(min=1e-8)
    z = (x - mean.unsqueeze(1)) / std.unsqueeze(1)
    z = torch.where(nan_mask, 0.0, z)
    z = z.clamp(-100, 100)  # heavy outliers shouldn't blow up activations
    return z, nan_mask


def standardize_target(
    y: torch.Tensor, train_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y: (B, N) float regression target. Stats from train rows only.

    Returns (y standardized, mean, std) — mean/std needed for inverse-transform.
    """
    y_train = y[:, :train_size]
    mean = y_train.mean(dim=1, keepdim=True)
    std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (y - mean) / std, mean, std
