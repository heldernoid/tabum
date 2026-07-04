"""Configuration and per-task hyperparameter sampling for the synthetic prior.

Every knob that controls task diversity lives here so that a pretraining run
can be reproduced from (config, seed) alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    # --- task shape envelope ---
    min_rows: int = 50
    max_rows: int = 2048  # v1.1 runs raise this via CLI (log-uniform: big = rare tail)
    min_features: int = 2
    max_features: int = 100
    max_cells_per_task: int = 1_000_000  # cap rows*ceil(features/3): bounds training memory

    # --- task type ---
    p_classification: float = 0.7  # v1.1 uses 0.5 (eval battery grades 50/50)
    min_classes: int = 2
    max_classes: int = 10  # raise toward 100 once many-class training is wanted

    # --- generator family mixture (normalized at sample time) ---
    w_scm: float = 0.6
    w_gp: float = 0.2
    w_tree: float = 0.2

    # --- post-processing coverage ---
    # Categorical mix is bimodal in real data (2026-07-02 validation vs 32
    # OpenML datasets): many all-numeric tables, a tail of cat-heavy ones.
    p_task_all_numeric: float = 0.45
    p_task_cat_heavy: float = 0.2  # cat fraction ~ U(0.5, 1.0)
    # otherwise: cat fraction ~ U(0.05, 0.4)
    max_cat_cardinality: int = 50
    cardinality_skew: float = 5.0  # card = 2 + (max-2)*u^skew; real median is ~3
    p_task_has_missing: float = 0.25  # most real tables are fully observed
    p_missing_col: float = 0.3  # per-column, within tasks that have missingness
    max_missing_rate: float = 0.3
    p_mar: float = 0.4  # among missing columns: MAR (vs MCAR)
    p_outlier_col: float = 0.15
    outlier_cell_rate: float = 0.01
    p_scale_col: float = 0.5  # per-column probability of scale/shift distortion
    max_noise_feature_frac: float = 0.4  # irrelevant features, fraction of informative count

    # --- split modes ---
    p_shift_split: float = 0.18  # temporal / covariate-shift split (sorted by latent)
    min_train_frac: float = 0.3
    max_train_frac: float = 0.9
    min_train_rows: int = 10
    min_test_rows: int = 5


@dataclass
class TaskSpec:
    """Concrete hyperparameters for one synthetic task, sampled from GeneratorConfig."""

    n_rows: int
    n_features: int  # informative features requested from the family generator
    n_noise_features: int
    task_type: str  # "classification" | "regression"
    n_classes: int  # 0 for regression
    family: str  # "scm" | "gp" | "tree"
    shift_split: bool
    train_frac: float
    seed: int
    extra: dict = field(default_factory=dict)


def _bucket(x: float, lo: int, hi: int) -> int:
    """Snap to a half-octave grid (x1.41 steps) so tensor shapes recur and the
    CUDA caching allocator can reuse pools — unique shapes every batch made
    reserved memory grow unboundedly (v1.1 launch postmortem)."""
    import math

    return int(min(hi, max(lo, 2 ** (round(math.log2(max(x, 1)) * 2) / 2))))


def sample_task_spec(cfg: GeneratorConfig, rng) -> TaskSpec:
    n_rows = _bucket(_log_uniform(rng, cfg.min_rows, cfg.max_rows),
                     cfg.min_rows, cfg.max_rows)
    n_feat_total = _bucket(_log_uniform(rng, cfg.min_features, cfg.max_features),
                           cfg.min_features, cfg.max_features)
    n_noise = int(rng.uniform(0.0, cfg.max_noise_feature_frac) * n_feat_total)
    n_informative = max(1, n_feat_total - n_noise)

    # cap task cost: rows x feature-groups bounds activation memory in training
    groups = -(-n_feat_total // 3)
    if n_rows * groups > cfg.max_cells_per_task:
        n_feat_total = max(cfg.min_features, int(cfg.max_cells_per_task / n_rows) * 3)
        n_noise = min(n_noise, n_feat_total - 1)
        n_informative = max(1, n_feat_total - n_noise)

    is_cls = rng.random() < cfg.p_classification
    n_classes = int(_log_uniform(rng, cfg.min_classes, cfg.max_classes + 1)) if is_cls else 0
    # a k-class task needs enough train rows to exhibit k classes meaningfully
    n_classes = min(n_classes, cfg.max_classes, max(2, int(n_rows * cfg.min_train_frac / 10)))

    weights = [cfg.w_scm, cfg.w_gp, cfg.w_tree]
    total = sum(weights)
    family = rng.choice(["scm", "gp", "tree"], p=[w / total for w in weights])

    return TaskSpec(
        n_rows=n_rows,
        n_features=n_informative,
        n_noise_features=n_noise,
        task_type="classification" if is_cls else "regression",
        n_classes=n_classes,
        family=str(family),
        shift_split=rng.random() < cfg.p_shift_split,
        train_frac=float(rng.uniform(cfg.min_train_frac, cfg.max_train_frac)),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def _log_uniform(rng, lo: float, hi: float) -> float:
    import numpy as np

    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
