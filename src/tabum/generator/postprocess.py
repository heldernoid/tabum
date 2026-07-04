"""Post-processing: where realistic quirks are injected into clean family output.

Order matters and is fixed: noise features -> categorical conversion -> scale
distortion -> outliers -> missingness -> column shuffle. Missingness comes
after everything else so that MAR conditioning columns are already in their
final (distorted) form; the shuffle is last so column position carries no
information about column kind.
"""

from __future__ import annotations

import numpy as np

from .config import GeneratorConfig, TaskSpec


def postprocess(
    X: np.ndarray, spec: TaskSpec, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_final with NaNs for missing, cat_mask)."""
    n_rows = X.shape[0]

    # 1. irrelevant noise features
    if spec.n_noise_features > 0:
        noise = rng.standard_normal((n_rows, spec.n_noise_features))
        X = np.concatenate([X, noise], axis=1)
    n_cols = X.shape[1]
    cat_mask = np.zeros(n_cols, dtype=bool)

    # 2. categorical conversion: quantile-bin a continuous latent, then permute
    #    codes so the categories are genuinely non-ordinal. The fraction of
    #    categorical columns is sampled per task with a bimodal profile, and
    #    cardinality is skewed low, both matched to the Phase 1 validation.
    u = rng.random()
    if u < cfg.p_task_all_numeric:
        cat_frac = 0.0
    elif u < cfg.p_task_all_numeric + cfg.p_task_cat_heavy:
        cat_frac = rng.uniform(0.5, 1.0)
    else:
        cat_frac = rng.uniform(0.05, 0.4)
    for j in range(n_cols):
        if rng.random() >= cat_frac:
            continue
        card = int(2 + (cfg.max_cat_cardinality - 2) * rng.random() ** cfg.cardinality_skew)
        col = X[:, j]
        if np.unique(col).size < card:
            continue
        qs = np.quantile(col, np.linspace(0, 1, card + 1)[1:-1])
        codes = np.searchsorted(qs, col).astype(np.float64)
        perm = rng.permutation(card)
        X[:, j] = perm[codes.astype(np.int64)]
        cat_mask[j] = True

    # 3. scale diversity (numeric columns only; categorical codes stay integers)
    for j in np.flatnonzero(~cat_mask):
        if rng.random() >= cfg.p_scale_col:
            continue
        scale = 10.0 ** rng.uniform(-3, 3)
        if rng.random() < 0.05:  # near-constant column
            scale = 10.0 ** rng.uniform(-8, -6)
        X[:, j] = X[:, j] * scale + rng.standard_normal() * scale * rng.uniform(0, 10)

    # 4. outliers / heavy tails on a subset of numeric columns
    for j in np.flatnonzero(~cat_mask):
        if rng.random() >= cfg.p_outlier_col:
            continue
        cells = rng.random(n_rows) < cfg.outlier_cell_rate
        if rng.random() < 0.5:
            X[cells, j] *= rng.uniform(5, 50)
        else:
            X[cells, j] += rng.standard_cauchy(int(cells.sum())) * (np.abs(X[:, j]).mean() + 1e-9)

    # 5. missingness: gated per task (most real tables are fully observed),
    #    then MCAR or MAR (conditioned on another observed column) per column
    task_has_missing = rng.random() < cfg.p_task_has_missing
    for j in range(n_cols):
        if not task_has_missing or rng.random() >= cfg.p_missing_col:
            continue
        rate = rng.uniform(0.0, cfg.max_missing_rate)
        if rng.random() < cfg.p_mar and n_cols > 1:
            k = int(rng.choice([c for c in range(n_cols) if c != j]))
            cond = X[:, k]
            cond = np.where(np.isnan(cond), np.nanmedian(cond), cond)
            thr = np.quantile(cond, 1.0 - min(2 * rate, 0.95))
            candidates = cond > thr
            mask = candidates & (rng.random(n_rows) < min(1.0, 2 * rate))
        else:
            mask = rng.random(n_rows) < rate
        X[mask, j] = np.nan

    # 6. shuffle column order
    perm = rng.permutation(n_cols)
    return X[:, perm].astype(np.float32), cat_mask[perm]
