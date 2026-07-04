"""Random tree/forest generator family.

Covers piecewise-constant, axis-aligned, interaction-heavy structure common in
categorical-heavy business data. The target is produced by a randomly grown
decision tree (or a small ensemble) evaluated over the features, fully
vectorized: each split routes the whole row set at once.
"""

from __future__ import annotations

import numpy as np


def _sample_X(rng: np.random.Generator, n_rows: int, n_features: int) -> np.ndarray:
    kind = rng.choice(["gauss", "mixture", "uniform", "correlated"])
    if kind == "gauss":
        return rng.standard_normal((n_rows, n_features))
    if kind == "uniform":
        return rng.uniform(-2, 2, size=(n_rows, n_features))
    if kind == "mixture":
        k = int(rng.integers(2, 6))
        centers = rng.standard_normal((k, n_features)) * 2.0
        comp = rng.integers(0, k, size=n_rows)
        return centers[comp] + rng.standard_normal((n_rows, n_features)) * rng.uniform(0.3, 1.0)
    # correlated gaussian via random low-rank mixing
    r = max(1, n_features // 2)
    A = rng.standard_normal((r, n_features))
    return rng.standard_normal((n_rows, r)) @ A / np.sqrt(r)


def _eval_random_tree(
    rng: np.random.Generator, X: np.ndarray, max_depth: int
) -> np.ndarray:
    """Grow a random tree top-down; returns a leaf value per row."""
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    stack = [(np.arange(n), 0)]
    while stack:
        idx, depth = stack.pop()
        if depth >= max_depth or idx.size < 8 or rng.random() < 0.15:
            out[idx] = rng.standard_normal()
            continue
        j = int(rng.integers(0, X.shape[1]))
        col = X[idx, j]
        thr = np.quantile(col, rng.uniform(0.2, 0.8))
        left = idx[col <= thr]
        right = idx[col > thr]
        if left.size == 0 or right.size == 0:
            out[idx] = rng.standard_normal()
            continue
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return out


def generate_tree(
    rng: np.random.Generator, n_rows: int, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_continuous, shift_latent)."""
    X = _sample_X(rng, n_rows, n_features)
    n_trees = int(rng.integers(1, 5))
    max_depth = int(rng.integers(2, 9))
    y = np.zeros(n_rows, dtype=np.float64)
    for _ in range(n_trees):
        y += _eval_random_tree(rng, X, max_depth)
    y /= np.sqrt(n_trees)
    noise = np.exp(rng.uniform(np.log(0.01), np.log(0.4)))
    y += noise * rng.standard_normal(n_rows)
    return X, y, X[:, int(rng.integers(0, n_features))].copy()
