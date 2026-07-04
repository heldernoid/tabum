"""Gaussian-process generator family.

Covers smooth nonlinear structure that DAG+MLP combiners underrepresent.
Sampling uses random Fourier features (RFF) rather than exact GP draws, so a
task costs O(n_rows * n_rff) instead of O(n_rows^3) — required to meet the
throughput budget at large row counts. Kernel diversity comes from the RFF
frequency distribution: gaussian frequencies (RBF kernel) with sampled
lengthscale, plus optional discrete/periodic frequency components.
"""

from __future__ import annotations

import numpy as np


def _rff_function(rng: np.random.Generator, d_in: int, n_rff: int = 96):
    kernel = rng.choice(["rbf", "rbf_mix", "periodic"], p=[0.6, 0.25, 0.15])
    if kernel == "rbf":
        ls = np.exp(rng.uniform(np.log(0.2), np.log(3.0)))
        omega = rng.standard_normal((d_in, n_rff)) / ls
    elif kernel == "rbf_mix":  # sum of two lengthscales -> multi-scale structure
        ls1 = np.exp(rng.uniform(np.log(0.1), np.log(1.0)))
        ls2 = np.exp(rng.uniform(np.log(1.0), np.log(5.0)))
        omega = np.concatenate(
            [
                rng.standard_normal((d_in, n_rff // 2)) / ls1,
                rng.standard_normal((d_in, n_rff - n_rff // 2)) / ls2,
            ],
            axis=1,
        )
    else:  # periodic-ish: frequencies on a discrete grid
        base = rng.uniform(0.5, 4.0)
        omega = base * rng.integers(1, 6, size=(d_in, n_rff)).astype(np.float64)
        omega *= rng.standard_normal((d_in, n_rff)) * 0.1 + 1.0
    phase = rng.uniform(0, 2 * np.pi, size=n_rff)
    w = rng.standard_normal(n_rff) * np.sqrt(2.0 / n_rff)

    def f(Z: np.ndarray) -> np.ndarray:
        return np.cos(Z @ omega + phase) @ w

    return f


def generate_gp(
    rng: np.random.Generator, n_rows: int, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_continuous, shift_latent)."""
    d_latent = max(1, min(n_features, int(rng.integers(1, 9))))
    Z = rng.standard_normal((n_rows, d_latent))

    # Features are randomized (possibly nonlinear, possibly noisy) views of the
    # latent space, so features are correlated with each other and with y.
    X = np.empty((n_rows, n_features), dtype=np.float64)
    for j in range(n_features):
        if rng.random() < 0.5 and d_latent >= 1:
            X[:, j] = Z[:, int(rng.integers(0, d_latent))] + 0.1 * rng.standard_normal(n_rows)
        else:
            fj = _rff_function(rng, d_latent, n_rff=32)
            X[:, j] = fj(Z)

    f = _rff_function(rng, d_latent)
    noise = np.exp(rng.uniform(np.log(0.01), np.log(0.5)))
    y = f(Z) + noise * rng.standard_normal(n_rows)
    return X, y, Z[:, 0].copy()
