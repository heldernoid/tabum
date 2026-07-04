"""SCM / DAG generator family (primary).

Samples a random DAG, evaluates structural equations X_i = f_i(parents) + noise
in topological order (fully vectorized over rows), then extracts features and a
target node from the same causal graph, so feature-feature and feature-target
relationships are genuinely causal rather than independently painted on.

Structural diversity knobs (per TabPFN-3's "what naive generators get wrong"
checklist, reimplemented from the paper description only):
- three DAG samplers: Erdos-Renyi over a topological order, layered (MLP-like),
  and preferential attachment (hub-heavy)
- a pool of combiner mechanisms beyond weighted sums: small random MLPs with
  varied activations, piecewise-linear maps, periodic (incl. high-frequency)
  maps, hard thresholds, and multiplicative interactions
- varied exogenous noise: gaussian, uniform, student-t (heavy tails)
"""

from __future__ import annotations

import numpy as np

_ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sin": np.sin,
    "abs": np.abs,
    "identity": lambda z: z,
    "square": lambda z: np.sign(z) * np.minimum(z * z, 1e6),
}


def _sample_dag(rng: np.random.Generator, n_nodes: int) -> list[np.ndarray]:
    """Return parents[i] = array of node indices < i (nodes are in topo order)."""
    kind = rng.choice(["erdos", "layered", "hub"])
    parents: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    if kind == "erdos":
        p = rng.uniform(0.05, 0.5)
        for i in range(1, n_nodes):
            mask = rng.random(i) < p
            parents.append(np.flatnonzero(mask))
    elif kind == "layered":
        n_layers = int(rng.integers(2, 6))
        layer_of = np.sort(rng.integers(0, n_layers, size=n_nodes))
        for i in range(1, n_nodes):
            prev = np.flatnonzero(layer_of[:i] == layer_of[i] - 1)
            if prev.size == 0:
                prev = np.arange(i)
            k = int(rng.integers(1, min(prev.size, 6) + 1))
            parents.append(rng.choice(prev, size=k, replace=False))
    else:  # hub / preferential attachment
        degree = np.ones(n_nodes)
        for i in range(1, n_nodes):
            k = int(rng.integers(1, min(i, 4) + 1))
            w = degree[:i] / degree[:i].sum()
            chosen = rng.choice(i, size=k, replace=False, p=w)
            degree[chosen] += 1.0
            parents.append(np.sort(chosen))
    return parents


def _sample_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    kind = rng.choice(["gauss", "uniform", "student_t"], p=[0.6, 0.2, 0.2])
    if kind == "gauss":
        e = rng.standard_normal(n)
    elif kind == "uniform":
        e = rng.uniform(-1.7, 1.7, size=n)
    else:
        e = rng.standard_t(df=float(rng.uniform(2.0, 6.0)), size=n)
    return (scale * e).astype(np.float64)


def _combine(rng: np.random.Generator, P: np.ndarray) -> np.ndarray:
    """Map parent matrix P (n_rows, k) to one output column, vectorized."""
    n, k = P.shape
    kind = rng.choice(
        ["linear", "mlp", "piecewise", "periodic", "threshold", "interaction"],
        p=[0.25, 0.3, 0.15, 0.12, 0.1, 0.08],
    )
    if kind == "linear":
        w = rng.standard_normal(k)
        out = P @ w
    elif kind == "mlp":
        h = int(rng.integers(2, 9))
        act = _ACTIVATIONS[str(rng.choice(list(_ACTIVATIONS)))]
        w1 = rng.standard_normal((k, h)) / np.sqrt(k)
        b1 = rng.standard_normal(h)
        w2 = rng.standard_normal(h) / np.sqrt(h)
        out = act(P @ w1 + b1) @ w2
    elif kind == "piecewise":
        w = rng.standard_normal(k)
        z = P @ w
        knots = np.sort(rng.uniform(-2, 2, size=int(rng.integers(1, 5))))
        slopes = rng.standard_normal(knots.size + 1) * 2.0
        out = slopes[0] * z
        for j, t in enumerate(knots):
            out = out + (slopes[j + 1] - slopes[j]) * np.maximum(z - t, 0.0)
    elif kind == "periodic":
        w = rng.standard_normal(k)
        freq = np.exp(rng.uniform(np.log(0.5), np.log(20.0)))  # incl. high-frequency
        out = np.sin(freq * (P @ w) + rng.uniform(0, 2 * np.pi))
    elif kind == "threshold":
        w = rng.standard_normal(k)
        out = np.where(P @ w > rng.standard_normal() * 0.5, 1.0, -1.0)
    else:  # interaction
        i, j = rng.integers(0, k, size=2)
        out = np.clip(P[:, i] * P[:, j], -1e6, 1e6)
        if k > 2:
            out = out + P @ (rng.standard_normal(k) * 0.3)
    return out.astype(np.float64)


def generate_scm(
    rng: np.random.Generator, n_rows: int, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_continuous, shift_latent)."""
    n_nodes = max(n_features + 2, int(n_features * rng.uniform(1.2, 2.5)) + 1)
    parents = _sample_dag(rng, n_nodes)

    values = np.empty((n_rows, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        noise_scale = float(np.exp(rng.uniform(np.log(0.05), np.log(0.7))))
        if parents[i].size == 0:
            values[:, i] = _sample_noise(rng, n_rows, 1.0)
        else:
            signal = _combine(rng, values[:, parents[i]])
            std = signal.std()
            if std > 1e-12:
                signal = (signal - signal.mean()) / std
            values[:, i] = signal + _sample_noise(rng, n_rows, noise_scale)

    # Target: prefer a well-connected non-root node so y has causal parents.
    candidates = [i for i in range(n_nodes) if parents[i].size > 0]
    target = int(rng.choice(candidates)) if candidates else n_nodes - 1

    feature_pool = [i for i in range(n_nodes) if i != target]
    feat_idx = rng.choice(feature_pool, size=min(n_features, len(feature_pool)), replace=False)
    X = values[:, feat_idx]
    if X.shape[1] < n_features:  # degenerate tiny graphs: pad with noise columns
        pad = rng.standard_normal((n_rows, n_features - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)

    shift_latent = values[:, int(rng.choice(feat_idx))]
    return X, values[:, target], shift_latent
