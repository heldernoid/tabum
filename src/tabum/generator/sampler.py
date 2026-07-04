"""Top-level task sampler: family mixture, target conversion, splits.

Classification and regression share one code path: every family produces a
continuous target, and classification tasks quantile-bin it with Dirichlet-
sampled class proportions (so class balance varies realistically). The split
is applied by *reordering rows* so rows [0:train_size] are train — downstream
code never needs a separate index array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig, TaskSpec, sample_task_spec
from .gp import generate_gp
from .postprocess import postprocess
from .scm import generate_scm
from .tree import generate_tree

_FAMILIES = {"scm": generate_scm, "gp": generate_gp, "tree": generate_tree}


@dataclass
class SyntheticTask:
    X: np.ndarray  # (n_rows, n_features) float32, NaN = missing; train rows first
    y: np.ndarray  # (n_rows,) float32 (regression) or int64 (classification)
    task_type: str
    n_classes: int
    train_size: int
    cat_mask: np.ndarray
    family: str
    shift_split: bool


class TaskSampler:
    def __init__(self, cfg: GeneratorConfig | None = None, seed: int = 0):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(seed)

    def sample(self, spec: TaskSpec | None = None) -> SyntheticTask:
        """If a spec is given, its shape fields are honored on every retry
        (only the seed is redrawn) — batches rely on a shared shape."""
        cfg = self.cfg
        fixed = spec
        for attempt in range(40):  # rejection loop for degenerate draws
            if fixed is not None:
                s = fixed if attempt == 0 else TaskSpec(
                    **{**fixed.__dict__, "seed": int(self.rng.integers(0, 2**31 - 1))}
                )
            else:
                s = sample_task_spec(cfg, self.rng)
            task = self._try_generate(s, cfg, np.random.default_rng(s.seed))
            if task is not None:
                return task
        raise RuntimeError("failed to generate a non-degenerate task in 40 attempts")

    def _try_generate(
        self, s: TaskSpec, cfg: GeneratorConfig, rng: np.random.Generator
    ) -> SyntheticTask | None:
        X, y_cont, latent = _FAMILIES[s.family](rng, s.n_rows, s.n_features)
        if not np.isfinite(y_cont).all() or y_cont.std() < 1e-12:
            return None
        X, cat_mask = postprocess(X, s, cfg, rng)

        # --- target conversion (shared path for both task types) ---
        if s.task_type == "classification":
            y = _binarize_target(y_cont, s.n_classes, rng)
            if y is None:
                return None
        else:
            y = y_cont.astype(np.float32)

        # --- split: reorder rows so train rows come first ---
        n = s.n_rows
        train_size = int(np.clip(round(s.train_frac * n), cfg.min_train_rows, n - cfg.min_test_rows))
        if train_size >= 128:  # quantize: recurring kv_len shapes for the allocator
            train_size = (train_size // 64) * 64
        if train_size < cfg.min_train_rows or n - train_size < cfg.min_test_rows:
            return None
        if s.shift_split:
            order = np.argsort(latent, kind="stable")
            if rng.random() < 0.5:
                order = order[::-1].copy()
        else:
            order = rng.permutation(n)
        X, y = X[order], y[order]

        if s.task_type == "classification":
            # every class must appear in the train split; remap labels to a
            # dense range based on train-set classes, drop test rows with
            # train-unseen classes (mirrors what real eval protocols do)
            train_classes = np.unique(y[:train_size])
            if train_classes.size < 2:
                return None
            keep = np.isin(y, train_classes)
            keep[:train_size] = True
            X, y = X[keep], y[keep]
            remap = np.full(int(y.max()) + 1, -1, dtype=np.int64)
            remap[train_classes] = np.arange(train_classes.size)
            y = remap[y]
            if y.shape[0] - train_size < cfg.min_test_rows:
                return None
            n_classes = int(train_classes.size)
        else:
            n_classes = 0

        return SyntheticTask(
            X=np.ascontiguousarray(X, dtype=np.float32),
            y=y,
            task_type=s.task_type,
            n_classes=n_classes,
            train_size=train_size,
            cat_mask=cat_mask,
            family=s.family,
            shift_split=s.shift_split,
        )

    def sample_batch(self, batch_size: int) -> list[SyntheticTask]:
        """B tasks sharing one shape/type spec (so they collate into one tensor)."""
        base = sample_task_spec(self.cfg, self.rng)
        out = []
        for _ in range(batch_size):
            s = TaskSpec(**{**base.__dict__, "seed": int(self.rng.integers(0, 2**31 - 1))})
            out.append(self.sample(s))
        return out

    def sample_batch_budget(
        self, cell_budget: int, min_batch: int = 1, max_batch: int = 128,
        row_budget: int = 0,
    ) -> list[SyntheticTask]:
        """Batch size scales inversely with per-task cost (rows x feature
        groups), so small-shape batches don't underfill the GPU and
        large-shape batches don't blow past memory. row_budget separately caps
        rows x batch: the cell budget measures Stage-1 cost, but Stage-3 cost
        is rows x batch — a narrow task (1 feature group) at full batch puts
        30x the usual token count through the row transformer (the 2026-07-03
        freeze was a (64, 5760, 2) batch)."""
        base = sample_task_spec(self.cfg, self.rng)
        n_cols = base.n_features + base.n_noise_features
        cells = base.n_rows * max(1, -(-n_cols // 3))
        b = int(np.clip(cell_budget // max(cells, 1), min_batch, max_batch))
        if row_budget > 0:
            b = max(min_batch, min(b, row_budget // base.n_rows))
        out = []
        for _ in range(b):
            s = TaskSpec(**{**base.__dict__, "seed": int(self.rng.integers(0, 2**31 - 1))})
            out.append(self.sample(s))
        return out


def _binarize_target(
    y_cont: np.ndarray, n_classes: int, rng: np.random.Generator
) -> np.ndarray | None:
    props = rng.dirichlet(np.full(n_classes, 2.0))
    edges = np.quantile(y_cont, np.cumsum(props)[:-1])
    y = np.searchsorted(edges, y_cont).astype(np.int64)
    if np.unique(y).size < 2:  # target too discrete/degenerate for these cuts
        return None
    perm = rng.permutation(n_classes)  # class ids carry no ordinal information
    return perm[y]
