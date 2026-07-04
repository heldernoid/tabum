"""Quantitative validation of .explain() against generator ground truth.

We own the synthetic prior, so explanations are testable: build tasks from a
family generator (informative columns) plus appended pure-noise columns, and
check that feature_importances ranks informative above noise (ROC-AUC of the
separation). Also sanity-checks neighbor explanations: the top-attended train
row should usually carry the predicted class.

Run: uv run python scripts/validate_explain.py --model release/v1.1 --tasks 40
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tabum.generator.gp import generate_gp
from tabum.generator.scm import generate_scm
from tabum.generator.tree import generate_tree
from tabum.inference import TabUMClassifier, TabUMRegressor
from tabum.model import TabUM

FAMILIES = [generate_scm, generate_gp, generate_tree]
N_ROWS, N_TRAIN, N_INFORMATIVE, N_NOISE = 700, 500, 6, 6


def make_task(rng: np.random.Generator, kind: str):
    """Informative columns first, then pure-noise columns — known ground truth."""
    gen = FAMILIES[rng.integers(len(FAMILIES))]
    for _ in range(20):
        X, y_cont, _ = gen(rng, N_ROWS, N_INFORMATIVE)
        if np.isfinite(y_cont).all() and y_cont.std() > 1e-9 and X.shape[1] == N_INFORMATIVE:
            break
    else:
        return None
    X = np.concatenate([X, rng.standard_normal((N_ROWS, N_NOISE))], axis=1)
    X = X.astype(np.float32)
    if kind == "cls":
        n_classes = int(rng.integers(2, 7))
        edges = np.quantile(y_cont, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.searchsorted(edges, y_cont).astype(np.int64)
        if np.unique(y[:N_TRAIN]).size < 2:
            return None
    else:
        y = y_cont.astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="release/v1.1")
    ap.add_argument("--tasks", type=int, default=40, help="tasks per kind")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabUM.from_pretrained(args.model, device=device)
    rng = np.random.default_rng(args.seed)
    truth = np.array([1] * N_INFORMATIVE + [0] * N_NOISE)

    for kind, cls in (("cls", TabUMClassifier), ("reg", TabUMRegressor)):
        aucs, top1_agree = [], []
        made = 0
        while made < args.tasks:
            task = make_task(rng, kind)
            if task is None:
                continue
            X, y = task
            est = cls(model=model, device=device).fit(X[:N_TRAIN], y[:N_TRAIN])
            out = est.explain(X[N_TRAIN:])
            imp = out["feature_importances"]
            if not np.isfinite(imp).all():
                continue
            aucs.append(roc_auc_score(truth, imp))
            if kind == "cls":
                pred = est.predict(X[N_TRAIN:])
                top1_agree.append((out["neighbor_label"][:, 0] == pred).mean())
            made += 1
        aucs = np.array(aucs)
        print(f"{kind}: informative-vs-noise ranking AUC "
              f"mean {aucs.mean():.3f}  median {np.median(aucs):.3f}  "
              f"(perfect=1.0, chance=0.5; {int((aucs > 0.9).sum())}/{len(aucs)} tasks >0.9)")
        if top1_agree:
            print(f"     top-1 neighbor carries predicted class: "
                  f"{np.mean(top1_agree):.1%} of test rows")


if __name__ == "__main__":
    main()
