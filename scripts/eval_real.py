"""Zero-shot eval of a checkpoint on the real-data holdout slice (Phase 4).

These datasets are excluded from the Phase 1 generator-validation comparison,
so no knowledge of them has influenced generator tuning. Reports accuracy /
ROC-AUC (classification) and R2 (regression) vs simple reference baselines
(majority class / train mean, plus logistic or linear regression).

Run: uv run --group validation python scripts/eval_real.py --checkpoint ckpt.pt
     [--untrained]  # baseline numbers with a random-init model
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabum.inference.encoding import to_numeric
from tabum.inference import TabUMClassifier, TabUMRegressor
from tabum.model import ModelConfig, TabUM
from tabum.train import Trainer

EVAL_HOLDOUT = {  # keep in sync with scripts/validate_generator.py
    "mfeat-zernike": (22, "cls"), "optdigits": (28, "cls"), "splice": (46, "cls"),
    "pendigits": (32, "cls"), "wdbc": (1510, "cls"), "cpu_act": (227, "reg"),
}
MAX_TRAIN, MAX_TEST = 2000, 1000




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.untrained:
        model = TabUM(ModelConfig())
    else:
        assert args.checkpoint, "pass --checkpoint or --untrained"
        model = Trainer.load_model(args.checkpoint, device=args.device)

    rows = []
    for name, (did, kind) in EVAL_HOLDOUT.items():
        ds = fetch_openml(data_id=did, as_frame=True, parser="auto")
        X = to_numeric(ds.data)
        y = ds.target
        y = pd.factorize(y)[0] if kind == "cls" else y.to_numpy(dtype=np.float64)
        n_train = min(MAX_TRAIN, int(0.7 * len(y)))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=n_train,
            test_size=min(MAX_TEST, len(y) - n_train),
            random_state=args.seed, stratify=y if kind == "cls" else None,
        )
        imput = np.where(np.isnan(Xtr), np.nanmean(Xtr, 0, keepdims=True), Xtr)
        imput_te = np.where(np.isnan(Xte), np.nanmean(Xtr, 0, keepdims=True), Xte)
        scaler = StandardScaler().fit(imput)

        if kind == "cls":
            est = TabUMClassifier(model=model, device=args.device).fit(Xtr, ytr)
            proba = est.predict_proba(Xte)
            acc = accuracy_score(yte, proba.argmax(1))
            auc = roc_auc_score(yte, proba[:, 1] if proba.shape[1] == 2 else proba,
                                multi_class="ovr" if proba.shape[1] > 2 else "raise")
            lin = LogisticRegression(max_iter=2000).fit(scaler.transform(imput), ytr)
            lin_acc = accuracy_score(yte, lin.predict(scaler.transform(imput_te)))
            maj = accuracy_score(yte, np.full_like(yte, np.bincount(ytr).argmax()))
            rows.append({"dataset": name, "type": kind, "tabum_acc": acc,
                         "tabum_auc": auc, "logreg_acc": lin_acc, "majority_acc": maj})
        else:
            est = TabUMRegressor(model=model, device=args.device).fit(Xtr, ytr)
            r2 = r2_score(yte, est.predict(Xte))
            lin = LinearRegression().fit(scaler.transform(imput), ytr)
            lin_r2 = r2_score(yte, lin.predict(scaler.transform(imput_te)))
            rows.append({"dataset": name, "type": kind, "tabum_r2": r2,
                         "linreg_r2": lin_r2})
        print(f"  {name}: {rows[-1]}", flush=True)

    df = pd.DataFrame(rows)
    print("\n", df.round(4).to_string(index=False))
    cls_rows = df[df["type"] == "cls"]
    if len(cls_rows):
        print(f"\nmean cls accuracy: tabum {cls_rows['tabum_acc'].mean():.4f} "
              f"vs logreg {cls_rows['logreg_acc'].mean():.4f} "
              f"vs majority {cls_rows['majority_acc'].mean():.4f}")


if __name__ == "__main__":
    main()
