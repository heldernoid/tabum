"""Run a checkpoint over validation/benchmark_suite.json (resumable).

Results stream to validation/benchmark_results.csv — one row per dataset,
appended as they finish; already-recorded dids are skipped on restart, so the
run can be interrupted freely. Datasets are processed smallest-first so
results accumulate fast.

Run: uv run --group validation python -u scripts/eval_benchmark.py \
        --checkpoint checkpoints/v1/step00050000.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

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
from tabum.train import Trainer

MAX_TRAIN, MAX_TEST = 2000, 1000
FIELDS = ["did", "name", "kind", "n_rows", "n_features", "n_classes",
          "in_tuning_set", "tabum", "baseline", "floor", "auc", "seconds", "status"]




def eval_one(model, entry: dict, kind: str, device: str, n_ensemble: int = 1,
             finetune: bool = False) -> dict:
    t0 = time.perf_counter()
    row = {**{k: entry.get(k) for k in ("did", "name", "n_rows", "n_features",
                                        "n_classes", "in_tuning_set")},
           "kind": kind, "status": "ok"}
    ds = fetch_openml(data_id=entry["did"], as_frame=True, parser="auto")
    X = to_numeric(ds.data)
    y = ds.target
    if y is None:
        raise ValueError("no target")
    y = pd.factorize(y)[0] if kind == "cls" else y.to_numpy(dtype=np.float64)
    if kind == "reg" and not np.isfinite(y).all():
        keep = np.isfinite(y)
        X, y = X[keep], y[keep]

    n_train = min(MAX_TRAIN, int(0.7 * len(y)))
    n_test = min(MAX_TEST, len(y) - n_train)
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=0,
            stratify=y if kind == "cls" else None)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=0)

    col_mean = np.nanmean(Xtr, 0, keepdims=True)
    col_mean = np.where(np.isnan(col_mean), 0, col_mean)
    imput = np.where(np.isnan(Xtr), col_mean, Xtr)
    imput_te = np.where(np.isnan(Xte), col_mean, Xte)
    scaler = StandardScaler().fit(imput)

    if kind == "cls":
        classes = np.unique(ytr)
        keep = np.isin(yte, classes)
        Xte, yte, imput_te = Xte[keep], yte[keep], imput_te[keep]
        if yte.size < 10 or classes.size < 2:
            raise ValueError("degenerate test split")
        est = TabUMClassifier(model=model, device=device, n_ensemble=n_ensemble)
        est = est.finetune(Xtr, ytr) if finetune else est.fit(Xtr, ytr)
        proba = est.predict_proba(Xte)
        row["tabum"] = accuracy_score(yte, est.classes_[proba.argmax(1)])
        try:
            row["auc"] = roc_auc_score(
                yte, proba[:, 1] if proba.shape[1] == 2 else proba,
                multi_class="ovr" if proba.shape[1] > 2 else "raise",
                labels=classes)
        except ValueError:
            row["auc"] = np.nan
        base = LogisticRegression(max_iter=1000).fit(scaler.transform(imput), ytr)
        row["baseline"] = accuracy_score(yte, base.predict(scaler.transform(imput_te)))
        row["floor"] = accuracy_score(yte, np.full_like(yte, np.bincount(ytr).argmax()))
    else:
        est = TabUMRegressor(model=model, device=device, n_ensemble=n_ensemble)
        est = est.finetune(Xtr, ytr) if finetune else est.fit(Xtr, ytr)
        row["tabum"] = r2_score(yte, est.predict(Xte))
        base = LinearRegression().fit(scaler.transform(imput), ytr)
        row["baseline"] = r2_score(yte, base.predict(scaler.transform(imput_te)))
        row["floor"] = 0.0
        row["auc"] = np.nan
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="validation/benchmark_suite.json")
    ap.add_argument("--out", default="validation/benchmark_results.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ensemble", type=int, default=1,
                    help="average predictions over N permuted views (test-time ensembling)")
    ap.add_argument("--finetune", action="store_true",
                    help="per-dataset gradient adaptation before predicting (finetune API)")
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text())
    work = ([(e, "cls") for e in suite["classification"]]
            + [(e, "reg") for e in suite["regression"]])
    work.sort(key=lambda w: w[0]["n_rows"])  # small first: results accumulate fast

    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {int(r["did"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} datasets already recorded")
    else:
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    model = Trainer.load_model(args.checkpoint, device=args.device)
    n_done = 0
    for entry, kind in work:
        if entry["did"] in done:
            continue
        try:
            row = eval_one(model, entry, kind, args.device, args.ensemble, args.finetune)
        except Exception as e:  # noqa: BLE001 — record and move on
            row = {**{k: entry.get(k) for k in ("did", "name", "n_rows",
                                                "n_features", "n_classes",
                                                "in_tuning_set")},
                   "kind": kind, "status": f"failed: {type(e).__name__}: {e}"[:200]}
        with out.open("a", newline="") as f:
            csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(row)
        n_done += 1
        if row.get("status") == "ok":
            print(f"[{n_done}] {entry['name'][:32]:<32s} [{kind}] "
                  f"tabum={row['tabum']:.4f} base={row['baseline']:.4f} "
                  f"({row['seconds']}s)", flush=True)
        else:
            print(f"[{n_done}] {entry['name'][:32]:<32s} {row['status'][:80]}",
                  flush=True)
        if args.limit and n_done >= args.limit:
            break
    print("benchmark pass complete")


if __name__ == "__main__":
    main()
