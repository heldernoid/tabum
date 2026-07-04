"""Run a non-tabUM baseline over a benchmark suite (resumable, CPU-friendly).

Same protocol as eval_benchmark.py (≤2000 train / ≤1000 test, split seed 0),
so rows join 1:1 with tabum results on `did`. Baselines are model-independent
of our checkpoints, so this can run while training does.

Baselines:
  histgb  — sklearn HistGradientBoosting (LightGBM-class); NaN-native, raw input
  tabpfn  — TabPFN (pip package, black-box; clean-room: no source consulted)
  tabicl  — TabICL (pip package, black-box; classification only)

Run: uv run python -u scripts/eval_baselines.py --model histgb \
        --suite validation/tabarena_suite.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabum.inference.encoding import to_numeric

MAX_TRAIN, MAX_TEST = 2000, 1000
FIELDS = ["did", "name", "kind", "n_rows", "n_features", "n_classes",
          "in_tuning_set", "score", "auc", "seconds", "status"]


def make_estimator(name: str, kind: str, device: str):
    if name == "histgb":
        cls = HistGradientBoostingClassifier if kind == "cls" else HistGradientBoostingRegressor
        return cls(random_state=0)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        cls = TabPFNClassifier if kind == "cls" else TabPFNRegressor
        # our protocol (≤2000 train rows) is inside TabPFN's pretraining range;
        # the flag only lifts its CPU>1000-rows speed guard
        return cls(device=device, ignore_pretraining_limits=True)
    if name == "tabicl":
        if kind != "cls":
            raise ValueError("tabicl is classification-only")
        from tabicl import TabICLClassifier

        return TabICLClassifier(device=device)
    raise ValueError(f"unknown model {name}")


def eval_one(model_name: str, entry: dict, kind: str, device: str) -> dict:
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

    est = make_estimator(model_name, kind, device)
    if kind == "cls":
        classes = np.unique(ytr)
        keep = np.isin(yte, classes)
        Xte, yte = Xte[keep], yte[keep]
        if yte.size < 10 or classes.size < 2:
            raise ValueError("degenerate test split")
        est.fit(Xtr, ytr)
        proba = est.predict_proba(Xte)
        pred_classes = getattr(est, "classes_", classes)
        row["score"] = accuracy_score(yte, np.asarray(pred_classes)[proba.argmax(1)])
        try:
            row["auc"] = roc_auc_score(
                yte, proba[:, 1] if proba.shape[1] == 2 else proba,
                multi_class="ovr" if proba.shape[1] > 2 else "raise",
                labels=classes)
        except ValueError:
            row["auc"] = np.nan
    else:
        est.fit(Xtr, ytr)
        row["score"] = r2_score(yte, est.predict(Xte))
        row["auc"] = np.nan
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["histgb", "tabpfn", "tabicl"])
    ap.add_argument("--suite", default="validation/benchmark_suite.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")  # GPU belongs to training right now
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text())
    work = [(e, "cls") for e in suite["classification"]]
    if args.model != "tabicl":
        work += [(e, "reg") for e in suite["regression"]]
    work.sort(key=lambda w: w[0]["n_rows"])

    suite_tag = Path(args.suite).stem.replace("_suite", "")
    out = Path(args.out or f"validation/baselines_{args.model}_{suite_tag}.csv")
    done = set()
    if out.exists():
        with out.open() as f:
            done = {int(r["did"]) for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} datasets already recorded")
    else:
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    n_done = 0
    for entry, kind in work:
        if entry["did"] in done:
            continue
        try:
            row = eval_one(args.model, entry, kind, args.device)
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
                  f"{args.model}={row['score']:.4f} ({row['seconds']}s)", flush=True)
        else:
            print(f"[{n_done}] {entry['name'][:32]:<32s} {row['status'][:80]}",
                  flush=True)
        if args.limit and n_done >= args.limit:
            break
    print(f"{args.model} baseline pass complete -> {out}")


if __name__ == "__main__":
    main()
