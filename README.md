# tabUM

A small tabular in-context-learning foundation model: 14.31M parameters, zero-shot
classification and regression on any table, native missing-value handling, up to
100 classes and 2,000 features. MIT licensed, built clean-room.

**Weights:** [huggingface.co/helmo/tabum-v1.1](https://huggingface.co/helmo/tabum-v1.1)

## How it works

tabUM does not train on your data. `fit(X, y)` stores your rows; `predict(X_test)`
runs one forward pass in which every test row attends to every training row
(in-context learning). The model was pretrained once on millions of synthetic
tasks sampled from causal graphs, Gaussian processes, and random tree ensembles,
and learned a general algorithm for reading a table it has never seen.

Practical consequences:

- prediction in seconds, no tuning, no pipelines
- `NaN` is a first-class value: missingness is treated as signal, never imputed
- calibrated probabilities and full regression distributions (any quantile) from one pass
- context sizes validated to 64k rows

## Install

```bash
pip install git+https://github.com/heldernoid/tabum
```

## Quickstart

```python
import numpy as np
from huggingface_hub import snapshot_download
from tabum.model import TabUM
from tabum.inference import TabUMClassifier, TabUMRegressor

model = TabUM.from_pretrained(snapshot_download("helmo/tabum-v1.1"), device="cuda")

clf = TabUMClassifier(model=model, n_ensemble=8).fit(X_train, y_train)
proba = clf.predict_proba(X_test)          # calibrated probabilities
labels = clf.predict(X_test)

reg = TabUMRegressor(model=model, n_ensemble=8).fit(X_train, y_train)
mean = reg.predict(X_test)
p90 = reg.predict_quantile(X_test, 0.9)    # any quantile, same forward pass
```

Beyond zero-shot:

```python
clf = clf.finetune(X_train, y_train)   # ~15s per-dataset adaptation (cloned weights, early-stopped)
out = clf.explain(X_test)              # feature importances + the training rows the model attended to
```

See the [guided notebook](notebooks/release_v1_1.ipynb) for a full tour with charts.

## Benchmarks

TabArena-v0.1, all 51 datasets, identical protocol for every model (at most
2,000 train / 1,000 test rows, single split, seed 0). Competitors run as
black-box pip packages on the same splits. Full per-dataset numbers in
[results/](results/).

| model | params | cls accuracy (38) | reg R² (13) |
|---|---|---|---|
| logistic / linear regression (fitted) | | 0.8417 | 0.5597 |
| tabUM zero-shot, 1 pass | 14M | 0.8523 | 0.6284 |
| tabUM zero-shot, `n_ensemble=8` | 14M | 0.8581 | 0.6519 |
| **tabUM `finetune()` + `n_ensemble=8`** | 14M | **0.8635** | **0.6907** |
| HistGradientBoosting (fitted, default) | | 0.8669 | 0.7091 |
| TabPFN v2 (zero-shot) | ~11M | 0.8768 | 0.7471 |
| TabICL (zero-shot) | ~500M | 0.8801 | n/a |

Honest positioning: tabUM beats fitted linear baselines everywhere, ties default
gradient boosting on classification when finetuned (20/38 wins), and sits 1.3 to
1.7 accuracy points behind TabPFN v2 and TabICL, which are the state of the art
in this class. Where tabUM stands out:

- **many classes**: TabPFN-class models cap at 10 classes; tabUM handles 100
  (letter, 26 classes: 0.79 vs 0.74 for fitted logistic regression)
- **small noisy tables** (under ~1,000 rows): the learned prior beats boosting,
  which overfits
- **missing values**: consumed natively; on Titanic (12% missing cells fed raw)
  tabUM beats an imputed+scaled logistic regression pipeline
- **explainability**: `explain()` returns leave-one-covariate-out feature
  importances (retraining-free by construction) and the exact retrieval-attention
  votes behind each classification, validated against synthetic ground truth
  (ranking AUC 0.99 regression / 0.94 median classification)

## Repository layout

- `src/tabum/` model, synthetic data generator, training loop, sklearn-style estimators
- `scripts/` pretraining, benchmark runners, explain validation, safetensors export
- `notebooks/release_v1_1.ipynb` guided tour
- `results/` benchmark tables and checkpoint evaluation history
- `tests/` shape, invariance, and generator tests

## Training your own

The full pipeline is included: `scripts/pretrain.py` streams freshly generated
synthetic tasks (never a fixed dataset) into the model. v1.1 was trained on a
single NVIDIA DGX Spark (GB10, 128GB unified memory) in roughly 7 hours of
wall-clock for 20,000 steps, warm-started from v1 (which was trained for 50,000 steps).

## License

MIT
