# tabUM v1.1 — final results (TabArena-v0.1, 51 datasets)

Model: 14.31M params, checkpoint `checkpoints/v1.1/step00020000.pt`
(warm-start from v1 step50000; envelope 16,384 rows / 2,000 features /
100 classes / 50-50 cls-reg; run stopped at 20k of 30k steps by owner decision
— TabArena curve was flat from step 7,500; fully resumable).
Release weights: `release/v1.1/model.safetensors` (57.2 MB, round-trip verified).

Protocol: ≤2,000 train / ≤1,000 test rows, single split (seed 0), identical
for every model. Competitors run as black-box pip packages on the same splits
(tabpfn 8.0.8 with ungated v2 weights, ens. default; tabicl 2.1.1, cls-only).

## Headline table (means over the suite)

| model | cls acc (38) | reg R² (13) |
|---|---|---|
| tabUM v1.1 zero-shot, 1 pass | 0.8523 | 0.6284 |
| tabUM v1.1 zero-shot, ens8 | 0.8581 | 0.6519 |
| **tabUM v1.1 finetuned + ens8** | **0.8635** | **0.6907** |
| tabUM v1 (zero-shot, 1 pass) | 0.8509 | 0.6223 |
| logreg / linreg (fitted) | 0.8417 | 0.5597 |
| HistGB (fitted, default) | 0.8669 | 0.7091 |
| TabPFN v2 (zero-shot, ens.) | 0.8768 | 0.7471 |
| TabICL (zero-shot, 500M) | 0.8801 | n/a |

Win counts for finetuned tabUM: 20/38 vs HistGB (cls), 3/13 (reg);
4/38 vs TabPFN v2 and TabICL (cls).

## What each inference upgrade bought

- **Ensembling (8 permuted views — column order + class indices, averaged):**
  cls +0.6 pts, reg +2.4 pts, many-class letter (26c) 0.604 → 0.793
  (past fitted logreg's 0.740). Cost: 8× forward passes (<1s/dataset GPU).
- **finetune() (per-dataset ICL adaptation, ~11s/dataset, early-stopped):**
  further cls +0.5 pts, reg +3.9 pts; helped 28/51, hurt 2/51.

## Honest positioning

- Beats fitted linear baselines everywhere, in zero-shot or better.
- Finetuned+ensembled ties default HistGB on classification (20/38 wins,
  −0.3 pts mean) and closes regression to −1.8 pts (was −5.7 zero-shot 1-pass).
- Remains ~1.3–1.7 pts behind TabPFN v2 / TabICL (SOTA zero-shot) on
  classification and ~5.6 pts R² behind TabPFN v2 on regression. The gap is
  prior/architecture quality, not inference technique; closing it would be a
  research program (2D cell-grid attention, prior iteration), not a longer run.
- Distinctive card: 14M params (35× smaller than TabICL), MIT/clean-room,
  native NaN handling, 100 classes (TabPFN-class models cap at 10),
  validated context extrapolation to 64k rows, finetune() built in.

Full per-dataset table: `final_table.csv`. Checkpoint eval trajectory:
`eval_history.md`.
