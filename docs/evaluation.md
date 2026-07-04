# Evaluation walkthrough

Source: `scripts/eval_benchmark.py`, `scripts/eval_baselines.py`,
`scripts/eval_real.py`, `scripts/build_benchmark_suite.py`. Results:
`results/`.

## 1. The protocol

Every number in the README table comes from one protocol, identical for every
model:

- at most 2,000 training rows and 1,000 test rows per dataset (70/30 when the
  dataset is smaller);
- one split, `random_state=0`, stratified for classification when possible;
- inputs through `to_numeric`; NaNs left in for models that accept them,
  mean-imputed and standardized for the linear baselines;
- classification metric: accuracy (AUC also recorded); regression metric: R2;
- test rows with train-unseen classes are dropped (all models see the same
  filtered test set).

Single-split evaluation is a deliberate trade: it makes a 51-dataset sweep
cheap enough to run per checkpoint, at the cost of per-dataset variance.
Conclusions should be drawn from suite-level means and win counts, not from
any single dataset's number.

## 2. The suite

TabArena-v0.1 (51 datasets: 38 classification, 13 regression), pinned by
OpenML dataset id in `validation/tabarena_suite.json` (in the development
repo). The runners are resumable: results stream to CSV one row per dataset,
recorded ids are skipped on restart, so interrupted sweeps continue where
they left off. Failures are recorded as rows with a `status` message rather
than aborting the sweep.

## 3. Running tabUM on the suite

    uv run python scripts/eval_benchmark.py \
      --checkpoint <ckpt.pt> --suite <suite.json> --out results.csv \
      --ensemble 8            # optional test-time ensembling
      --finetune              # optional per-dataset finetune() before predicting

`eval_real.py` is the quick 6-dataset health probe used between checkpoints
during training (5 classification datasets + one regression), useful as a
fast smoke test after model changes.

## 4. Baselines and competitors

`eval_baselines.py --model {histgb,tabpfn,tabicl}` runs the same protocol:

- `histgb`: sklearn HistGradientBoosting, default hyperparameters, NaN-native
  so it receives the same raw matrix tabUM does. This is the "strong
  classical" reference; fitted logistic/linear regression (computed inside
  `eval_benchmark.py`) is the floor, not the bar.
- `tabpfn`, `tabicl`: installed as black-box pip packages and run on the same
  splits. No competitor source code was consulted (clean-room constraint);
  only their public APIs are called. TabPFN's newer weight versions are gated
  behind a browser license acceptance; the ungated v2 weights were used
  (`TABPFN_MODEL_VERSION=v2`), with `ignore_pretraining_limits=True` which
  only lifts its CPU speed guard (the protocol is inside its supported
  range). TabICL is classification-only.

Fairness notes worth repeating in any publication: TabPFN ensembles multiple
preprocessed views by default (comparable to our `n_ensemble=8` mode, which
is what the README table reports for tabUM's ensembled rows), and both
competitors are much better funded research artifacts; the honest headline is
where tabUM sits relative to them at 14M parameters and MIT license, not a
claim of parity.

## 5. Reading the results files

- `results/final_table.csv`: per-dataset scores for every model column
  (`zs_1pass`, `zs_ens8`, `ft_ens8`, `v1`, `histgb`, `tabpfn`, `tabicl`,
  `linear`), joined on OpenML dataset id.
- `results/v1_1_results.md`: the headline table plus what each inference
  upgrade contributed.
- `results/eval_history.md`: checkpoint-by-checkpoint trajectory during the
  v1.1 run (holdout accuracy, regression R2, and a 26-class probe), plus the
  training-incident log.

## 6. Adding a new evaluation

Copy the `eval_one` pattern: fetch by dataset id, `to_numeric`, split with
the protocol constants, compute the metric, append a CSV row. Keep the
protocol constants (`MAX_TRAIN`, `MAX_TEST`, seed) imported or duplicated
exactly; the value of the whole results directory rests on every row having
been produced by the same split.
