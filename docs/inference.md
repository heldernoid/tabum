# Inference walkthrough

Source: `src/tabum/inference/`.

## 1. Estimators (`estimator.py`)

`TabUMClassifier` and `TabUMRegressor` share `_BaseEstimator`. Construction
takes either a live `TabUM` (`model=`), a training checkpoint path
(`checkpoint=`), or nothing (untrained weights, tests only). For released
weights: `TabUM.from_pretrained(dir)` then pass `model=`.

- `fit(X, y)`: stores the data. No weight updates, no statistics; instant.
  The classifier additionally label-encodes `y` and records `classes_`.
- `predict / predict_proba / predict_quantile`: one forward pass per test
  chunk. Test rows are processed in chunks of `test_chunk=8192` and results
  concatenated; this is safe because test rows never attend to each other
  (architecture.md section 5), so chunking is mathematically invisible.
- Inputs are numeric numpy arrays; `NaN` means missing and is welcome.
  DataFrames go through `to_numeric` first (section 4).

## 2. Test-time ensembling (`n_ensemble`)

`n_ensemble=k` runs k views of the same data and averages probabilities.
View 0 is the identity; views 1..k-1 permute column order (which changes the
stage-1 feature-triplet grouping) and, for classification, class indices
(which changes label-embedding assignment). Both are choices the model should
be invariant to but is not perfectly, so each view carries the same signal
with different arbitrary-choice noise, and the average cancels the noise.

Implementation notes: class permutations are inverted before averaging
(`probs_view[..., class_perm]` gathers back to original class order);
regression averages bar-distribution probabilities across views and re-enters
them as `log(mean_prob)`, which is exact because softmax of log-probabilities
is the identity. Permutations are seeded, so results are reproducible.

Measured effect (TabArena, 51 datasets): +0.6 accuracy points classification,
+2.4 R2 points regression; on a 26-class problem (letter) +19 points, because
class-index noise grows with class count. Cost is k forward passes.
`n_ensemble=8` is the recommended default; 1 is the fastest; beyond 16 the
gains are marginal.

## 3. finetune()

`finetune(X, y, steps=300, lr=3e-5, val_frac=0.15, patience=6, ...)` adapts a
CLONE of the model to one dataset:

1. deep-copies the model (the base weights are never modified);
2. holds out `val_frac` of the provided training rows (never used as
   prediction targets during finetuning);
3. each step, randomly splits the remaining rows 80/20 into pseudo-context
   and pseudo-targets and trains on the same ICL objective as pretraining;
   classification targets whose class is absent from the pseudo-context are
   masked out of the loss;
4. every `eval_every` steps computes validation loss (predicting the held-out
   slice from the rest); the best state is kept and training stops after
   `patience` non-improving evaluations.

About 10-20 seconds per dataset on a GPU. Measured on TabArena: helped 28/51
datasets, hurt 2, mean +1.4 points overall and +3.9 R2 on regression.
Composes with ensembling (`.finetune(...)` then predictions use the views as
usual). The test-set rows must never be passed to `finetune`, that is
leakage; give it exactly what you would give `fit`.

## 4. Encoding (`encoding.py`)

`to_numeric(df)` converts a DataFrame column-wise: numeric passes through;
strings/categoricals factorize to integer codes with missing mapped to NaN;
datetimes become epoch seconds; timedeltas become seconds. Categorical codes
carry no order, which matches how the generator builds categorical columns,
so this encoding is exactly the model's native language. Free-text and
ID-like columns become near-unique codes (pure noise); drop them.

## 5. explain()

Two methods, both faithful by construction rather than post-hoc
approximations, validated in `scripts/validate_explain.py` against synthetic
tasks with known informative/noise features (ranking AUC 0.99 regression,
0.94 median classification):

- `feature_importances(X_test)` (both estimators): for each feature, remove
  the column from BOTH the stored context and the test rows, re-predict, and
  measure the drop (predicted-class probability for classification, mean
  absolute prediction shift in train-std units for regression). This is
  leave-one-covariate-out, ordinarily priced at one retraining per feature,
  but free here because fit() is storage-only. Caveat: correlated features
  split credit, as in every perturbation method.
- classifier `explain(X_test, top_k=5)` additionally returns the retrieval
  head's top-k attended training rows per test row (`neighbor_index`,
  `neighbor_weight`, `neighbor_label`). These weights ARE the votes the
  prediction was computed from. Read them correctly: over hundreds of context
  rows attention is diffuse, so the top-k are the strongest individual votes
  while the predicted probability is the tally of all of them; expect top-1
  neighbor and prediction to agree often but not always.

## 6. Memory behavior at inference

Inference is light: under 2GB for typical benchmark sizes, and validated to
64k-row contexts in about 4.7GB. Two guards exist because of unified-memory
hardware: test-row chunking (`test_chunk`) and the retrieval head's internal
chunking (architecture.md section 6). Keep both if you touch this code; both
were added after real machine freezes.
