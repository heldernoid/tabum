# Synthetic prior walkthrough

Source: `src/tabum/generator/`. The generator IS the training data: the model
never sees a real table during pretraining, so everything it knows about
"tableness" was put there by this code. If you want to improve the model,
this is the highest-leverage place to work (see ideas.md).

## 1. Flow of one task

`TaskSampler.sample()` in `sampler.py`:

1. `sample_task_spec(cfg, rng)` draws a `TaskSpec`: shape (rows, features,
   noise features), task type, class count, family, split mode, train
   fraction, seed. All randomness downstream derives from that seed, so a
   task is reproducible from `(config, spec)`.
2. The family generator produces `(X, y_cont, latent)`: informative features
   and a continuous target. All families return a continuous target;
   classification is derived from it afterwards, so both task types share one
   code path.
3. `postprocess(X, spec, cfg, rng)` makes the table look real (section 3).
4. Target conversion: regression keeps `y_cont`; classification quantile-bins
   it with Dirichlet-sampled class proportions (realistic imbalance) and then
   permutes class ids so they carry no ordinal information.
5. Split: rows are REORDERED so train rows come first (no index arrays).
   Usually a random permutation; with probability `p_shift_split=0.18` rows
   are sorted by the latent variable instead, simulating temporal or
   covariate-shift splits. Train size is quantized to multiples of 64 above
   128 rows (allocator shape reuse, see training.md).
6. Classification guards: every class must appear in the train split; test
   rows with train-unseen classes are dropped and labels are remapped to a
   dense range (this mirrors what real evaluation protocols do).

A rejection loop retries degenerate draws (constant target, one class, etc.)
up to 40 times. When a spec is supplied (batch members share one shape), only
the seed is redrawn on retry; shapes are honored, batches rely on this.

## 2. Families (`scm.py`, `gp.py`, `tree.py`)

Mixture weights: SCM 0.6, GP 0.2, tree 0.2 (`w_scm`, `w_gp`, `w_tree`).

- **SCM** (`scm.py`): samples a random DAG, evaluates structural equations
  `X_i = f_i(parents) + noise` with several DAG samplers, combiner functions,
  and exogenous noise families (gaussian, uniform, student-t). Observed
  features and the target are nodes of the graph, so features have genuine
  causal correlation structure.
- **GP** (`gp.py`): draws smooth functions via random Fourier features, giving
  tasks with continuous nonlinear response surfaces.
- **tree** (`tree.py`): random decision-tree ensembles generate the target,
  giving axis-aligned, discontinuous decision structure.

Each family returns a `latent` scalar per row used for shift splits.

## 3. Postprocessing (`postprocess.py`)

Applied in order:

1. **Noise features**: up to `max_noise_feature_frac=0.4` of the informative
   count, standard normal, appended then shuffled in with the rest by a final
   column permutation. Teaches the model to ignore irrelevant columns.
2. **Categorical conversion**: per task, the categorical fraction is bimodal
   (matched to a validation against 32 real OpenML datasets):
   `p_task_all_numeric=0.45` all-numeric, `p_task_cat_heavy=0.2` with cat
   fraction uniform in (0.5, 1.0), otherwise uniform in (0.05, 0.4).
   A column becomes categorical by quantile-binning with cardinality
   `2 + (max-2) * u^5` (skewed low, real-data median is about 3), then the
   codes are permuted so categories are non-ordinal, exactly like the output
   of `to_numeric` at inference.
3. **Missingness**: gated per task (`p_task_has_missing=0.25`, most real
   tables are fully observed), then per column (`p_missing_col=0.3`, rate up
   to 0.3). 40% of missing columns are MAR (missingness depends on another
   column's values), the rest MCAR. This is why the model can treat
   missingness as signal.
4. **Outliers** (`p_outlier_col=0.15`, cell rate 0.01) and **scale/shift
   distortion** (`p_scale_col=0.5`): real tables are not standardized; stage 0
   must undo this, and training data has to exercise that.

## 4. Shape envelope and cost caps (`config.py`)

`GeneratorConfig` holds every knob. Shape-related fields interact with
training memory and are worth understanding before changing:

- `min_rows/max_rows`, `min_features/max_features`, `max_classes`: sampled
  log-uniformly, so large shapes are a rare tail, not the norm. v1.1 used
  16,384 rows / 2,000 features / 100 classes via CLI overrides.
- `_bucket()`: rows and features snap to a half-octave grid (steps of x1.41).
  Without this, every batch had a unique tensor shape and the CUDA caching
  allocator grew without bound (a hard-won lesson, see training.md).
- `max_cells_per_task`: caps `rows x ceil(features/3)`; when exceeded, the
  feature count is reduced. Bounds stage-1 activation memory per task.
- The class count is additionally capped so a k-class task has enough train
  rows to exhibit k classes meaningfully.

## 5. Batching (`sampler.py`)

`sample_batch_budget(cell_budget, max_batch, row_budget)` draws ONE spec and
fills a batch with reseeded copies, so members collate into a dense tensor:

- batch size is `cell_budget / (rows x groups)` clipped to `[1, max_batch]`,
  so small tasks get large batches and huge tasks run alone;
- `row_budget` separately caps `rows x batch`. This is essential: the cell
  budget measures stage-1 cost, but stage-3 cost is `rows x batch`. A
  2-feature task has 1 feature group and passes the cell budget at full
  batch, putting roughly 30x the usual token count through stage 3. Before
  this cap existed, exactly such a batch (64 x 5760 rows x 2 features)
  exhausted a 128GB machine.

## 6. Validating changes to the prior

`scripts/validate_generator.py` compares generated tables against 32 real
OpenML datasets on summary statistics (cardinalities, missingness, skew,
correlation structure). Run it after any distribution change. The deeper
validation is always the same: pretrain a small model and benchmark it; the
prior is only as good as the model it produces.
