# Architecture walkthrough

Source: `src/tabum/model/`. Total: 14.31M parameters (`ModelConfig()` defaults).

## 1. The task tensor

Everything the model consumes is a single batch of same-shaped tasks:

- `x`: `(B, N, F)` float32. Raw feature values, `NaN` where missing.
  Rows `[0:train_size]` are the training (context) rows, the rest are test rows.
  This ordering convention replaces any index arrays and is relied on everywhere.
- `y_train`: `(B, train_size)`. Labels for the context rows only. Test labels
  never enter the model (there is a regression test asserting this,
  `tests/test_invariance.py`).
- `task_type`: `"classification"` or `"regression"`, one per batch.
- `groups`: `(G, 3)` int tensor assigning features to triplets (see stage 1).

`TabUM.forward(x, y_train, train_size, task_type, n_classes, groups)` returns
`{"probs"}` for classification or `{"logits", "y_mean", "y_std"}` for
regression. `TabUM.embed_rows(...)` exposes stages 0 to 3 without the heads,
used by the label-leakage test and by `explain()`.

## 2. Stage 0: statistics and grouping (`preprocessing.py`)

No learned parameters.

- `standardize(x, train_size)`: per-column z-score using statistics computed
  from the TRAIN rows only (never the test rows, that would leak). Returns the
  standardized tensor and a NaN mask; NaN cells are zero-filled after masking
  so downstream math is finite, and the mask itself is an input feature.
- `build_groups(F, seed)`: features are packed into triplets. A "cell token"
  in stage 1 is one row's triplet of values plus their NaN indicators, so the
  token count per row is `ceil(F/3)`, three times fewer than one-token-per-cell.
  During training the grouping is reshuffled per batch (seeded) so the model
  never learns a fixed feature adjacency; at inference it is deterministic.
- `standardize_target(y, train_size)`: z-scores regression targets by train
  statistics, returning `(y_std, y_mean, y_std_dev)` so predictions can be
  mapped back to the original scale.

## 3. Stage 1: cell embedding and column attention (`stage1.py`)

Per row, each feature triplet becomes a token:

- `FourierFeatures` (in `layers.py`) embeds each standardized value with a set
  of sin/cos frequencies plus the raw value, which gives the network
  resolution at multiple scales without binning.
- The NaN indicators are embedded and added.
- Label information for TRAIN rows is embedded and added to every cell token
  of that row (class embedding table of size `max_classes=100` for
  classification, a linear map of the standardized target for regression).
  A `train_mask` gate guarantees test rows receive a zero label embedding.

Then `stage1_blocks=2` `InducingBlock`s run attention across the column axis.
Direct all-pairs column attention would be quadratic in `ceil(F/3)`, so each
block first summarizes columns into `n_inducing=48` learned inducing tokens
and then lets every cell token attend to the summaries. Important detail: the
inducing summaries are computed from TRAIN rows only (`x[:, :train_size]`),
so no test-row statistics can flow into other rows through this path.

Width at this stage: `d_stage1=128`.

## 4. Stage 2: row aggregation (`stage2.py`)

Each row's variable-length set of cell tokens must become one fixed-width row
vector. `n_cls_tokens=4` learned CLS tokens cross-attend over the row's cell
tokens; their concatenated outputs are projected to `d_model=320`. Attention
here is batched over `B x N` rows; the SDPA batch axis is chunked at 32,768
because CUDA kernels have a grid-dimension limit (see `Attention.forward`).

## 5. Stage 3: the ICL transformer (`stage3.py`)

`stage3_blocks=10` self-attention blocks (8 heads) over the row axis, where
the in-context learning actually happens. The causal structure of ICL
("train rows are public, test rows are private") is implemented structurally,
with no attention masks:

- keys and values are sliced to the first `kv_len=train_size` rows, so EVERY
  row (train or test) attends to train rows only;
- train rows therefore see each other bidirectionally;
- test rows never attend to themselves or to other test rows, which makes
  predictions independent of test-set composition and lets inference chunk
  the test rows freely (`test_chunk` in the estimators).

Mask-free slicing is also faster than masked attention and keeps kernel
shapes recurring, which matters for allocator behavior (see training.md).

`Attention` applies QASSMax query scaling: per-head learned scale multiplied
by `log(context_length)`, initialized at 0.15. This is what lets a model
trained on at most 16k-row contexts extrapolate (validated to 64k rows,
accuracy still improving with context). Norms are `RMSNorm` throughout.

## 6. Heads (`heads.py`)

**RetrievalClassifier.** Projects test-row embeddings to queries and train-row
embeddings to keys (`head_dim`), then class probabilities are the
softmax-attention-weighted average of one-hot train labels, with a learned
temperature. Properties worth preserving in any refactor:

- non-parametric in class count: 2 or 100 classes use the same weights;
- the attention matrix is materialized per test-row chunk (`chunk=2048`),
  never in full: a full `(N_test x N_train)` matrix at large sizes exceeds
  system memory on unified-memory hardware;
- `top_neighbors()` returns the exact attention weights and indices, which is
  what makes `explain()`'s neighbor view faithful rather than post-hoc.

**BarDistributionRegressor.** An MLP maps each test-row embedding to logits
over `n_reg_bins=100` fixed bins spanning `[-reg_support, +reg_support]` in
standardized-target space; the outermost bins act as half-open tails. One
forward pass yields the full distribution; `mean()` and `quantile()` decode
point estimates and arbitrary quantiles, and `nll()` is the training loss
(cross-entropy against the bin containing the target).

## 7. Parameter budget

`ModelConfig()` defaults: d_stage1 128, n_inducing 48, stage1_blocks 2,
n_cls_tokens 4, d_model 320, stage3_blocks 10, stage3_heads 8, n_reg_bins 100,
max_classes 100. Stage 3 dominates the count. `ModelConfig.toy()` builds a
tiny variant used by the test suite.

## 8. Known architectural limitation

Rows are compressed to a single 320-dim vector in stage 2, and all ICL
reasoning afterwards is row-to-row over those compressed vectors. Cell-level
detail is unavailable to stage 3, so feature interactions that only become
relevant in context cannot be recomputed there. State-of-the-art competitors
keep a 2D grid of cell states alive through the whole network with alternating
row/column attention, which is more powerful and more expensive. See ideas.md.
