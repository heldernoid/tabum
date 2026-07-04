# Known limitations and promising directions

Ranked by expected impact on accuracy per unit of effort, based on what the
v1.1 evaluation actually showed (results/v1_1_results.md). The measured gaps:
1.3 to 1.7 accuracy points behind TabPFN v2 / TabICL on classification and
about 5.6 R2 points on regression, with the biggest per-dataset losses on
large, low-noise, structured datasets.

## 1. Prior iteration (highest impact, ongoing effort)

The synthetic prior is a first build that was validated on summary statistics
and then trained on twice. State-of-the-art priors are the product of many
iterations of: change the task distribution, pretrain a small model, and
benchmark on real data. The loop costs about a day per cycle on a single
strong GPU and compounds. Concrete threads to pull:

- regression is the weaker half (the eval battery grades 50/50, and
  regression is where boosting stays clearly ahead): target-noise models,
  heteroscedasticity, and target-vs-feature scale diversity are all
  underexplored in the current prior;
- interaction-heavy tabular structure (the in_vehicle_coupon / churn style
  losses): the SCM family's combiner functions could sample deeper
  feature-interaction patterns;
- per-family win/loss attribution: run the benchmark with models pretrained
  on each family alone to learn which family teaches what.

## 2. Architecture: keep cell states alive (high impact, a rebuild)

Stage 2 compresses each row to one 320-dim vector; stage 3 reasons only
row-to-row (architecture.md section 8). The strongest competitors alternate
attention across rows and across features on a 2D grid of PER-CELL states for
the whole depth of the network, so feature interactions can be recomputed in
context at every layer. A tabUM v2 on that design would keep the same
retrieval/bar-distribution heads and the same generator, and would need a
fresh pretraining run. Parameter count is not the obstacle (TabPFN v2 class
models are around 11M); memory and speed engineering is where the work lives.

## 3. Inference-time search (medium impact, cheap)

Ensembling and finetune() are implemented; natural extensions:

- preprocessing-variant views in the ensemble (quantile-transform, log-scale
  for skewed columns) alongside permutation views;
- auto_preprocess: k candidate encodings scored on a held-out slice of the
  training rows, one forward pass each, pick the winner (fit is free, so this
  is nearly free insurance for pathological datasets);
- ensemble weighting by per-view validation score instead of uniform
  averaging.

## 4. Regression head upgrades (medium impact, contained)

The bar distribution is trained with per-bin cross-entropy only. Candidates:
distributional regularization across neighboring bins, more bins with learned
edges, or a small mixture-density head. Any change here is cheap to evaluate:
the 13 TabArena regression datasets plus the quantile-coverage check in the
notebook.

## 5. Many-class beyond 100 (low effort, niche)

The retrieval head is non-parametric in class count; the 100 limit comes only
from the label-embedding table (`max_classes`). Raising it costs parameters
linearly and needs generator support for high-class tasks with enough rows
per class. Nothing else changes.

## 6. Text-aware columns (research direction)

String columns are factorized to unordered codes; semantics are discarded
("low/medium/high" loses its order, city names lose their geography). Feeding
string columns through a frozen text encoder and giving stage 1 embedded
values would fix this at the cost of a large dependency and a new pretraining
data question (how to synthesize realistic text-valued columns). Out of scope
for the current design, but it is the most-requested capability in this model
class.

## 7. Engineering debts worth paying

- torch.compile: measured about +30% training throughput after shape
  bucketing, but not used for released runs; needs a memory-behavior
  qualification pass on unified-memory hardware.
- generator postprocessing is numpy-loop-heavy for very wide tasks; profile
  shows it can bottleneck the data workers at 2,000 features.
- the evaluation runner is single-process; a work-stealing parallel version
  would make prior-iteration cycles (idea 1) faster.

## What NOT to spend time on

- More pretraining steps on the current prior and architecture: the v1.1 run
  was stopped at 20k of 30k steps because the benchmark curve had been flat
  since step 7500. The remaining gap is not a training-duration problem.
- Inference tricks as a substitute for prior/architecture work: ensembling
  and finetune() together bought about half the distance to gradient
  boosting; the measured remainder is model quality.
- Growing parameters without changing the architecture: 14M is not the
  binding constraint (see idea 2).
