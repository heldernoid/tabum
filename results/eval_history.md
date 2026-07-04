# v1.1 checkpoint eval history

Continued pretraining from v1 step50000 (attempt 8, seed 11, 30k steps).
Envelope: 16,384 rows / 2,000 features / 100 classes / 50-50 cls-reg.
Protocol: scripts/eval_real.py holdout (≤2000 train / ≤1000 test, seed 0)
plus letter (did 6, 26 classes) as the many-class probe.

References — v1 step50000: holdout mean cls acc **0.8426**, cpu_act R² **0.932**,
letter **0.294** (logreg 0.733–0.740 depending on split libs).

| step | mean cls acc | cpu_act R² | letter (26c) | notes |
|------|-------------|-----------|--------------|-------|
| 1500 | 0.8099 | 0.9022 | **0.402** | expected transient dip on holdout (fresh optimizer, shifted task mix); letter +0.108 over v1 — many-class learning emerging |
| 4500 | 0.8232 | 0.9213 | **0.511** | holdout recovering toward v1 (0.8426); letter +0.217 over v1 and climbing; wdbc 0.930 / splice 0.913 already at v1 level |
| 7500 | 0.8417 | 0.9041 | **0.561** | holdout fully recovered to v1 level (0.8426); splice 0.928 matches logreg exactly; letter nearly doubled vs v1 |
| 11000 | 0.8345 | **0.9341** | 0.568 | cpu_act R² now above v1 (0.932); holdout cls oscillating at v1 level; letter plateauing near 0.56–0.57 |
| 14000 | 0.8333 | 0.8818 | 0.584 | letter resumed climbing (+0.016); cpu_act noisy (0.90→0.93→0.88 band); holdout cls steady ~0.834 |
| 20000 | **0.8521** | 0.9333 | **0.604** | FINAL (user stopped run here, resumable): best holdout cls of the run, above v1 (0.8426); letter more than doubled vs v1; cpu_act at v1 level |

**Run stopped at step 20000 by user decision (2026-07-03 ~21:00)** — TabArena results
showed zero-shot SOTA parity (TabPFN v2) out of reach at this scale; checkpoint
step00020000.pt is the v1.1 final artifact. Resume later with
`--resume checkpoints/v1.1/step00020000.pt` if ever desired.

Per-dataset at step 1500: mfeat-zernike 0.632, optdigits 0.765, splice 0.912,
pendigits 0.811, wdbc 0.930; AUCs all ≥0.94.

## Run log
- Attempts 1–6 failed on memory (see git log + STATUS.md); attempt 6 froze the
  machine at step ~1500 on 2026-07-03 11:27 (driver NV_ERR_NO_MEMORY): narrow-task
  batches (e.g. 64×5760×2) pass the cell budget at full batch → ~30× Stage-3 tokens.
- Attempt 8 (2026-07-03 15:41, commit 9af0221): mem-fraction 0.5 ceiling +
  OOM-skip + rows×batch ≤ 131k + checkpoints every 500 steps. Zero OOM skips
  through step 1550; drv peak 46G of 119G.
