# Development guide

## 1. Environment

The project uses [uv](https://docs.astral.sh/uv/) exclusively (no pip, no
manual venvs) and pins every dependency to an exact version:

    uv sync                  # default groups: dev (pytest) + validation (pandas, matplotlib)
    uv run pytest tests -q   # 18 tests, a few seconds
    uv run jupyter lab       # notebooks (group installed on demand)

Python 3.13+. torch 2.12.x. When adding a dependency, pin an exact version
that has been on PyPI for at least 14 days, and avoid GPL/non-commercial
licenses; the released package must stay MIT-clean.

## 2. Repository conventions

- `src/tabum/` is the package: `generator/`, `model/`, `train/`, `inference/`.
  Scripts under `scripts/` are entry points and experiments; nothing in
  `src/` may import from `scripts/`.
- Reproducibility is a hard rule: every source of randomness flows from an
  explicit seed (`GeneratorConfig` + seed reproduces a training stream;
  checkpoints carry full state; ensemble views are seeded).
- The row-ordering convention (train rows first, `train_size` marks the
  boundary) is global. No index arrays, anywhere.
- Test labels must never influence anything: `tests/test_invariance.py`
  injects garbage at test-label positions and asserts identical outputs.
  Any change to stage 0/1 or the estimators should keep that test meaningful.
- Comment style: comments state constraints the code cannot express
  (why a chunk size exists, why a backend is disabled), not what the next
  line does.

## 3. Tests

`tests/` covers three areas:

- `test_generator.py`: task validity, spec-shape stability under retries,
  batch shape sharing;
- `test_shapes.py`: forward-pass shapes across task types and sizes (uses
  `ModelConfig.toy()`);
- `test_invariance.py`: label-leakage guard, estimator equivalences
  (chunked vs unchunked, ensembled vs single when k=1).

Run the suite before and after any change (`uv run pytest tests -q`). It is
fast on purpose; there is no excuse to skip it. Occasional SIGBUS crashes of
the test process on aarch64 unified-memory machines are a known environment
flake: rerun once before investigating.

For model-behavior changes, the fast real-data probe is
`scripts/eval_real.py --checkpoint <ckpt>` (six datasets, about a minute),
and the real gate is the TabArena sweep (evaluation.md section 3).

## 4. Hardware notes

Development happened on an NVIDIA DGX Spark (GB10, aarch64, 128GB unified
memory). Consequences you will inherit:

- GPU memory and system RAM are the same pool; a GPU leak can freeze the
  machine. Read training.md section 3 before touching training memory
  behavior, and keep the defenses on.
- Long-running work should run under systemd with a memory cap:
  `systemd-run --user --property=MemoryMax=80G ...` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Anything CUDA-shape-related (attention backends, batching, bucketing) was
  tuned for allocator pool reuse; novel-shape-per-batch patterns will bite.

## 5. Clean-room policy

The model, generator, and training code were written without consulting the
source of TabPFN, TabICL, AutoGluon, or TabFM. Keep it that way: competitor
packages may be RUN as black-box baselines (that is what
`scripts/eval_baselines.py` does), but do not port their code or copy their
implementations. Published papers are fine to learn from; repositories of the
above projects are not.

## 6. Release procedure

1. Export weights: `uv run python scripts/export_safetensors.py
   --checkpoint <ckpt.pt> --out release/<version>` (writes model.safetensors
   + config.json and verifies a bit-exact round trip).
2. Upload the folder to HuggingFace with a model card carrying the current
   results table (the card must state protocol and limitations; keep the
   honest-positioning section).
3. Bump `version` in `pyproject.toml`; tag the repo.
4. Re-execute `notebooks/release_v1_1.ipynb` end to end against the uploaded
   weights before announcing; the notebook doubles as the release smoke test.

## 7. Where to start contributing

Read `docs/README.md` for the tour order, then `ideas.md` for the open
problems ranked by expected impact. Small, well-scoped first contributions:
generator postprocessing performance on wide tasks, additional evaluation
suites, and explain() visualizations.
