# Training walkthrough

Source: `src/tabum/train/loop.py`, `scripts/pretrain.py`.

## 1. The loop

`Trainer.run()` consumes an infinite `SyntheticTaskStream` (an IterableDataset
whose workers each own an independently seeded `TaskSampler`), so no task is
ever seen twice. Each batch is B same-shaped tasks; the loss is:

- classification: NLL of the retrieval head's probabilities on the test rows;
- regression: bar-distribution NLL (`reg_head.nll`) on train-standardized
  targets.

Optimizer: fused AdamW, betas (0.9, 0.95), cosine schedule with linear warmup
decaying to 5% of peak. bf16 autocast. Gradient clip 1.0.

Checkpoints (`save_checkpoint`) carry model weights, model config, optimizer
and scheduler state, step count, and the full train config including the
generator config: everything needed to reproduce or exactly resume a run.

`scripts/pretrain.py` wires CLI flags to `TrainConfig`/`GeneratorConfig`.
Two starting modes:

- `--init-from <ckpt.pt | safetensors dir>`: warm-start WEIGHTS only, fresh
  optimizer and schedule. Accepts a released HuggingFace snapshot directory.
  This is how v1.1 was trained from v1.
- `--resume <ckpt.pt>`: exact continuation, optimizer included.

## 2. Dynamic batching

`cells_per_batch` (default 400,000) is the memory budget per batch, measured
in `rows x ceil(features/3) x tasks`. The sampler sizes each batch to fill it
(see generator.md section 5), with two additional caps:

- `max_batch=64` tasks;
- `rows_per_batch=131,072` rows x tasks, bounding stage-3 token count. Do not
  remove this; narrow-feature tasks pass the cell budget at full batch and
  overwhelm stage 3 (a machine-freezing failure mode, found the hard way).

Collation (`collate_tasks`) truncates members to the shortest row count
(classification can drop unseen-class test rows) and quantizes the test
length to multiples of 64.

## 3. Memory safety on unified-memory hardware

v1.1 was trained on an NVIDIA DGX Spark (GB10): CPU and GPU share 128GB.
On such hardware a GPU memory leak IS a system RAM leak, and exhausting it
freezes the whole machine rather than killing one process. Eight training
launches produced the following defenses, all on by default; keep them unless
you know your hardware forgives you:

1. **Shape bucketing** (generator `_bucket`, train-size and test-length
   quantization): the CUDA caching allocator keeps a pool per tensor shape.
   With continuously novel shapes, reserved memory grows without bound.
   Bucketing makes shapes recur so pools get reused.
2. **cuDNN SDPA disabled** (`torch.backends.cuda.enable_cudnn_sdp(False)` in
   pretrain.py): cuDNN caches an execution plan and workspace PER ATTENTION
   SHAPE outside the torch allocator, invisible to `memory_reserved()`. With
   our shape diversity this grew to 100GB+. Flash and mem-efficient backends
   only. The math backend is also disabled: it materializes full attention
   matrices.
3. **Allocator ceiling** (`mem_fraction=0.5`,
   `torch.cuda.set_per_process_memory_fraction`): an oversized batch now
   raises a catchable `torch.OutOfMemoryError` instead of exhausting the
   machine mid-step, where no external watchdog can see it.
4. **OOM-skip**: `Trainer.run()` catches that OOM, drops the batch, logs it,
   and continues. A handful of skips per run is normal; a storm of them means
   your budgets are wrong.
5. **empty_cache valve**: reserved memory above 30GB triggers
   `torch.cuda.empty_cache()` after the step.
6. **Run under a cgroup** if you can:
   `systemd-run --user --property=MemoryMax=80G ...` plus
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The log line prints both `mem` (torch reserved) and `drv` (driver-level used,
from `torch.cuda.mem_get_info`). Healthy runs keep `drv` within ~10GB of
`mem`; a growing gap means something outside the allocator is leaking, which
is exactly how defense 2 was found.

## 4. Monitoring

`scripts/train_status.py` parses `train.log`: loss curve, steps/s, ETA
(auto-detects total steps from the live process), memory telemetry, and
checkpoint inventory. `--plot` writes a loss curve image. Loss is noisy
across windows because the task mixture varies (a 100-class window reads
worse than a binary window at equal model quality); judge trends over
thousands of steps, and judge quality by real-data evals of checkpoints, not
by loss.

## 5. Reproducing v1.1

v1 (from scratch, about 10h on the GB10):

    uv run python scripts/pretrain.py --steps 50000 --seed 0

v1.1 (warm-start, wider envelope, about 7h to step 20000):

    uv run python scripts/pretrain.py --steps 30000 \
      --init-from <v1 checkpoint or snapshot> \
      --max-rows 16384 --max-features 2000 --max-classes 100 \
      --max-cells 500000 --p-classification 0.5 --p-cat-heavy 0.3 \
      --lr 1.5e-4 --warmup-steps 1000 --seed 11 --checkpoint-every 500

The released v1.1 is step 20000 of that run; the curve on the evaluation
suite was flat from step 7500, so training was stopped early (results in
`results/eval_history.md`).

## 6. Hardware requirements

- v1.1 envelope defaults: about 50GB of GPU-addressable memory.
- v1 envelope (`--max-rows 2048 --max-features 100 --max-cells 250000`):
  fits in about 24GB.
- Below that, training this architecture is not practical; inference runs
  in under 2GB (see inference.md).

`--compile` (torch.compile) is available and measured about +30% throughput
once shapes are bucketed, but was not used for the released runs; treat it
as experimental with respect to memory behavior.
