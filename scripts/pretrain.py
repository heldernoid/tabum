"""Phase 4 pretraining entry point: continuous stream of freshly generated
synthetic tasks (never a fixed pre-generated set).

Run: uv run python scripts/pretrain.py --steps 200000 --batch-size 32
Resume: --resume checkpoints/stepXXXXXXXX.pt

Gate reminder (PLAN.md): do NOT launch a real run before the Phase 3 toy
overfit check passes on this same code path, and the Phase 1 generator
validation has been human-reviewed.
"""

import argparse

import torch

from tabum.generator import GeneratorConfig
from tabum.model import ModelConfig, TabUM
from tabum.train import TrainConfig, Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--cells-per-batch", type=int, default=400_000,
                    help="dynamic batch budget (rows x feature-groups x tasks); 0 = fixed batch-size")
parser.add_argument("--max-batch", type=int, default=64)
parser.add_argument("--rows-per-batch", type=int, default=131_072,
                    help="cap rows*batch (Stage-3 token count) — narrow tasks otherwise "
                         "reach full batch on the cell budget alone")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-rows", type=int, default=2048)
parser.add_argument("--min-rows", type=int, default=50)
parser.add_argument("--max-features", type=int, default=100)
parser.add_argument("--max-classes", type=int, default=10)
parser.add_argument("--max-cells", type=int, default=1_000_000,
                    help="cap rows*ceil(features/3) per task — bounds backward-pass memory")
parser.add_argument("--p-classification", type=float, default=0.7)
parser.add_argument("--p-cat-heavy", type=float, default=0.2)
parser.add_argument("--warmup-steps", type=int, default=2000)
parser.add_argument("--checkpoint-every", type=int, default=2000)
parser.add_argument("--mem-fraction", type=float, default=0.5,
                    help="hard CUDA allocator ceiling (fraction of device mem); "
                         "oversized batches OOM-and-skip instead of freezing the box")
parser.add_argument("--init-from", default=None,
                    help="checkpoint to warm-start MODEL WEIGHTS from (fresh optimizer/schedule)")
parser.add_argument("--checkpoint-dir", default="checkpoints")
parser.add_argument("--resume", default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
# The math SDPA backend materializes full attention matrices — tens of GB at
# 16k-row tasks, and on unified memory that is system RAM (froze/OOMed this
# box repeatedly). Fail loudly instead of falling back silently.
torch.backends.cuda.enable_math_sdp(False)
# cuDNN SDPA caches an execution plan + workspace per attention shape OUTSIDE
# the torch allocator — with our shape diversity that grows unboundedly on
# unified memory. Flash/mem-efficient backends only.
torch.backends.cuda.enable_cudnn_sdp(False)
cfg = TrainConfig(
    steps=args.steps,
    batch_size=args.batch_size,
    cells_per_batch=args.cells_per_batch,
    max_batch=args.max_batch,
    rows_per_batch=args.rows_per_batch,
    compile=args.compile,
    lr=args.lr,
    num_workers=args.num_workers,
    seed=args.seed,
    checkpoint_dir=args.checkpoint_dir,
    warmup_steps=args.warmup_steps,
    checkpoint_every=args.checkpoint_every,
    mem_fraction=args.mem_fraction,
    generator=GeneratorConfig(
        max_rows=args.max_rows, min_rows=args.min_rows,
        max_features=args.max_features, max_classes=args.max_classes,
        p_classification=args.p_classification, p_task_cat_heavy=args.p_cat_heavy,
        max_cells_per_task=args.max_cells,
    ),
)

from pathlib import Path

if args.init_from and Path(args.init_from).is_dir():
    # released weights (model.safetensors + config.json), e.g. a HuggingFace
    # snapshot: warm-start from those instead of a training checkpoint
    model = TabUM.from_pretrained(args.init_from)
    print(f"warm-started weights from released model {args.init_from}")
elif args.init_from:
    model = TabUM(ModelConfig())
    ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"warm-started weights from {args.init_from} (step {ckpt['step']})")
else:
    model = TabUM(ModelConfig())
print(f"model: {model.n_parameters()/1e6:.2f}M params")
print(f"device: {cfg.device}, amp: {cfg.amp_dtype}")

trainer = Trainer(model, cfg)
if args.resume:
    ckpt = torch.load(args.resume, map_location=cfg.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.opt.load_state_dict(ckpt["optimizer_state"])
    trainer.sched.load_state_dict(ckpt["scheduler_state"])
    trainer.step = ckpt["step"]
    print(f"resumed from {args.resume} at step {trainer.step}")

trainer.run()
