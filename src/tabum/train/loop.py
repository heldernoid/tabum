"""Training loop: generator stream -> model -> mixed cls/reg loss.

Each batch is B tasks sharing one shape and task type (sampled per batch), so
they collate into dense tensors. Losses: NLL of retrieved class probabilities
for classification, discretized-CDF (bar-distribution) NLL for regression.

Reproducibility: checkpoints carry model config, generator config, optimizer
state, step count, and the data seed — everything needed to restart a run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..generator import GeneratorConfig, SyntheticTask, TaskSampler
from ..model import ModelConfig, TabUM
from ..model.preprocessing import build_groups


@dataclass
class TrainConfig:
    steps: int = 100_000
    batch_size: int = 32  # used when cells_per_batch == 0 (fixed-size batches)
    cells_per_batch: int = 400_000  # dynamic batch budget: rows x feature-groups x tasks
    max_batch: int = 64
    rows_per_batch: int = 131_072  # cap rows x batch (Stage-3 tokens); v1's stable worst case
    compile: bool = False  # torch.compile(dynamic=True) on the forward
    lr: float = 3e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 8
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype: str = "bfloat16"  # GB10-appropriate; "float32" to disable autocast
    checkpoint_every: int = 2000
    log_every: int = 50
    checkpoint_dir: str = "checkpoints"
    # Hard allocator ceiling as a fraction of device memory. On GB10 unified
    # memory an unbounded step can exhaust the *system* (driver OOM at
    # 2026-07-03 11:27 froze the box mid-step, invisible to any watchdog);
    # with a ceiling the step raises torch.OutOfMemoryError instead, which
    # run() catches by skipping the batch. 0 disables.
    mem_fraction: float = 0.5
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)


@dataclass
class Batch:
    x: torch.Tensor  # (B, N, F)
    y_train: torch.Tensor  # (B, n_train)
    y_test: torch.Tensor  # (B, N - n_train)
    train_size: int
    task_type: str
    n_classes: int
    groups: torch.Tensor

    def to(self, device) -> "Batch":
        return Batch(
            self.x.to(device, non_blocking=True),
            self.y_train.to(device, non_blocking=True),
            self.y_test.to(device, non_blocking=True),
            self.train_size, self.task_type, self.n_classes,
            self.groups.to(device, non_blocking=True),
        )


def collate_tasks(tasks: list[SyntheticTask], group_seed: int | None = None) -> Batch:
    """Tasks share a spec but classification may drop unseen-class test rows, so
    row counts can differ — truncate to the shortest (drops test rows only,
    train rows always come first and have equal count)."""
    n = min(t.X.shape[0] for t in tasks)
    train_size = tasks[0].train_size
    n_test = n - train_size
    if n_test > 64:  # quantize test length: recurring shapes for the allocator
        n = train_size + (n_test // 64) * 64
    task_type = tasks[0].task_type
    x = torch.from_numpy(np.stack([t.X[:n] for t in tasks]))
    y = np.stack([t.y[:n] for t in tasks])
    y_t = torch.from_numpy(y)
    n_classes = max(t.n_classes for t in tasks) if task_type == "classification" else 0
    return Batch(
        x=x,
        y_train=y_t[:, :train_size],
        y_test=y_t[:, train_size:],
        train_size=train_size,
        task_type=task_type,
        n_classes=n_classes,
        groups=build_groups(x.shape[2], seed=group_seed),
    )


class SyntheticTaskStream(IterableDataset):
    """Yields fully-collated batches; each DataLoader worker gets its own seed."""

    def __init__(self, gen_cfg: GeneratorConfig, batch_size: int, seed: int,
                 cells_per_batch: int = 0, max_batch: int = 64,
                 rows_per_batch: int = 0):
        self.gen_cfg = gen_cfg
        self.batch_size = batch_size
        self.seed = seed
        self.cells_per_batch = cells_per_batch
        self.max_batch = max_batch
        self.rows_per_batch = rows_per_batch

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        sampler = TaskSampler(self.gen_cfg, seed=self.seed * 10_000 + wid)
        rng = np.random.default_rng(self.seed * 10_000 + wid + 1)
        while True:
            if self.cells_per_batch > 0:
                tasks = sampler.sample_batch_budget(self.cells_per_batch,
                                                    max_batch=self.max_batch,
                                                    row_budget=self.rows_per_batch)
            else:
                tasks = sampler.sample_batch(self.batch_size)
            yield collate_tasks(tasks, group_seed=int(rng.integers(0, 2**31 - 1)))


def compute_loss(model: TabUM, batch: Batch) -> torch.Tensor:
    out = model(
        batch.x, batch.y_train, batch.train_size, batch.task_type,
        n_classes=batch.n_classes or None, groups=batch.groups,
    )
    if batch.task_type == "classification":
        probs = out["probs"].clamp(min=1e-9)
        return F.nll_loss(probs.log().reshape(-1, probs.shape[-1]), batch.y_test.reshape(-1))
    y_std = (batch.y_test - out["y_mean"]) / out["y_std"]
    return model.reg_head.nll(out["logits"], y_std)


class Trainer:
    def __init__(self, model: TabUM, cfg: TrainConfig):
        self.raw_model = model.to(cfg.device)  # uncompiled ref: state_dict/checkpoints
        self.model = (
            torch.compile(self.raw_model, dynamic=True) if cfg.compile else self.raw_model
        )
        self.cfg = cfg
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=(cfg.device == "cuda"),
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)
        self.step = 0
        if cfg.mem_fraction > 0 and cfg.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction)

    def _lr_lambda(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return (step + 1) / c.warmup_steps
        t = (step - c.warmup_steps) / max(1, c.steps - c.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))

    def train_step(self, batch: Batch) -> float:
        c = self.cfg
        self.model.train()
        batch = batch.to(c.device)
        amp = c.amp_dtype != "float32" and c.device == "cuda"
        with torch.autocast(c.device, dtype=getattr(torch, c.amp_dtype), enabled=amp):
            loss = compute_loss(self.model, batch)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
        self.opt.step()
        if c.device == "cuda" and torch.cuda.memory_reserved() > 30e9:
            torch.cuda.empty_cache()  # safety valve against allocator bloat
        self.sched.step()
        self.step += 1
        return float(loss.detach())

    def run(self):
        c = self.cfg
        stream = SyntheticTaskStream(c.generator, c.batch_size, seed=c.seed,
                                     cells_per_batch=c.cells_per_batch,
                                     max_batch=c.max_batch,
                                     rows_per_batch=c.rows_per_batch)
        loader = DataLoader(
            stream, batch_size=None, num_workers=c.num_workers,
            pin_memory=(c.device == "cuda"),
            persistent_workers=c.num_workers > 0,
            prefetch_factor=2 if c.num_workers > 0 else None,
        )
        t0, losses = time.perf_counter(), []
        for batch in loader:
            try:
                loss = self.train_step(batch)
            except torch.OutOfMemoryError:
                # a single oversized batch must not take down the run (or, on
                # unified memory, the machine) — drop it and keep going
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(
                    f"step {self.step:>7d}  OOM: skipped batch "
                    f"x{tuple(batch.x.shape)} train_size={batch.train_size}",
                    flush=True,
                )
                self.sched.step()
                self.step += 1
                continue
            losses.append(loss)
            if self.step % c.log_every == 0:
                dt = time.perf_counter() - t0
                if c.device == "cuda":
                    drv_free, drv_total = torch.cuda.mem_get_info()
                    mem = (f"  mem {torch.cuda.memory_reserved()/1e9:.1f}G"
                           f"  drv {(drv_total - drv_free)/1e9:.1f}G")
                else:
                    mem = ""
                print(
                    f"step {self.step:>7d}  loss {np.mean(losses):.4f}  "
                    f"lr {self.sched.get_last_lr()[0]:.2e}  "
                    f"{c.log_every / dt:.2f} steps/s{mem}",
                    flush=True,
                )
                t0, losses = time.perf_counter(), []
            if self.step % c.checkpoint_every == 0:
                self.save_checkpoint()
            if self.step >= c.steps:
                break
        self.save_checkpoint()

    def save_checkpoint(self, path: str | Path | None = None):
        d = Path(self.cfg.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else d / f"step{self.step:08d}.pt"
        torch.save(
            {
                "model_state": self.raw_model.state_dict(),
                "model_config": self.raw_model.cfg.to_dict(),
                "optimizer_state": self.opt.state_dict(),
                "scheduler_state": self.sched.state_dict(),
                "step": self.step,
                "train_config": {**asdict(self.cfg), "generator": asdict(self.cfg.generator)},
            },
            path,
        )
        print(f"checkpoint saved: {path}", flush=True)

    @staticmethod
    def load_model(path: str | Path, device: str = "cpu") -> TabUM:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = TabUM(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state"])
        return model.to(device).eval()
