"""One-shot training status: progress, loss trend, throughput, ETA, GPU, checkpoints.

Usage:
  uv run python scripts/train_status.py                 # status snapshot
  uv run python scripts/train_status.py --plot          # also write validation/loss_curve.png
  watch -n 60 uv run python scripts/train_status.py     # live dashboard
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--log", default="train.log")
parser.add_argument("--total-steps", type=int, default=0,
                    help="0 = auto-detect from the running pretrain.py command line")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

if not args.total_steps:  # read --steps from the live process, else default
    cmdline = subprocess.run(["pgrep", "-af", "scripts/pretrain.py"],
                             capture_output=True, text=True).stdout
    m = re.search(r"--steps[= ](\d+)", cmdline)
    args.total_steps = int(m.group(1)) if m else 50_000

pat = re.compile(r"step\s+(\d+)\s+loss\s+(-?[\d.]+)\s+lr\s+([\d.e+-]+)\s+([\d.]+) steps/s")
steps, losses, rates = [], [], []
for line in Path(args.log).read_text().splitlines():
    m = pat.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        rates.append(float(m.group(4)))

if not steps:
    raise SystemExit(f"no training lines parsed from {args.log}")

step = steps[-1]
rate = np.median(rates[-20:])
remaining = max(0, args.total_steps - step)
eta = dt.datetime.now() + dt.timedelta(seconds=remaining / rate)


def window(n):
    return float(np.mean(losses[-n:])) if len(losses) >= 1 else float("nan")


print(f"step        : {step:,} / {args.total_steps:,} ({step/args.total_steps:5.1%})")
print(f"loss        : last {losses[-1]:.4f} | mean last-10 logs {window(10):.4f} "
      f"| last-100 {window(100):.4f} | first-10 {np.mean(losses[:10]):.4f}")
print(f"throughput  : {rate:.2f} steps/s (median of recent logs)")
print(f"ETA         : {eta:%Y-%m-%d %H:%M} ({remaining/rate/3600:.1f} h remaining)")

ckpts = sorted(Path("checkpoints").glob("step*.pt"))
print(f"checkpoints : {len(ckpts)}" + (f", latest {ckpts[-1].name}" if ckpts else ""))

try:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    print(f"gpu         : {gpu}")
except Exception:
    pass

alive = subprocess.run(["pgrep", "-f", "scripts/pretrain.py"], capture_output=True, text=True)
print(f"process     : {'RUNNING (pid ' + alive.stdout.split()[0] + ')' if alive.stdout.strip() else 'NOT RUNNING'}")

if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, losses, lw=0.6, alpha=0.4, color="C0")
    if len(losses) > 20:
        k = max(1, len(losses) // 100)
        smooth = np.convolve(losses, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        ax.plot(steps[k:-k], smooth, lw=1.8, color="C0")
    ax.set_xlabel("step")
    ax.set_ylabel("mixed cls/reg loss")
    ax.set_title(f"tabUM pretraining — step {step:,}")
    ax.grid(alpha=0.3)
    Path("validation").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("validation/loss_curve.png", dpi=120)
    print("plot        : validation/loss_curve.png")
