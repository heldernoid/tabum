"""Export a training checkpoint to a HuggingFace-ready directory:
model.safetensors (weights only) + config.json (architecture parameters).

Run: uv run python scripts/export_safetensors.py \
        --checkpoint checkpoints/v1.1/step00020000.pt --out release/v1.1
Verify: reloads the export and checks logits match the original checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from tabum.model import ModelConfig, TabUM


def load_exported(export_dir: str | Path, device: str = "cpu") -> TabUM:
    """Rebuild a TabUM from an exported directory (the HF loading path)."""
    return TabUM.from_pretrained(export_dir, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = {k: v.contiguous() for k, v in ckpt["model_state"].items()}
    save_file(state, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({
        "model_type": "tabum",
        "model_config": ckpt["model_config"],
        "trained_steps": ckpt["step"],
        "n_parameters": sum(v.numel() for v in state.values()),
    }, indent=2) + "\n")

    # round-trip verification on a synthetic task
    model = load_exported(out)
    ref = TabUM(ModelConfig(**ckpt["model_config"]))
    ref.load_state_dict(ckpt["model_state"])
    ref.eval()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, 96, 7)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 3, 64).astype(np.int64)).unsqueeze(0)
    with torch.inference_mode():
        a = model(x, y, 64, "classification", n_classes=3)["probs"]
        b = ref(x, y, 64, "classification", n_classes=3)["probs"]
    assert torch.equal(a, b), "exported weights diverge from checkpoint"
    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"exported {args.checkpoint} -> {out}/ ({size_mb:.1f} MB, "
          f"{json.loads((out / 'config.json').read_text())['n_parameters']/1e6:.2f}M params, "
          f"round-trip verified)")


if __name__ == "__main__":
    main()
