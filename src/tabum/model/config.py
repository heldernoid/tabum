from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    # Stage 1
    d_stage1: int = 128
    n_inducing: int = 48
    stage1_blocks: int = 2
    stage1_heads: int = 4
    n_fourier_freqs: int = 6
    max_classes: int = 100  # label-embedding table size (classification)
    # Stage 2
    n_cls_tokens: int = 4
    stage2_blocks: int = 2
    stage2_heads: int = 4
    # Stage 3
    d_model: int = 320
    stage3_blocks: int = 10
    stage3_heads: int = 8
    # Heads
    n_reg_bins: int = 100
    reg_support: float = 5.0  # bar-distribution support: [-reg_support, reg_support] in std-y units
    head_dim: int = 128  # retrieval-head projection dim

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def toy() -> "ModelConfig":
        return ModelConfig(
            d_stage1=64,
            n_inducing=16,
            stage1_blocks=1,
            stage2_blocks=1,
            n_cls_tokens=2,
            d_model=128,
            stage3_blocks=3,
            stage3_heads=4,
            n_reg_bins=32,
            head_dim=64,
        )
