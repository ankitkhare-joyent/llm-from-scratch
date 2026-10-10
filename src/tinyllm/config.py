"""Configuration objects and device selection.

Every knob that changes the model or the training run lives here, so a checkpoint
can record exactly what produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# src/tinyllm/config.py -> repository root
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the network. Defaults give roughly 3M parameters."""

    vocab_size: int = 512
    seq_len: int = 256  # context window the model is trained on
    n_layer: int = 6
    n_head: int = 6  # query heads
    n_kv_head: int = 2  # key/value heads; < n_head means grouped-query attention
    n_embd: int = 192  # residual stream width
    mlp_ratio: int = 4
    rope_base: float = 10_000.0
    logit_softcap: float = 15.0  # 0 disables the tanh soft cap

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings. Defaults finish in a few minutes on an M1."""

    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 400
    warmup_steps: int = 30
    warmdown_ratio: float = 0.4  # last 40% of steps decay the LR linearly
    final_lr_frac: float = 0.0
    optimizer: str = "muon"  # "muon" (Muon + AdamW) or "adamw"
    matrix_lr: float = 0.02  # Muon LR for 2-D weights inside blocks
    embed_lr: float = 0.05  # AdamW LR for the token embedding
    head_lr: float = 0.004  # AdamW LR for the output head
    adamw_lr: float = 3e-3  # single LR used when optimizer == "adamw"
    weight_decay: float = 0.1  # decoupled; applied to Muon and AdamW groups alike
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 1337
    device: str = "auto"


def pick_device(requested: str = "auto") -> torch.device:
    """CUDA if present, then Apple's Metal (MPS), then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
