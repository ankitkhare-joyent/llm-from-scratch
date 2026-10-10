"""Dataset download, tokenisation, and batching.

TinyShakespeare (about 1.1 MB of plays) is small enough to train on a laptop in minutes
and big enough that the model must learn real structure: speaker names, verse, and
spelling. The file is pinned by SHA-256 so a changed upstream can't silently change
the lab.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tinyllm.config import DATA_DIR, ModelConfig
from tinyllm.tokenizer import BPETokenizer

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
TEXT_PATH = DATA_DIR / "tinyshakespeare.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
VAL_FRACTION = 0.1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: Path = TEXT_PATH) -> Path:
    """Fetch the corpus once and verify its checksum."""
    if path.exists() and _sha256(path) == DATASET_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATASET_URL}")
    with (
        tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp,
        urllib.request.urlopen(DATASET_URL, timeout=60) as resp,
    ):
        tmp.write(resp.read())
    tmp_path = Path(tmp.name)
    if (digest := _sha256(tmp_path)) != DATASET_SHA256:
        tmp_path.unlink()
        raise RuntimeError(f"checksum mismatch for {DATASET_URL}: got {digest}")
    tmp_path.replace(path)
    return path


def load_tokenizer(vocab_size: int = ModelConfig().vocab_size) -> BPETokenizer:
    """Load the trained tokenizer, training it first if it doesn't exist yet."""
    if TOKENIZER_PATH.exists():
        tok = BPETokenizer.load(TOKENIZER_PATH)
        if tok.vocab_size == vocab_size:
            return tok
    print(f"training a {vocab_size}-token BPE tokenizer")
    tok = BPETokenizer.train(download().read_text(encoding="utf-8"), vocab_size)
    tok.save(TOKENIZER_PATH)
    return tok


def prepare(tok: BPETokenizer) -> tuple[Path, Path]:
    """Tokenise the corpus into train/val token files (uint16, one id per token)."""
    train_path, val_path = DATA_DIR / "train.bin", DATA_DIR / "val.bin"
    stamp = DATA_DIR / "bins.stamp"
    key = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
    if train_path.exists() and val_path.exists() and stamp.exists() and stamp.read_text() == key:
        return train_path, val_path
    ids = np.array(tok.encode(download().read_text(encoding="utf-8")), dtype=np.uint16)
    split = int(len(ids) * (1 - VAL_FRACTION))
    # Split by position, not randomly: the val set is unseen *later* text, which is
    # how the model will be used, and it avoids leaking overlapping windows.
    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)
    stamp.write_text(key)
    print(f"tokenised {len(ids):,} tokens -> train {split:,} / val {len(ids) - split:,}")
    return train_path, val_path


@dataclass
class TokenStream:
    """A flat sequence of token ids that yields random (input, target) windows."""

    tokens: torch.Tensor  # 1-D int64 on CPU

    @classmethod
    def from_file(cls, path: Path) -> TokenStream:
        return cls(torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64)))

    def batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Targets are the inputs shifted left by one: position t predicts token t+1.
        starts = torch.randint(0, len(self.tokens) - seq_len - 1, (batch_size,), generator=generator)
        offsets = torch.arange(seq_len + 1)
        windows = self.tokens[starts[:, None] + offsets]  # (B, T+1)
        x, y = windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def main() -> None:
    path = download()
    tok = load_tokenizer()
    train_path, val_path = prepare(tok)
    for p in (path, TOKENIZER_PATH, train_path, val_path):
        print(f"{p.relative_to(DATA_DIR.parent)}  {p.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
