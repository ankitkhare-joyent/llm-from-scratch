"""Lab 02: one training batch, decoded, plus the data budget arithmetic.

mise run -q lab:batch
mise run -q lab:batch -- --seed 3 --show 12
"""

from __future__ import annotations

import argparse

import torch

from tinyllm.config import DATA_DIR, ModelConfig, TrainConfig
from tinyllm.data import TokenStream, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=10, help="positions to print")
    args = parser.parse_args()
    mc, tc = ModelConfig(), TrainConfig()
    tok = load_tokenizer(mc.vocab_size)
    train = TokenStream.from_file(DATA_DIR / "train.bin")
    val = TokenStream.from_file(DATA_DIR / "val.bin")

    gen = torch.Generator().manual_seed(args.seed)
    x, y = train.batch(tc.batch_size, mc.seq_len, torch.device("cpu"), gen)
    print(f"x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
    print(f"y equals x shifted left by one: {torch.equal(x[:, 1:], y[:, :-1])}\n")
    print(f"{'pos':>3}  {'input x':<14} {'target y':<14}")
    for t in range(args.show):
        xi, yi = int(x[0, t]), int(y[0, t])
        print(f"{t:>3}  {tok.decode([xi])!r:<14} {tok.decode([yi])!r:<14}")

    n_train, n_val = len(train.tokens), len(val.tokens)
    per_step = tc.batch_size * tc.grad_accum_steps * mc.seq_len
    seen = per_step * tc.max_steps
    print(f"\ntrain tokens {n_train:,}   val tokens {n_val:,}   val share {n_val / (n_train + n_val):.1%}")
    print(f"tokens per step {per_step:,}   steps {tc.max_steps}   tokens seen {seen:,}")
    print(f"epochs {seen / n_train:.2f}   predictions per window {mc.seq_len}")


if __name__ == "__main__":
    main()
