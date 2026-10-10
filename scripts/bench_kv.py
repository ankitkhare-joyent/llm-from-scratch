"""Compare generation speed with and without the KV cache (lesson 06).

uv run python scripts/bench_kv.py [--tokens 250] [--device cpu]
"""

from __future__ import annotations

import argparse
import time

import torch

from tinyllm.config import CHECKPOINT_DIR, pick_device
from tinyllm.sample import generate, load_model, sample_next


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=250)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = load_model(CHECKPOINT_DIR / "shakespeare.pt", pick_device(args.device))
    device = next(model.parameters()).device
    window = model.cfg.seq_len

    start = time.perf_counter()
    cached = list(generate(model, [1], args.tokens, temperature=0))
    with_cache = time.perf_counter() - start

    # Naive decoding: re-run the whole (growing) context for every new token.
    ids = [1]
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.tokens):
            logits, _ = model(torch.tensor([ids[-window:]], device=device))
            ids.append(sample_next(logits[0, -1], temperature=0))
    without_cache = time.perf_counter() - start

    n = args.tokens
    print(f"with KV cache   : {with_cache:5.2f}s  {n / with_cache:6.0f} tok/s")
    print(f"without KV cache: {without_cache:5.2f}s  {n / without_cache:6.0f} tok/s")
    print(f"speed-up        : {without_cache / with_cache:.1f}x")
    # Past the context window the two strategies keep different amounts of history.
    k = min(n, window - 1)
    print(f"greedy outputs identical for the first {k} tokens: {cached[:k] == ids[1 : 1 + k]}")


if __name__ == "__main__":
    main()
