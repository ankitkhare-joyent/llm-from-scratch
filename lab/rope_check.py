"""Lab 03: check numerically that RoPE scores depend only on relative distance.

mise run -q lab:rope
mise run -q lab:rope -- --base 100
"""

from __future__ import annotations

import argparse

import torch

from tinyllm.model import apply_rope, rope_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=float, default=10_000.0)
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)
    cos, sin = rope_tables(256, args.dim, args.base)
    q, k = torch.randn(args.dim), torch.randn(args.dim)

    def at(v: torch.Tensor, pos: int) -> torch.Tensor:
        return apply_rope(v.view(1, 1, 1, -1), cos[pos : pos + 1], sin[pos : pos + 1]).view(-1)

    print(f"head_dim={args.dim} base={args.base:g}")
    print(f"|q| before {q.norm():.4f}  after rotating to pos 77: {at(q, 77).norm():.4f}\n")
    print(f"{'query pos m':>11} {'key pos n':>9} {'m - n':>5}  score q·k")
    for m, n in [(5, 2), (50, 47), (200, 197), (5, 4), (50, 10), (200, 160)]:
        print(f"{m:>11} {n:>9} {m - n:>5}  {at(q, m) @ at(k, n):+.5f}")
    print(f"\nno rotation (plain dot product): {q @ k:+.5f}")

    slow = 1.0 / (args.base ** ((args.dim - 2) / args.dim))
    fast = 1.0
    print(f"fastest pair turns {fast:.3f} rad/token (one lap every {6.2832 / fast:.0f} tokens)")
    print(f"slowest pair turns {slow:.5f} rad/token (one lap every {6.2832 / slow:,.0f} tokens)")


if __name__ == "__main__":
    main()
