"""Lab 07: back-of-the-envelope numbers that decide real training and serving budgets.

mise run -q lab:scale
mise run -q lab:scale -- --tok-per-s 11000      # plug in the throughput you measured
"""

from __future__ import annotations

import argparse

# name: (params, layers, kv_heads, head_dim, training tokens)
MODELS = {
    "this lab": (2.56e6, 6, 2, 32, 3.28e6),
    "Llama 3 8B": (8.0e9, 32, 8, 128, 15e12),
    "Llama 3 70B": (70.6e9, 80, 8, 128, 15e12),
}


def human(n: float, unit: str = "") -> str:
    for div, suffix in [
        (1e24, "Y"),
        (1e21, "Z"),
        (1e18, "E"),
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
    ]:
        if abs(n) >= div:
            return f"{n / div:.1f} {suffix}{unit}"
    return f"{n:.0f} {unit}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok-per-s", type=float, default=11_000, help="your measured training throughput")
    parser.add_argument("--gpu-tflops", type=float, default=400, help="sustained bf16 TFLOP/s per H100")
    args = parser.parse_args()

    print("training compute ~ 6 x params x tokens (2 for the forward pass, 4 for backward)\n")
    print(f"{'model':<12} {'params':>9} {'tokens':>9} {'tok/param':>9} {'FLOPs':>10} {'H100-days':>10}")
    for name, (n, _, _, _, d) in MODELS.items():
        flops = 6 * n * d
        gpu_days = flops / (args.gpu_tflops * 1e12) / 86_400
        print(f"{name:<12} {human(n):>9} {human(d):>9} {d / n:>9.0f} {human(flops):>10} {gpu_days:>10,.2f}")

    n = MODELS["this lab"][0]
    achieved = 6 * n * args.tok_per_s
    print(f"\nthis lab at {args.tok_per_s:,.0f} tok/s does {achieved / 1e9:.2f} GFLOP/s of model maths")
    print("(attention and the 512-way softmax add more; the M1 GPU peaks near 2,600 GFLOP/s fp32)")
    for name, (_, _, _, _, d) in list(MODELS.items())[1:]:
        years = MODELS[name][0] * d / (n * args.tok_per_s) / 3.15e7
        print(f"at that FLOP rate, {name} would take {years:,.0f} years on this Mac")

    print("\n(Meta reports 1.46 M H100-hours for Llama 3 8B: about 3x this ideal, since real")
    print(" clusters reach 40% or less of peak and runs restart after failures)")
    print("\nKV cache = 2 (K and V) x layers x kv_heads x head_dim x 2 bytes (bf16), per token")
    print(f"{'model':<12} {'per token':>10} {'8k ctx':>9} {'128k ctx':>9} {'128k x 32 users':>16}")
    for name, (_, layers, kv, hd, _) in list(MODELS.items())[1:]:
        per_tok = 2 * layers * kv * hd * 2
        print(
            f"{name:<12} {human(per_tok, 'B'):>10} {human(per_tok * 8192, 'B'):>9} "
            f"{human(per_tok * 131072, 'B'):>9} {human(per_tok * 131072 * 32, 'B'):>16}"
        )


if __name__ == "__main__":
    main()
