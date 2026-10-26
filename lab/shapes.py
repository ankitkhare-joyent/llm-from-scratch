"""Lab 00: print every tensor shape on the way through the model, plus where the parameters live.

mise run -q lab:shapes                      # default ModelConfig, batch 2 x 16 tokens
mise run -q lab:shapes -- --n-head 5        # any ModelConfig field can be overridden
"""

from __future__ import annotations

import argparse
import dataclasses

import torch

from tinyllm.config import ModelConfig
from tinyllm.model import GPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    for f in dataclasses.fields(ModelConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)
    args = vars(parser.parse_args())
    names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in args.items() if k in names and v is not None})

    torch.manual_seed(0)
    model = GPT(cfg).eval()
    print(f"config: {cfg}\n")

    watch = {
        "wte (token embedding)": model.wte,
        "block0.attn.c_q": model.blocks[0].attn.c_q,
        "block0.attn.c_k": model.blocks[0].attn.c_k,
        "block0.attn.c_v": model.blocks[0].attn.c_v,
        "block0.attn (output)": model.blocks[0].attn,
        "block0.mlp.c_fc": model.blocks[0].mlp.c_fc,
        "block0.mlp (output)": model.blocks[0].mlp,
        f"block{cfg.n_layer - 1} (output)": model.blocks[-1],
        "lm_head (logits)": model.lm_head,
    }
    for label, module in watch.items():
        module.register_forward_hook(
            lambda _m, _i, out, label=label: print(f"  {label:<24} {tuple(out.shape)}")
        )
    idx = torch.randint(0, cfg.vocab_size, (args["batch"], args["seq"]))
    print(f"input ids                  {tuple(idx.shape)}")
    with torch.no_grad():
        logits, _ = model(idx)
    print(f"returned logits            {tuple(logits.shape)}\n")

    groups = {"wte": model.wte.weight.numel(), "lm_head": model.lm_head.weight.numel()}
    groups["attention (all blocks)"] = sum(p.numel() for b in model.blocks for p in b.attn.parameters())
    groups["mlp (all blocks)"] = sum(p.numel() for b in model.blocks for p in b.mlp.parameters())
    total = model.num_params()
    for name, n in groups.items():
        print(f"{name:<24} {n:>10,}  {100 * n / total:5.1f}%")
    print(f"{'total':<24} {total:>10,}")
    kv_bytes = 2 * cfg.n_layer * cfg.n_kv_head * cfg.head_dim * 4
    print(
        f"\nKV cache per token (fp32): {kv_bytes:,} bytes; full {cfg.seq_len}-token window: "
        f"{kv_bytes * cfg.seq_len / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
