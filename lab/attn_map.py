"""Lab 04: what each attention head in the trained model looks at.

mise run -q lab:attn                                   # every layer and head, averaged over a passage
mise run -q lab:attn -- --layer 0 --text "ROMEO:\nBut, soft! what light"   # one layer, token by token
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from tinyllm.config import CHECKPOINT_DIR
from tinyllm.data import TEXT_PATH, load_tokenizer
from tinyllm.model import apply_rope, norm
from tinyllm.sample import load_model


def attention_weights(model, idx: torch.Tensor) -> list[torch.Tensor]:
    """Recompute each layer's softmax(q·kᵀ/√d) with the causal mask: list of (H, T, T)."""
    captured: list[torch.Tensor] = []
    hooks = [
        b.attn.register_forward_pre_hook(lambda _m, args: captured.append(args[0])) for b in model.blocks
    ]
    with torch.no_grad():
        model(idx)
    for h in hooks:
        h.remove()
    T = idx.size(1)
    cos, sin = model.cos[:T], model.sin[:T]
    out = []
    for block, x in zip(model.blocks, captured, strict=True):
        a = block.attn
        q = a.c_q(x).view(1, T, a.n_head, a.head_dim).transpose(1, 2)
        k = a.c_k(x).view(1, T, a.n_kv_head, a.head_dim).transpose(1, 2)
        q, k = norm(apply_rope(q, cos, sin)), norm(apply_rope(k, cos, sin))
        k = k.repeat_interleave(a.n_head // a.n_kv_head, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(a.head_dim)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        out.append(scores.masked_fill(~mask, float("-inf")).softmax(-1)[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "shakespeare.pt"))
    parser.add_argument("--text", default=None, help="defaults to a 256-token passage from the corpus")
    parser.add_argument("--layer", type=int, default=None, help="show one layer token by token")
    args = parser.parse_args()
    model = load_model(Path(args.checkpoint), torch.device("cpu"))
    tok = load_tokenizer(model.cfg.vocab_size)
    if args.text is None:
        ids = tok.encode(TEXT_PATH.read_text(encoding="utf-8")[100_000:102_000])[: model.cfg.seq_len]
    else:
        ids = tok.encode(args.text.replace("\\n", "\n"))
    maps = attention_weights(model, torch.tensor([ids]))

    T = len(ids)
    rows = torch.stack([w.sum(-1) for w in maps])
    future = max(w.triu(1).abs().max().item() for w in maps)
    print(
        f"check: every row sums to 1 ({rows.min():.4f}..{rows.max():.4f}); "
        f"max weight on a future token {future:.1e}\n"
    )

    if args.layer is None:
        # What a head spreading attention evenly over the visible tokens would put on t-1.
        uniform = sum(1 / (t + 1) for t in range(1, T)) / (T - 1)
        print(f"{T} tokens. Mean weight each head puts on self (t), the previous token (t-1) and the")
        print(f"first token (0). A head attending evenly would put {uniform:.3f} on t-1 (xunif = prev/that).")
        print("argmax=prev: share of positions whose strongest key is t-1.\n")
        print(
            f"{'layer':>5} {'head':>4} {'kv':>2}  {'self':>5} {'prev':>5} {'first':>5} {'xunif':>5} "
            f"{'argmax=prev':>11}"
        )
        group = model.cfg.n_head // model.cfg.n_kv_head
        best = (0.0, 0, 0)
        for layer, w in enumerate(maps):
            for h in range(w.size(0)):
                prev = w[h].diagonal(-1).mean().item()
                hits = (w[h, 1:].argmax(-1) == torch.arange(T - 1)).float().mean().item()
                best = max(best, (prev, layer, h))
                print(
                    f"{layer:>5} {h:>4} {h // group:>2}  {w[h].diagonal().mean():5.2f} {prev:5.2f} "
                    f"{w[h, 1:, 0].mean():5.2f} {prev / uniform:5.1f} {hits:>11.0%}"
                )
            print()
        print(f"most previous-token-like head: layer {best[1]} head {best[2]} ({best[0]:.2f} on t-1)")

    if args.layer is not None:
        w = maps[args.layer]
        pieces = [tok.decode([i]) for i in ids]
        print(f"\nlayer {args.layer}: for each token, the token each head attends to most")
        for t, piece in enumerate(pieces):
            best = [pieces[int(w[h, t].argmax())] for h in range(w.size(0))]
            print(f"{t:>3} {piece!r:<10} " + " ".join(f"{b!r:<9}" for b in best))


if __name__ == "__main__":
    main()
