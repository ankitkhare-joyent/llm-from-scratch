"""Lab 06: the model's next-token distribution, and what temperature and top-p do to it.

mise run -q lab:next -- "ROMEO:\nWhat light through yonder"
mise run -q lab:next -- --temps 0.5 1 2 "KING"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from tinyllm.config import CHECKPOINT_DIR
from tinyllm.data import load_tokenizer
from tinyllm.sample import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "shakespeare.pt")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    model = load_model(args.checkpoint, torch.device("cpu"))
    tok = load_tokenizer(model.cfg.vocab_size)
    ids = tok.encode(args.prompt.replace("\\n", "\n"))
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0][0, -1]
    print(
        f"prompt {tok.decode(ids)!r} ({len(ids)} tokens); logits range "
        f"{logits.min():.2f}..{logits.max():.2f} (soft-cap {model.cfg.logit_softcap:g})\n"
    )
    for temp in args.temps:
        probs = (logits / temp).softmax(-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        sorted_probs = probs.sort(descending=True).values
        nucleus = int((sorted_probs.cumsum(-1) < 0.9).sum()) + 1
        top = probs.topk(args.top)
        print(
            f"temperature {temp:g}: entropy {entropy:.2f} nats (= choosing among ~{math.exp(entropy):.0f}),"
            f" top-p 0.9 keeps {nucleus} tokens"
        )
        for p, i in zip(top.values, top.indices, strict=True):
            print(f"   {p:6.1%}  {tok.decode([int(i)])!r}")
        print()


if __name__ == "__main__":
    main()
