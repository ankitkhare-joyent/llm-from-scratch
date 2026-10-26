"""Short training runs for the drills, with optional deliberate breakage.

Takes every flag `python -m tinyllm.train` takes, with lab-sized defaults
(100 steps, eval every 25), plus these switches:

    --copy-targets   targets are the inputs themselves (an off-by-one bug)           Lab 02
    --no-rope        skip rotary embeddings entirely                                  Lab 03
    --head-std X     initialise lm_head with std X instead of 0.001                   Lab 05

    mise run -q lab:train -- --run lab-base
    mise run -q lab:train -- --run lab-norope --no-rope
"""

from __future__ import annotations

import sys

import torch

import tinyllm.model as model_mod
import tinyllm.train as train_mod
from tinyllm.data import TokenStream


def main() -> None:
    argv = sys.argv[1:]

    def take(flag: str, has_value: bool = False):
        if flag not in argv:
            return None
        i = argv.index(flag)
        value = argv[i + 1] if has_value else True
        del argv[i : i + (2 if has_value else 1)]
        return value

    if take("--copy-targets"):
        original = TokenStream.batch

        def copy_batch(self, *a, **kw):
            x, _ = original(self, *a, **kw)
            return x, x.clone()

        TokenStream.batch = copy_batch
        print("BREAK: targets = inputs")
    if take("--no-rope"):
        model_mod.apply_rope = lambda x, cos, sin: x
        print("BREAK: no rotary embeddings")
    if (std := take("--head-std", has_value=True)) is not None:
        original_reset = model_mod.GPT.reset_parameters

        @torch.no_grad()
        def reset(self):
            original_reset(self)
            torch.nn.init.normal_(self.lm_head.weight, std=float(std))

        model_mod.GPT.reset_parameters = reset
        print(f"BREAK: lm_head init std {std}")

    defaults = {"--max-steps": "100", "--eval-interval": "25", "--run": "lab"}
    for flag, value in defaults.items():
        if flag not in argv:
            argv += [flag, value]
    sys.argv = [sys.argv[0], *argv]
    train_mod.main()


if __name__ == "__main__":
    main()
