"""Pre-training loop: next-token prediction on TinyShakespeare.

    python -m tinyllm.train                     # defaults from TrainConfig
    python -m tinyllm.train --optimizer adamw   # compare against plain AdamW
    python -m tinyllm.train --max-steps 300     # quick smoke run

Writes checkpoints/<run>.pt (best validation loss) and runs/<run>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from tinyllm.config import CHECKPOINT_DIR, RUNS_DIR, ModelConfig, TrainConfig, pick_device
from tinyllm.data import TokenStream, load_tokenizer, prepare
from tinyllm.model import GPT


def build_optimizers(model: GPT, tc: TrainConfig) -> list[torch.optim.Optimizer]:
    """Muon for the 2-D weight matrices inside blocks, AdamW for embeddings and head.

    Muon orthogonalises each matrix update (Newton-Schulz iterations), so every
    direction in the weight matrix moves by a similar amount. It only makes sense for
    matrices that act on hidden activations; the embedding (a lookup table) and the
    output head stay on AdamW, as in nanochat and modded-nanogpt.
    """
    if tc.optimizer == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=tc.adamw_lr, betas=(0.9, 0.95), weight_decay=tc.weight_decay
        )
        return [opt]
    if tc.optimizer != "muon":
        raise ValueError(f"unknown optimizer {tc.optimizer!r}")
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2]
    adamw = torch.optim.AdamW(
        [
            {"params": [model.wte.weight], "lr": tc.embed_lr},
            {"params": [model.lm_head.weight], "lr": tc.head_lr},
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=tc.weight_decay,
    )
    muon = torch.optim.Muon(matrices, lr=tc.matrix_lr, momentum=0.95, weight_decay=tc.weight_decay)
    return [adamw, muon]


def lr_multiplier(step: int, tc: TrainConfig) -> float:
    """Warmup -> stable -> linear warmdown (the "WSD" schedule).

    WSD replaced cosine decay in many recent recipes: the stable phase can run as long
    as you like and the short warmdown at the end delivers most of the final gain.
    """
    if step < tc.warmup_steps:
        return (step + 1) / tc.warmup_steps
    warmdown_start = int(tc.max_steps * (1 - tc.warmdown_ratio))
    if step < warmdown_start:
        return 1.0
    progress = (step - warmdown_start) / max(1, tc.max_steps - warmdown_start)
    return 1.0 - progress * (1.0 - tc.final_lr_frac)


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, tc: TrainConfig, device: torch.device) -> float:
    """Mean loss over a fixed set of batches, so successive evals are comparable."""
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(tc.eval_batches):
        x, y = stream.batch(tc.batch_size, model.cfg.seq_len, device, gen)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(mc: ModelConfig, tc: TrainConfig, run_name: str) -> Path:
    torch.manual_seed(tc.seed)
    device = pick_device(tc.device)
    tok = load_tokenizer(mc.vocab_size)
    train_path, val_path = prepare(tok)
    train_data, val_data = TokenStream.from_file(train_path), TokenStream.from_file(val_path)

    model = GPT(mc).to(device)
    optimizers = build_optimizers(model, tc)
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    tokens_per_step = tc.batch_size * tc.grad_accum_steps * mc.seq_len
    print(f"device={device.type} params={model.num_params():,} train_tokens={len(train_data.tokens):,}")
    print(
        f"tokens/step={tokens_per_step:,} total={tokens_per_step * tc.max_steps:,} "
        f"(~{tokens_per_step * tc.max_steps / len(train_data.tokens):.1f} epochs)"
    )
    print(f"expected loss at step 0: ln({mc.vocab_size}) = {math.log(mc.vocab_size):.3f}")

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{run_name}.pt"
    best_val = float("inf")
    gen = torch.Generator().manual_seed(tc.seed)
    t0 = time.perf_counter()

    with (run_dir / "metrics.jsonl").open("w") as log:
        for step in range(tc.max_steps + 1):
            last = step == tc.max_steps
            if step % tc.eval_interval == 0 or last:
                val_loss = evaluate(model, val_data, tc, device)
                improved = val_loss < best_val
                print(
                    f"step {step:5d} | val {val_loss:.4f} | ppl {math.exp(val_loss):7.2f}"
                    f"{' | saved' if improved else ''}"
                )
                log.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                if improved:
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_config": dataclasses.asdict(mc),
                            "train_config": dataclasses.asdict(tc),
                            "step": step,
                            "val_loss": val_loss,
                        },
                        ckpt_path,
                    )
            if last:
                break

            mult = lr_multiplier(step, tc)
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * mult

            # Gradient accumulation: several micro-batches add up to one optimizer step.
            train_loss = 0.0
            for _ in range(tc.grad_accum_steps):
                x, y = train_data.batch(tc.batch_size, mc.seq_len, device, gen)
                _, loss = model(x, y)
                (loss / tc.grad_accum_steps).backward()
                train_loss += loss.item() / tc.grad_accum_steps
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)

            if step % 10 == 0:
                elapsed = time.perf_counter() - t0
                tok_s = tokens_per_step * (step + 1) / elapsed
                print(
                    f"step {step:5d} | loss {train_loss:.4f} | lr x{mult:.2f} "
                    f"| grad {grad_norm:5.2f} | {tok_s:,.0f} tok/s"
                )
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": train_loss,
                            "lr_mult": mult,
                            "grad_norm": grad_norm,
                            "tok_per_s": tok_s,
                        }
                    )
                    + "\n"
                )
                log.flush()

    print(f"done in {time.perf_counter() - t0:.0f}s; best val loss {best_val:.4f} -> {ckpt_path}")
    return ckpt_path


def parse_args() -> tuple[ModelConfig, TrainConfig, str]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", default="shakespeare", help="name for the checkpoint and logs")
    # Every dataclass field becomes a --flag, e.g. --n-layer 4 --matrix-lr 0.03
    for cls in (ModelConfig, TrainConfig):
        for f in dataclasses.fields(cls):
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)
    args = vars(parser.parse_args())

    def build(cls):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in args.items() if k in names and v is not None})

    return build(ModelConfig), build(TrainConfig), args["run"]


def main() -> None:
    mc, tc, run = parse_args()
    train(mc, tc, run)


if __name__ == "__main__":
    main()
