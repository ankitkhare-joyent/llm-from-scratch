"""Text generation: KV-cached decoding with temperature, top-k, and top-p sampling.

python -m tinyllm.sample --prompt "ROMEO:" --tokens 300
python -m tinyllm.sample --temperature 0          # greedy
python -m tinyllm.sample --top-p 0.9 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import torch

from tinyllm.config import CHECKPOINT_DIR, ModelConfig, pick_device
from tinyllm.data import load_tokenizer
from tinyllm.model import GPT, KVCache


def load_model(path: Path, device: torch.device) -> GPT:
    # weights_only=True refuses to unpickle arbitrary objects: never load untrusted
    # checkpoints any other way.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {path.name}: step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}", file=sys.stderr)
    return model


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Pick the next token id from a (V,) vector of logits."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits.float().cpu() / temperature
    if top_k is not None and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if top_p is not None and top_p < 1.0:
        # Nucleus sampling: keep the smallest set of tokens whose probability sums to top_p.
        sorted_probs, order = probs.sort(descending=True)
        drop = sorted_probs.cumsum(-1) - sorted_probs > top_p
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, order, sorted_probs)
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> Iterator[int]:
    """Yield new token ids one at a time.

    Prefill runs the whole prompt once and fills the KV cache; each decode step then
    feeds a single token. When the context window is full, the cache is rebuilt from
    the most recent half of the text (a "sliding" restart), since the model was never
    trained on positions beyond seq_len.
    """
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    window = model.cfg.seq_len
    tokens = list(prompt) or [0]
    cache = KVCache(model.cfg, 1, device, torch.float32)
    logits, _ = model(torch.tensor([tokens[-window:]], device=device), kv_cache=cache)
    for _ in range(max_new_tokens):
        next_id = sample_next(logits[0, -1], temperature, top_k, top_p, gen)
        tokens.append(next_id)
        yield next_id
        if cache.pos >= window:
            cache = KVCache(model.cfg, 1, device, torch.float32)
            context = tokens[-(window // 2) :]
            logits, _ = model(torch.tensor([context], device=device), kv_cache=cache)
        else:
            logits, _ = model(torch.tensor([[next_id]], device=device), kv_cache=cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "shakespeare.pt")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = load_model(args.checkpoint, pick_device(args.device))
    tok = load_tokenizer(model.cfg.vocab_size)
    ids = tok.encode(args.prompt)
    sys.stdout.write(args.prompt)
    # Decode the running sequence and print only the new suffix, so tokens that split
    # a multi-byte character still come out correctly.
    shown = tok.decode(ids)
    for new_id in generate(model, ids, args.tokens, args.temperature, args.top_k, args.top_p, args.seed):
        ids.append(new_id)
        text = tok.decode(ids)
        sys.stdout.write(text[len(shown) :])
        sys.stdout.flush()
        shown = text
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
