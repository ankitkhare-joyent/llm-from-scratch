"""Byte-level byte-pair encoding (BPE), the tokenizer family used by GPT-2 onward.

The algorithm:
  1. Split text into chunks with a regex so merges never cross word/number/space
     boundaries (the same pattern GPT-4's cl100k tokenizer uses).
  2. Start from the 256 possible byte values, so any string is encodable.
  3. Repeatedly find the most frequent adjacent pair of ids and mint a new id for it.

Production tokenizers (tiktoken, Hugging Face tokenizers, nanochat's rustbpe) run the
same algorithm in Rust over gigabytes; this one is pure Python and fast enough here
because it counts each distinct chunk once and weights it by frequency.

Run `python -m tinyllm.tokenizer` to train on TinyShakespeare and inspect the result.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import regex

# GPT-4 (cl100k_base) pre-tokenisation pattern.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]


def _pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for ids, freq in chunks.items():
        for pair in itertools.pairwise(ids):
            counts[pair] += freq
    return counts


def _merge(ids: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Byte-level BPE with a regex pre-split. Ids 0-255 are raw bytes."""

    def __init__(self, merges: list[Pair] | None = None, pattern: str = SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self._compiled = regex.compile(pattern)
        self.merges: list[Pair] = list(merges or [])
        # rank = order a merge was learned; lower rank merges are applied first
        self.ranks: dict[Pair, int] = {p: 256 + i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self._encode_chunk = lru_cache(maxsize=1 << 16)(self._encode_chunk_uncached)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(
        cls, text: str, vocab_size: int, verbose: bool = False, pattern: str = SPLIT_PATTERN
    ) -> BPETokenizer:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one id per byte)")
        compiled = regex.compile(pattern)
        chunk_freq = Counter(compiled.findall(text))
        chunks = {tuple(c.encode("utf-8")): f for c, f in chunk_freq.items()}
        merges: list[Pair] = []
        for i in range(vocab_size - 256):
            counts = _pair_counts(chunks)
            if not counts:
                break  # every chunk is a single token already
            pair, freq = max(counts.items(), key=lambda kv: kv[1])
            new_id = 256 + i
            chunks = {_merge(ids, pair, new_id): f for ids, f in chunks.items()}
            merges.append(pair)
            if verbose and (i < 10 or (i + 1) % 64 == 0):
                tok = cls(merges, pattern)
                print(f"merge {i + 1:4d}: {tok.vocab[new_id]!r:>16} (seen {freq} times)")
        return cls(merges, pattern)

    # ---------------------------------------------------------------- encode/decode
    def _encode_chunk_uncached(self, chunk: bytes) -> tuple[int, ...]:
        ids = tuple(chunk)
        while len(ids) >= 2:
            pairs = set(itertools.pairwise(ids))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break  # nothing left to merge
            ids = _merge(ids, best, self.ranks[best])
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def decode(self, ids: list[int]) -> str:
        # A token can end mid-way through a multi-byte character; "replace" keeps
        # streaming output from crashing on such partial sequences.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        payload = {"version": 1, "pattern": self.pattern, "merges": self.merges}
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        payload = json.loads(path.read_text())
        if payload.get("version") != 1:
            raise ValueError(f"unsupported tokenizer version in {path}")
        return cls([tuple(p) for p in payload["merges"]], payload["pattern"])


def main() -> None:
    from tinyllm.config import DATA_DIR, ModelConfig
    from tinyllm.data import download

    text = download().read_text(encoding="utf-8")
    vocab_size = ModelConfig().vocab_size
    print(f"training BPE to {vocab_size} tokens on {len(text):,} characters\n")
    tok = BPETokenizer.train(text, vocab_size, verbose=True)
    path = DATA_DIR / "tokenizer.json"
    tok.save(path)

    sample = "ROMEO:\nBut, soft! what light through yonder window breaks?"
    ids = tok.encode(sample)
    print(f"\nsaved {path.relative_to(DATA_DIR.parent)}")
    print(f"sample: {len(sample.encode())} bytes -> {len(ids)} tokens")
    print("pieces:", [tok.vocab[i].decode("utf-8", "replace") for i in ids])
    assert tok.decode(ids) == sample
    n_tokens = len(tok.encode(text))
    print(
        f"whole corpus: {len(text.encode()):,} bytes -> {n_tokens:,} tokens "
        f"({len(text.encode()) / n_tokens:.2f} bytes/token)"
    )


if __name__ == "__main__":
    main()
