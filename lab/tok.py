"""Lab 01: look inside the tokenizer.

mise run -q lab:tok -- "ROMEO: wherefore art thou"      # ids and pieces with the trained tokenizer
mise run -q lab:tok -- --vocab 1024 "some text"         # train a fresh tokenizer of another size
mise run -q lab:tok -- --no-split --vocab 300 --chars 200000 "and the"   # merges may cross words
mise run -q lab:tok -- --emoji                          # decode one token at a time vs all at once
"""

from __future__ import annotations

import argparse
import time

from tinyllm.data import download, load_tokenizer
from tinyllm.tokenizer import BPETokenizer

# Chunks are whole lines: the only boundary merges can't cross is a newline.
LINE_PATTERN = r"[^\n]*\n|[^\n]+"


def show(tok: BPETokenizer, text: str) -> None:
    ids = tok.encode(text)
    pieces = [tok.vocab[i].decode("utf-8", "replace") for i in ids]
    print(f"{text!r}: {len(text.encode())} bytes -> {len(ids)} tokens")
    print(f"  ids    {ids}")
    print(f"  pieces {pieces}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--vocab", type=int, default=None, help="train a fresh tokenizer of this size")
    parser.add_argument("--no-split", action="store_true", help="train without the GPT-4 regex pre-split")
    parser.add_argument("--chars", type=int, default=None, help="train on only the first N characters")
    parser.add_argument("--emoji", action="store_true", help="the multi-byte decoding drill")
    args = parser.parse_args()

    corpus = download().read_text(encoding="utf-8")
    if args.vocab or args.no_split:
        train_text = corpus[: args.chars] if args.chars else corpus
        t0 = time.perf_counter()
        kwargs = {"pattern": LINE_PATTERN} if args.no_split else {}
        tok = BPETokenizer.train(train_text, args.vocab or 512, **kwargs)
        print(
            f"trained {tok.vocab_size}-token tokenizer on {len(train_text):,} chars "
            f"in {time.perf_counter() - t0:.1f}s{' (no pre-split)' if args.no_split else ''}"
        )
        longest = sorted(tok.vocab.values(), key=len, reverse=True)[:8]
        print(f"longest tokens: {[t.decode('utf-8', 'replace') for t in longest]}")
    else:
        tok = load_tokenizer()

    sample = corpus[: args.chars] if args.chars else corpus
    n = len(tok.encode(sample))
    print(
        f"{'first ' + format(len(sample), ',') + ' chars' if args.chars else 'corpus'}: "
        f"{len(sample.encode()):,} bytes -> {n:,} tokens ({len(sample.encode()) / n:.2f} bytes/token)\n"
    )
    for text in args.text:
        show(tok, text)

    if args.emoji:
        text = "Romeo 🌹 Juliet"
        ids = tok.encode(text)
        one_by_one = "".join(tok.decode([i]) for i in ids)
        print(f"text            {text!r}")
        print(f"token ids       {ids}")
        print(f"decode each id  {one_by_one!r}")
        print(f"decode all ids  {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
