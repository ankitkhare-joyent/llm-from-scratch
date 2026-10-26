# Lab 03 · Embeddings and RoPE: how the model knows word order

One sitting, about 25 minutes. Background: [docs/03](../docs/03-embeddings-and-rope.md). Code:
`rope_tables` and `apply_rope` in [model.py](../src/tinyllm/model.py), [lab/rope_check.py](rope_check.py).

Attention compares tokens with dot products, and dot products ignore order. RoPE fixes that by rotating queries and
keys by their position. After this lab you can explain:

- why a RoPE attention score depends only on the distance between two tokens
- what `rope_base` controls, in tokens
- how much this model loses without any position encoding, and why it's less than you'd expect

## Before you start

```bash
vim +/'def rope_tables' src/tinyllm/model.py
```

## 1. Rotation keeps length

`lab:rope` makes one random 32-dim query and key, rotates them to various positions with the real `apply_rope`, and
prints their dot products.

**Predict (P1):** after rotating `q` to position 77, is its length bigger, smaller, or the same?

## 2. Only the distance matters

**Predict (P2):** the script scores query/key pairs at positions (5, 2), (50, 47) and (200, 197): all 3 apart. Then
(50, 10) and (200, 160): both 40 apart. Which scores will be equal?

```bash
mise run -q lab:rope
```

## 3. What the base sets

Each pair of channels turns at its own speed: `θᵢ = base^(-2i/d)` radians per token.

**Predict (P3):** with `--base 100` instead of 10,000, does the "only distance matters" property still hold? What
changes?

```bash
mise run -q lab:rope -- --base 100
```

## Break it, then fix it

**Break: no position encoding at all.** `--no-rope` replaces `apply_rope` with the identity, so queries and keys
carry no position.

**Predict (P4):** the baseline from Lab 02 reached val 3.6311 in 100 steps. Without RoPE: about the same, a bit
worse, or does it fail to learn? Hint: there's still a causal mask.

```bash
mise run -q lab:train -- --run lab-norope --no-rope
```

**Fix:** nothing to edit; the default has RoPE. The answer explains why it matters even though the model limps along
without it.

## Check your predictions

**P1.** The same: `|q| before 5.6920  after rotating to pos 77: 5.6920`. Rotation changes direction only. That's
why QK-norm can come after RoPE without any interaction: the norm doesn't care.

**P2.**

```text
query pos m key pos n m - n  score q·k
          5         2     3  -2.62997
         50        47     3  -2.62997
        200       197     3  -2.62997
          5         4     1  -1.92911
         50        10    40  -3.07140
        200       160    40  -3.07141
```

Same distance, same score, to five decimal places (the last digit is float rounding), wherever the pair sits in the
window. Rotating `q` by `mθ` and `k` by `nθ` is the same as rotating `q` by `(m - n)θ` relative to `k`. So a
pattern the model learns at one position, "look 1 back", "look at the speaker name 3 back", works at every
position for free.

**P3.** Still holds (distance 3 gives `-1.98312` three times; distance 40 gives `-0.16755` twice). The base
changes the *periods*:

```text
base 10000: fastest pair one lap every 6 tokens, slowest every 35,333 tokens
base   100: fastest pair one lap every 6 tokens, slowest every 471 tokens
```

The slowest pairs tell the model about long distances. Once a pair has turned a full lap, distances one lap apart
look the same to it. At base 10,000 nothing laps within this 256-token window. Long-context models raise the base
(Llama 3 uses 500,000) so nothing laps within 128 k tokens, and context-extension tricks like YaRN rescale exactly
these frequencies.

**P4.** A bit worse, and it still learns:

| run | step 25 | step 50 | step 100 |
|---|---|---|---|
| with RoPE (lab-base) | 4.2239 | 3.8909 | **3.6311** |
| no RoPE | 4.2233 | 3.9474 | **3.7926** |

Identical at first (early learning is mostly "which tokens are common", no positions needed), then the gap opens:
0.16 nats by step 100, and still growing. It learns at all because the causal mask leaks position: position 10 can
attend to 11 tokens and position 200 to 201, so the averages it computes differ and the model can infer roughly
where it is. This is a known result ("NoPE" models). But inferring position is harder and less precise than being
told, and it can't express "exactly 3 back" cleanly. That's the loss gap.

## What to remember

- RoPE rotates q and k by position; the score depends on `m - n` only, and lengths are unchanged.
- `rope_base` sets the longest distance the slowest channels can distinguish.
- A causal transformer without positions still learns, worse. A gap of 0.16 nats is large: it's more than the
  difference between 300 and 400 steps of the full run.

Next: [Lab 04 · Attention](04-attention.md).
