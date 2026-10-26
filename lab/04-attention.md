# Lab 04 · Attention: what the heads look at

One sitting, about 35 minutes. Background: [docs/04](../docs/04-attention-and-blocks.md). Code:
`CausalSelfAttention` in [model.py](../src/tinyllm/model.py), [lab/attn_map.py](attn_map.py),
[tests/test_tinyllm.py](../tests/test_tinyllm.py).

`lab:attn` reruns the trained model on a passage, captures what goes into each attention layer, and recomputes the
attention weights `softmax(q·kᵀ/√d)` with the causal mask, exactly as `CausalSelfAttention` does inside the fused
kernel. After this lab you can explain:

- what an attention row is, and the two properties every row must have
- what the heads of a 2.6 M-parameter model actually learned, and where in the stack
- what grouped-query attention looks like from the inside
- what happens, and what *doesn't* warn you, when the causal mask is missing

## Before you start

```bash
vim +/'class CausalSelfAttention' src/tinyllm/model.py
```

## 1. Two properties of every row

Row *t* of a head's attention matrix says how position *t* splits its attention over positions 0..t.

**Predict (P1):** what should every row sum to? What's the largest weight any position puts on a *later* position?

```bash
mise run -q lab:attn
```

The first line of the output checks both, across all 36 heads.

## 2. What the heads do

The table gives, per head, the mean weight on itself (`self`), on the previous token (`prev`), and on the first
token (`first`). `xunif` compares `prev` with a head that spreads attention evenly; `argmax=prev` is how often the
previous token is the head's single strongest pick.

**Predict (P2):** in which layers will you find heads that mostly look at the previous token: the first, the
middle, or the last? Will any head put more than half its attention on it?

**Predict (P3):** the `kv` column says which key/value group each head belongs to: heads 0 to 2 share K/V group 0,
heads 3 to 5 share group 1. Will heads in the same group behave alike?

## 3. One line, token by token

```bash
mise run -q lab:attn -- --layer 1 --text "ROMEO:\nBut, soft! what light"
```

For each token, this lists the token each of layer 1's six heads attends to most. Look at the rows for
` what`, ` li` and `ght`.

## Break it, then fix it

**Break: remove the causal mask.** Edit the source:

```bash
sed -i '' 's/is_causal=True)/is_causal=False)/' src/tinyllm/model.py
git diff src/tinyllm/model.py
```

**Predict (P4):** which of the 10 tests fail?

```bash
mise exec -- uv run pytest -q
```

**Predict (P5):** train 200 steps with the mask gone. Does the val loss look better or worse than a normal run? What
does the model generate?

```bash
mise run -q lab:train -- --run lab-nocausal --max-steps 200
mise run -q sample -- --checkpoint checkpoints/lab-nocausal.pt --seed 1 --tokens 60
```

**Fix:**

```bash
git restore src/tinyllm/model.py
mise exec -- uv run pytest -q          # 10 passed
```

## Check your predictions

**P1.** `check: every row sums to 1 (1.0000..1.0000); max weight on a future token 0.0e+00`. Softmax makes each row
a probability distribution, and the mask sets future scores to `-inf` before the softmax, so their weight is exactly
zero, not just small.

**P2.** The middle, and no head gets past a quarter:

```text
layer head kv   self  prev first xunif argmax=prev
    0    1  0   0.05  0.07  0.01   3.3         22%
    1    3  1   0.10  0.19  0.01   9.4         58%
    1    4  1   0.10  0.19  0.01   9.5         60%
    1    5  1   0.10  0.20  0.01  10.0         62%
    2    2  0   0.12  0.21  0.01  10.5         67%
    5    0  0   0.05  0.04  0.01   1.9          7%
most previous-token-like head: layer 2 head 2 (0.21 on t-1)
```

(Selected rows; run it for all 36.) Layers 1 and 2 hold the previous-token heads: 10x more weight on t-1 than an even
spread, and t-1 is their top pick about 60% of the time. Layer 0 can't do this well yet: its input is just
the token embedding, so its queries know only "which token am I", not much about context. The last layers spread
attention broadly; they're building the prediction from everything. No head is a clean 90% "previous token" head:
a 2.6 M model after 400 steps has soft, overlapping specialisation. Bigger, longer-trained models develop crisp
heads (the "induction heads" studied in interpretability research).

**P3.** Yes, visibly: layer 1's group-1 heads (3, 4, 5) all sit at 0.19 to 0.20 on t-1 and 58 to 62%, while its
group-0 heads are at 0.05 to 0.15. Heads in a group read the same keys, so with only their queries free to differ
they tend to learn related patterns. That's the price of GQA's 3x smaller cache: a little less freedom per head.

**Section 3.**

```text
 12 ' what'    '!'       '!'       '!'       ' what'   ' what'   '!'
 13 ' li'      '!'       ' what'   't'       ' what'   ' what'   ' what'
 14 'ght'      't'       ' what'   ' li'     ' li'     ' li'     ' li'
```

For each of the three, four of the six heads pick the previous token (`'!'`, `' what'`, `' li'`): the "look one
back" pattern from P2, token by token. Heads 3 and 4, which share KV group 1, pick the same token on all three rows.
The others look further back: head 1 at `'ght'` reaches past `' li'` to `' what'`, which is how the model can know
it is finishing the word *light* after *what*.

**P4.** Two fail:

```text
FAILED tests/test_tinyllm.py::test_model_is_causal - AssertionError: Tensor-l...
FAILED tests/test_tinyllm.py::test_kv_cache_matches_full_forward - AssertionE...
2 failed, 8 passed
```

`test_model_is_causal` changes token 20 and checks positions 0 to 19 don't move; now they do. The KV-cache test
fails because the cached path (one query at a time against past keys) can only ever see the past, so it no longer
matches the full forward pass.

**P5.** The loss looks *spectacular* and the model is useless:

```text
step    50 | val 3.8284 | ppl   45.99 | saved
step    75 | val 1.4083 | ppl    4.09 | saved
step   100 | val 0.0780 | ppl    1.08 | saved
step   200 | val 0.0225 | ppl    1.02 | saved
```

```text
ROMEO:
PPPPPPPPPTheThe,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
```

For 50 steps it looks normal, then it discovers it can attend to position t+1 and read the answer: val drops to
0.02. Validation is fooled because it runs the same broken attention. At generation time the next token doesn't
exist yet, so the trick has nothing to read, and the model has learned nothing else. Same lesson as Lab 02's
off-by-one: a too-good loss is a bug. The difference is that here a test caught it before any training.

## What to remember

- Every attention row is a probability distribution over the past; the mask makes the future exactly zero.
- Small models learn soft specialisations: previous-token heads in the early-middle layers, broad mixing later.
- Heads sharing a KV group behave alike.
- A missing causal mask trains to near-zero loss and generates garbage. Keep the causality test.

Next: [Lab 05 · Training](05-training.md).
