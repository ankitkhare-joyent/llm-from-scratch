# Lab 00 · The whole model: shapes and parameters

One sitting, about 20 minutes. Background: [docs/00](../docs/00-overview.md),
[docs/04](../docs/04-attention-and-blocks.md#parameter-count). Code: [model.py](../src/tinyllm/model.py),
[config.py](../src/tinyllm/config.py), [lab/shapes.py](shapes.py).

Before you study any part, see the whole thing run once. `lab:shapes` builds the model, pushes a random batch
through it, and prints the shape of the tensor leaving each stage. After this lab you can explain:

- what shape goes in, what comes out, and what each dimension means
- where the 2.56 M parameters live, and why it isn't where you'd guess
- what the config checks reject, and why

Answers to every **Predict** are at the bottom, with the output I got from a fresh start.

## Before you start

```bash
cd ~/workstuff/argocd-terraform-opentofu/llm-from-scratch
vim src/tinyllm/config.py          # ModelConfig: the 9 numbers that define the network
```

## 1. Follow one batch through

The input is a batch of 2 sequences of 16 token ids.

**Predict (P1):** write the shape after each of these: the token embedding, `c_q`, `c_k`, the MLP's first layer
`c_fc`, and the final logits. You need `n_embd=192`, `n_head=6`, `n_kv_head=2`, `mlp_ratio=4`, `vocab_size=512`.

```bash
mise run -q lab:shapes
```

## 2. Where the parameters are

**Predict (P2):** rank these by parameter count: the token embedding, all attention layers together, all MLPs
together, the output head.

The bottom half of the same output answers it.

## 3. What grouped-query attention saves

The default has 6 query heads but only 2 key/value heads.

**Predict (P3):** with `--n-kv-head 6` (plain multi-head attention), which three numbers in the output change, and by
how much?

```bash
mise run -q lab:shapes -- --n-kv-head 6
```

## Break it, then fix it

**Break 1: heads that don't divide the width.** Ask for 5 heads.

**Predict (P4):** does it build a model with a slightly odd head size, or refuse? If it refuses, where?

```bash
mise run -q lab:shapes -- --n-head 5
```

**Fix:** pick a head count that divides 192 (1, 2, 3, 4, 6, 8, 12, …) and a KV-head count that divides the head
count. `mise run -q lab:shapes -- --n-head 8 --n-kv-head 4` builds.

**Break 2: a sequence longer than the context.** Feed 300 tokens to a model trained on 256.

**Predict (P5):** works, works with a warning, or fails?

```bash
mise run -q lab:shapes -- --seq 300
```

**Fix:** keep inputs to `seq_len` or less. `sample.py` handles long generation by re-prefilling the most recent half
of the window (Lab 06); a real long-context model is trained on longer sequences instead.

## Check your predictions

**P1.**

```text
input ids                  (2, 16)
  wte (token embedding)    (2, 16, 192)
  block0.attn.c_q          (2, 16, 192)
  block0.attn.c_k          (2, 16, 64)
  block0.attn.c_v          (2, 16, 64)
  block0.attn (output)     (2, 16, 192)
  block0.mlp.c_fc          (2, 16, 768)
  block0.mlp (output)      (2, 16, 192)
  block5 (output)          (2, 16, 192)
  lm_head (logits)         (2, 16, 512)
```

The pattern is `(B, T, something)` all the way through: B sequences, T positions, and a vector per position. The
residual stream stays 192 wide between every block, which is why blocks can be stacked. `c_k` is 64 wide because 2
KV heads x 32 dims each, while `c_q` is 6 x 32 = 192. The logits have one score per vocabulary entry *at every
position*: training uses all 16 positions (16 predictions per sequence), generation only uses the last one.

**P2.**

```text
wte                          98,304    3.8%
lm_head                      98,304    3.8%
attention (all blocks)      589,824   23.1%
mlp (all blocks)          1,769,472   69.2%
total                     2,555,904
```

The MLPs hold over two thirds of the model. Each MLP is two 192 x 768 matrices (294,912), three times an attention
layer (98,304). Attention gets the attention in explanations, but the MLPs are where most of the weights, and
most of the compute per token, are. Production models keep roughly this ratio; Mixture-of-Experts models exist to
grow exactly this part without growing the cost per token.

**P3.** `c_k` and `c_v` grow from 64 to 192 wide, attention parameters rise from 589,824 to 884,736 (the total
from 2,555,904 to 2,850,816), and the KV cache per token goes from 3,072 to 9,216 bytes: 768 KiB to 2,304 KiB for a
full window. That 3x on the cache is the point of GQA. The parameter saving is small; the cache saving is what
matters when serving many long conversations (Lab 07 puts numbers on it).

**P4.** It refuses before building anything:

```text
  File ".../src/tinyllm/config.py", line 37, in __post_init__
    raise ValueError("n_embd must be divisible by n_head")
ValueError: n_embd must be divisible by n_head
```

`ModelConfig.__post_init__` checks three things: `n_embd % n_head`, `n_head % n_kv_head`, and an even `head_dim`
(RoPE rotates channels in pairs). Failing at config time, with the rule in the message, beats a reshape error deep
inside the first forward pass.

**P5.** Fails:

```text
ValueError: sequence of 300 exceeds context of 256
```

The RoPE tables only hold 256 positions, and the model never saw a position past 255 in training, so `GPT.forward`
refuses rather than silently slicing the tables short.

## What to remember

- Every tensor in the stack is `(batch, time, channels)`; only the last dimension changes width.
- Most parameters are in the MLPs. GQA barely changes the parameter count but cuts the KV cache by `n_head / n_kv_head`.
- Validate configs where they're created, with a message that states the rule.

Next: [Lab 01 · The tokenizer](01-tokenizer.md).
