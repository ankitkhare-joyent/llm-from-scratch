# Lab 02 · Data and batching: what the model is asked to predict

One sitting, about 25 minutes. Background: [docs/02](../docs/02-data-and-batching.md). Code:
[data.py](../src/tinyllm/data.py), [lab/batch.py](batch.py), [lab/quick_train.py](quick_train.py).

Pre-training has one objective: predict the next token. Everything in this lab is about getting the inputs and
targets exactly right, because a mistake here doesn't crash; it trains happily on the wrong task. After this lab you
can explain:

- how one 256-token window turns into 256 training examples
- how many times the model sees each token, and why that matters
- what an off-by-one target bug looks like in the loss curve

## Before you start

```bash
ls -la data/                                  # the download, the tokenizer, train.bin, val.bin, bins.stamp
vim +/'def batch' src/tinyllm/data.py
```

## 1. One batch, decoded

**Predict (P1):** a batch is `x` and `y`, each `(32, 256)`. How is row 0 of `y` related to row 0 of `x`? What's the
dtype, and why not `uint16` like the files on disk?

```bash
mise run -q lab:batch
```

## 2. The data budget

**Predict (P2):** the training split is 492,548 tokens and a run is 400 steps of 32 x 256 tokens. How many times does
the model see an average token? Is that more or less than a production pre-training run?

The bottom of the same output answers it.

## 3. A healthy short run

You need a baseline to compare the break against. This is the default model, trained for 100 steps instead of 400.

```bash
mise run -q lab:train -- --run lab-base
```

Write down the val loss at steps 25, 50 and 100.

## Break it, then fix it

**Break: the targets are the inputs.** `--copy-targets` makes `y = x` instead of `x` shifted by one. It's the
classic off-by-one: forget the `[:, 1:]` and every position is asked to predict *itself*.

**Predict (P3):** compared with the baseline, does the loss go down faster, slower, or not at all? What happens to
the *validation* loss? Would anything warn you?

```bash
mise run -q lab:train -- --run lab-copy --copy-targets
mise run -q sample -- --checkpoint checkpoints/lab-copy.pt --seed 1 --tokens 60
```

**Fix:** there's nothing to edit (the break is a flag), but know the two checks that would have caught it:

1. A loss far below anything plausible is a bug, not a breakthrough. Text has real uncertainty: no model predicts
   Shakespeare's next token with 99% confidence.
2. Assert the shift once, at the data boundary. `lab:batch` prints `y equals x shifted left by one: True`; a test
   that does the same costs one line.

## Check your predictions

**P1.**

```text
x (32, 256) torch.int64   y (32, 256) torch.int64
y equals x shifted left by one: True

pos  input x        target y
  0  'nt'           'ry'
  1  'ry'           "'s"
  2  "'s"           ' good'
  3  ' good'        ' with'
  4  ' with'        ' a'
```

Each target is the next input. Position 0 predicts `'ry'` having seen one token; position 255 predicts its target
having seen 255. One window gives 256 examples, each with a different amount of context, all computed in one pass
(the causal mask in Lab 04 is what stops each position seeing its own answer). `int64` because
`nn.Embedding` and `cross_entropy` want `long` indices; `uint16` on disk halves the file size.

Notice the window starts mid-word (`'nt'`): offsets are random token positions, not line or sentence starts.

**P2.**

```text
train tokens 492,548   val tokens 54,728   val share 10.0%
tokens per step 8,192   steps 400   tokens seen 3,276,800
epochs 6.65   predictions per window 256
```

About 6.7 passes over the data. Production runs see most tokens once (under one epoch) because repeated data stops
helping after a few passes and starts being memorised. This lab only gets away with it because the run is short and
weight decay is on; Lab 05's 600-step story shows what happens if you keep going.

**Baseline (section 3).** Val 4.2239 at step 25, 3.8909 at 50, 3.6311 at 100, in 90 s.

**P3.** It goes down much faster, to almost nothing, *and validation follows it*:

```text
BREAK: targets = inputs
step     0 | val 6.2369 | ppl  511.26 | saved
step    25 | val 0.3044 | ppl    1.36 | saved
step    50 | val 0.0224 | ppl    1.02 | saved
step   100 | val 0.0119 | ppl    1.01 | saved
```

Copying the input token is trivial: the embedding goes in, the head reads it back out. The validation loss falls too,
because the same bug is in the validation batches, so the usual "train good, val bad" over-fitting alarm never
fires. Nothing errors and nothing warns. Then sampling shows the truth:

```text
loaded lab-copy.pt: step 100, val loss 0.0119
ROMEO:
:
:
:
```

A model that predicts its own input repeats the last prompt token (`':\n'`) forever. A val loss of 0.01 (perplexity 1.01) on natural language is the warning.

## What to remember

- Inputs and targets are the same tokens, shifted by one. Assert it.
- Split by position so validation is unseen *later* text, and the same code path builds both splits, which means a
  data bug corrupts both.
- Count epochs. More than a handful means you're measuring memorisation as much as learning.

Next: [Lab 03 · Embeddings and RoPE](03-embeddings-and-rope.md).
