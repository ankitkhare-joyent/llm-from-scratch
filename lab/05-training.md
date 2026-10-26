# Lab 05 · Training: reading the loop, then breaking it

One sitting, about 35 minutes. Background: [docs/05](../docs/05-training.md). Code:
[train.py](../src/tinyllm/train.py), `reset_parameters` in [model.py](../src/tinyllm/model.py),
[lab/quick_train.py](quick_train.py).

You already have a trained model from the fresh start. This lab reads what that run logged, then breaks the two
things that most often ruin real runs: initialisation and the learning rate. After this lab you can explain:

- why step 0 must print 6.2385, and what any other number means
- the shape of the learning-rate schedule, step by step
- what the train/val gap and the gradient norm tell you
- how a bad initialisation and a bad learning rate show up in the log

## Before you start

```bash
vim +/'def reset_parameters' src/tinyllm/model.py
vim +/'def lr_multiplier' src/tinyllm/train.py
```

## 1. The first number

**Predict (P1):** the model has 512 tokens and `lm_head` starts with weights of standard deviation 0.001. What loss
should step 0 print? What would it be for a 1,024-token vocabulary?

The first lines of your fresh-start `mise run train` output have it, or rerun the short baseline:
`mise run -q lab:train -- --run lab-base | head -5`.

## 2. The schedule

**Predict (P2):** with `max_steps=400`, `warmup_steps=30` and `warmdown_ratio=0.4`, write the LR multiplier at steps
0, 14, 29, 239, 240, 320 and 399.

```bash
mise exec -- uv run python -c "
from tinyllm.train import lr_multiplier; from tinyllm.config import TrainConfig
tc = TrainConfig()
for s in (0, 14, 29, 30, 239, 240, 320, 399): print(s, round(lr_multiplier(s, tc), 4))"
```

## 3. Reading the log

Every run writes `runs/<name>/metrics.jsonl`. Pull out the fresh-start run's train loss, gradient norm and
throughput:

```bash
python3 -c "
import json
for l in open('runs/shakespeare/metrics.jsonl'):
    r = json.loads(l)
    if 'val_loss' in r or r['step'] in (0, 10, 20, 50, 100, 200, 300, 390): print(r)"
```

**Predict (P3):** at step 400 the val loss is 3.006. Is the train loss at the end higher, lower, or about the same?
What does the gap mean?

**Predict (P4):** when is the gradient norm largest: at the start, the middle, or the end?

## Break it, then fix it

**Break 1: a confident random start.** `--head-std 1.0` initialises `lm_head` with standard deviation 1.0, the
"obvious" default, instead of 0.001.

**Predict (P5):** what does step 0 print now? Does the run recover in 100 steps, and does it catch up with the
baseline (val 3.6311 at step 100)?

```bash
mise run -q lab:train -- --run lab-headstd --head-std 1.0
```

**Fix:** keep the tiny init. The general rule: the output layer should start close to "I don't know", so step 0
equals `ln(vocab)`. Check that number on every new run; it costs nothing.

**Break 2: a learning rate 17x too high, with no warmup.** Plain AdamW at `0.05` (the default AdamW LR is `0.003`),
warmup of 1 step.

**Predict (P6):** does it diverge to `nan`, spike and recover, or something else?

```bash
mise run -q lab:train -- --run lab-hotlr --optimizer adamw --adamw-lr 0.05 --warmup-steps 1
```

**Fix:** the defaults (warmup, then a tuned rate). When you change model size, sweep the learning rate on short
runs like these before a long one.

## Check your predictions

**P1.** `ln(512) = 6.2383` for train and `6.2385` for val (the val number is an average of 20 batches). Weights of
0.001 give logits of almost exactly zero, so the softmax is uniform: 1/512 for every token, and
`-ln(1/512) = ln 512`. For 1,024 tokens it would be `ln(1024) = 6.931`. The training script prints the expectation
before the first step so you can compare.

**P2.**

```text
0 0.0333      14 0.5      29 1.0      30 1.0
239 1.0       240 1.0     320 0.5     399 0.0062
```

Linear warmup over steps 0 to 29 (step *s* gets `(s+1)/30`), flat at 1.0 until step 240 (60% of 400), then linear
down to 0 at step 400. Step 320 is halfway through the warmdown, so 0.5.

**P3.** Much lower: train loss 2.109 at step 390 against val 3.006. That 0.9 gap is the model doing better on text
it has seen 6 times than on text it hasn't: some of its skill is memorisation. The val curve still went down every
eval, so it was still learning things that generalise as well.

```text
step   0  train 6.238  grad 0.27        val 6.2385
step  10  train 5.762  grad 1.58
step  20  train 4.482  grad 0.55
step 100  train 3.190  grad 0.37        val 3.5772
step 200  train 2.555  grad 0.41        val 3.1408
step 390  train 2.109  grad 0.39        val 3.0057
```

**P4.** Early, at step 10 (1.58), as the learning rate ramps up and the model makes its big first moves; it settles
around 0.3 to 0.4 for the rest of the run. With `grad_clip=1.0`, step 10 is one of the few steps where clipping
actually acts. In a long production run, a gradient norm that suddenly jumps mid-run is the warning sign of a loss
spike.

**P5.**

```text
BREAK: lm_head init std 1.0
step     0 | loss 17.9862 | lr x0.03 | grad 29.93
step     0 | val 18.0128 | ppl 66504494.79
step    10 | loss 10.8183 | lr x0.37 | grad 16.20
step    20 | loss 5.7243 | lr x0.70 | grad  4.33
step   100 | val 3.7626 | ppl   43.06
```

Step 0 prints 18, three times `ln(512)`: the model starts *confidently wrong*, with logits up to ±15 (the soft-cap is
the only thing stopping it from being worse). The gradient norm is 30, so clipping cuts every early update to
1/30th. It recovers, but spends its first 20 steps unlearning, and finishes at 3.7626 against the baseline's
3.6311. On a long run you'd lose more than steps: large early gradients are where instabilities start.

**P6.** Something else: no `nan`, no spike, it just stops learning.

```text
step     0 | loss 6.2383 | lr x1.00 | grad  0.27
step    10 | loss 6.0128 | lr x1.00 | grad  0.42
step    20 | loss 5.2888 | lr x1.00 | grad  0.10
step    50 | loss 4.7946 | lr x1.00 | grad  0.05
step   100 | val 4.5777 | ppl   97.29
```

Val 4.58 against 3.63: far worse than either break so far. The giveaway is the gradient norm: 0.05 by step 50,
almost ten times smaller than a healthy run. My reading (inferred, not measured here): the first huge steps pushed
many ReLU² units into their flat region, where the gradient is zero, and the model can't climb back out. Clipping and the logit soft-cap kept it
from going `nan`, which is why there's no crash to warn you. A collapsing gradient norm with a stalled loss is the
signature of a learning rate that was too high early.

## What to remember

- Check step 0 against `ln(vocab)` on every run.
- Warmup, stable, warmdown: know your multiplier at any step.
- Train/val gap shows memorisation; the gradient norm shows whether the optimizer is healthy.
- A bad run often doesn't crash. Compare against a short baseline before trusting a long one.

Next: [Lab 06 · Sampling and the KV cache](06-sampling-and-kv-cache.md).
