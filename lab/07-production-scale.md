# Lab 07 · Production scale: the arithmetic that decides real budgets

One sitting, about 25 minutes. Background: [docs/07](../docs/07-from-tiny-to-production.md). Code:
[lab/scale.py](scale.py), [lab/shapes.py](shapes.py).

Everything in Labs 00 to 06 is the real algorithm. What changes at production scale is the numbers, and a handful
of formulas let you estimate them before spending anything. After this lab you can explain:

- the `6 x params x tokens` rule for training compute, and what it says about Llama 3
- how far this Mac is from a real pre-training run, in years
- why the KV cache, not the weights, limits how many long conversations a GPU can serve
- what scaling up this model by 7.5x costs on your own machine

## Before you start

```bash
vim lab/scale.py          # MODELS: the published shapes of Llama 3 8B and 70B
```

## 1. Training compute

A forward pass costs about 2 FLOPs per parameter per token (one multiply, one add); the backward pass about twice
that. So training costs roughly `6 x N x D` for N parameters and D tokens.

**Predict (P1):** Llama 3 8B was trained on 15 trillion tokens. Roughly how many H100-days is that at an ideal
400 TFLOP/s per GPU? How many tokens per parameter did it see, against Chinchilla's "compute-optimal" 20?

```bash
mise run -q lab:scale
```

## 2. This Mac

**Predict (P2):** the fresh-start run trained at about 11,000 tokens/s. At that FLOP rate, how long would Llama 3 8B
take on this Mac: months, decades, or longer?

The middle of the same output answers it. If your own throughput differed, plug it in:
`mise run -q lab:scale -- --tok-per-s 9000`.

## 3. The KV cache at scale

Per token, the cache stores K and V for every layer and KV head: `2 x layers x kv_heads x head_dim x 2 bytes` in bf16.

**Predict (P3):** Llama 3 70B's weights are about 141 GB in bf16. How big is the KV cache for one 128 k-token
conversation, and for 32 of them at once?

## Break it, then fix it

**Break: scale this model up on the Mac.** 12 layers and 384 wide instead of 6 and 192, 20 steps only.

**Predict (P4):** how many parameters (use `lab:shapes` first if you like), and how many tokens per second against
the default's ~9,000 to 11,000? How long would the full 400-step run take?

```bash
mise run -q lab:shapes -- --n-layer 12 --n-embd 384 | tail -8
mise run -q lab:train -- --run lab-big --n-layer 12 --n-embd 384 --max-steps 20 --eval-interval 1000
```

**Fix:** there isn't one on this hardware, and that's the point. Production answers it with more and faster
hardware (thousands of GPUs, bf16 or FP8, fused kernels), sharded across machines with FSDP. For your own
experiments, the fix is the scaling-law habit: run a ladder of small models you *can* afford, and extrapolate.

## Check your predictions

**P1.**

```text
model           params    tokens tok/param      FLOPs  H100-days
this lab         2.6 M     3.3 M         1     50.4 T       0.00
Llama 3 8B       8.0 G    15.0 T      1875    720.0 Z  20,833.33
Llama 3 70B     70.6 G    15.0 T       212      6.4 Y 183,854.17

(Meta reports 1.46 M H100-hours for Llama 3 8B: about 3x this ideal, since real
 clusters reach 40% or less of peak and runs restart after failures)
```

About 21,000 ideal H100-days for the 8B, or about 61,000 in practice (Meta's 1.46 M GPU-hours), at 1,875 tokens per
parameter: almost 100x past the Chinchilla point. That's deliberate. Chinchilla minimises
*training* cost for a given loss; a model that will serve billions of requests is cheaper overall if it's small
and over-trained. Our lab sits at 1 token per parameter, and only gets there by repeating data.

**P2.** Longer than history:

```text
this lab at 11,000 tok/s does 168.96 GFLOP/s of model maths
(attention and the 512-way softmax add more; the M1 GPU peaks near 2,600 GFLOP/s fp32)
at that FLOP rate, Llama 3 8B would take 135,281 years on this Mac
at that FLOP rate, Llama 3 70B would take 1,193,858 years on this Mac
```

The lab uses well under 10% of the M1's peak, because a model this small is dominated by kernel-launch overhead and
small matrices; big models on big GPUs reach 40% or more. Even at the M1's full peak, the 8B would take thousands of
years. This is why "training from scratch" in practice means a tiny model like this one, or a fine-tune of someone
else's base model.

**P3.**

```text
model         per token    8k ctx  128k ctx  128k x 32 users
Llama 3 8B     131.1 kB    1.1 GB   17.2 GB         549.8 GB
Llama 3 70B    327.7 kB    2.7 GB   42.9 GB           1.4 TB
```

43 GB for one 128 k conversation: almost a third of the 70B's 141 GB of weights. Thirty-two of them need 1.4 TB,
ten times the weights. Weights are paid once per GPU; the cache is paid per conversation, which is why serving
engines are built around it: paged allocation, prefix sharing, 8-bit caches. And this is *with* GQA: Llama 3 has 8
KV heads against 64 query heads. Without GQA, multiply all of these by 8.

**P4.**

```text
device=mps params=19,267,584 train_tokens=492,548
step     0 | loss 6.2389 | lr x0.03 | grad  0.40 | 504 tok/s
step    10 | loss 5.1464 | lr x0.37 | grad  1.05 | 1,805 tok/s
step    20 | val 4.1703 | ppl   64.73 | saved
done in 91s
```

19.3 M parameters, 7.5x the default, and about 1,800 tokens/s against ~9,000 in the same session: 5x slower. The
full 400-step run would take around half an hour, and to train it at the same 1.3 tokens per parameter you'd need
7.5x the data, which TinyShakespeare doesn't have. Throughput fell less than the parameter count grew because the
small model wasn't using the GPU fully; bigger matrices use it better. That's the same effect, at the other end,
that makes large GPU clusters efficient.

## What to remember

- Training compute ≈ `6 x params x tokens`. Real runs reach a fraction of peak; budget about 3x the ideal.
- Production models train far past 20 tokens per parameter on purpose.
- For long-context serving, the KV cache is the memory bill. GQA and cache engineering exist because of it.
- On a laptop, the lever is small models and many short experiments. That's what this repo is.

Back to the [lab index](README.md). After this, the natural next step is
[nanochat](https://github.com/karpathy/nanochat), which runs the whole pipeline, including fine-tuning and RL, on
one 8-GPU node.
