# Lab 06 · Sampling and the KV cache: from logits to text

One sitting, about 30 minutes. Background: [docs/06](../docs/06-sampling-and-kv-cache.md). Code:
[sample.py](../src/tinyllm/sample.py), `KVCache` and `GPT.forward` in [model.py](../src/tinyllm/model.py),
[lab/next_token.py](next_token.py), [scripts/bench_kv.py](../scripts/bench_kv.py).

The trained model gives one probability per token. This lab looks at those probabilities directly, then at the cache
that makes generating from them cheap. After this lab you can explain:

- what temperature and top-p do to a real distribution, in numbers
- why the same top-p keeps 1 token in one place and 70 in another
- how much the KV cache saves, and what breaks if its bookkeeping is wrong

## Before you start

```bash
vim +/'def sample_next' src/tinyllm/sample.py
vim +/'class KVCache' src/tinyllm/model.py
```

## 1. A real next-token distribution

`lab:next` runs the prompt, takes the logits at the last position, and shows the top tokens at three temperatures,
with the distribution's entropy (`e^entropy` is "as uncertain as choosing among this many") and how many tokens
top-p 0.9 keeps.

**Predict (P1):** after `ROMEO:\nWhat light through yonder`, will the model be confident about the next token?
What happens to the top token's probability at temperature 0.5 and at 1.5?

```bash
mise run -q lab:next -- "ROMEO:\nWhat light through yonder"
```

**Predict (P2):** now `KING RICHARD`. How sure is it, and of what?

```bash
mise run -q lab:next -- --top 5 "KING RICHARD"
```

## 2. Temperature in text

**Predict (P3):** same seed, temperature 0.3 and then 1.5 with top-p off. Which one repeats itself, and which one
invents words?

```bash
mise run -q sample -- --seed 1 --tokens 60 --temperature 0.3
mise run -q sample -- --seed 1 --tokens 60 --temperature 1.5 --top-p 1.0
```

## 3. What the cache saves

**Predict (P4):** generating 250 tokens with and without the KV cache: how much faster is the cached version on this
CPU? Are the outputs identical?

```bash
mise run bench-kv
```

## Break it, then fix it

**Break: the cache never advances.** `GPT.forward` calls `kv_cache.advance(T)` after every pass so the next token is
written to the next slot and gets the next RoPE position. Comment it out:

```bash
sed -i '' 's/            kv_cache.advance(T)/            pass  # kv_cache.advance(T)/' src/tinyllm/model.py
git diff src/tinyllm/model.py
```

**Predict (P5):** which tests fail? Does sampling crash, or produce text? If text, what's wrong with it? Faster or
slower than before?

```bash
mise exec -- uv run pytest -q
mise run -q sample -- --seed 1 --tokens 60
mise run bench-kv
```

**Fix:**

```bash
git restore src/tinyllm/model.py
mise exec -- uv run pytest -q          # 10 passed
mise run -q sample -- --seed 1 --tokens 60
```

## Check your predictions

**P1.** Not confident: a spread-out distribution, and temperature reshapes it.

```text
temperature 0.5: entropy 2.52 nats (= choosing among ~12), top-p 0.9 keeps 18 tokens
    28.2%  ','       25.8%  ' is'      9.6%  ' w'      6.0%  ' c'
temperature 1:   entropy 3.98 nats (= choosing among ~53), top-p 0.9 keeps 53 tokens
     9.7%  ','        9.3%  ' is'      5.7%  ' w'      4.5%  ' c'
temperature 1.5: entropy 4.54 nats (= choosing among ~94), top-p 0.9 keeps 84 tokens
     5.1%  ','        4.9%  ' is'      3.5%  ' w'      3.0%  ' c'
```

The model doesn't know the quote. The line (`But, soft! what light through yonder window breaks?`) appears exactly
once in the 1.1 MB corpus (`grep -n "yonder window" data/tinyshakespeare.txt`), and a 2.6 M model doesn't memorise
a line it has seen six times among half a million tokens. So `,` and ` is` are generic guesses. Temperature divides the logits before the softmax: 0.5
doubles every gap and the top two tokens take over half the mass; 1.5 flattens it toward ~94 plausible tokens. The
*ranking* never changes, only how much the top choices dominate.

**P2.** Certain:

```text
temperature 0.5: entropy 0.00 nats, top-p 0.9 keeps 1 tokens     100.0%  ' I'
temperature 1:   entropy 0.14 nats, top-p 0.9 keeps 1 tokens      98.6%  ' I'
temperature 1.5: entropy 1.84 nats, top-p 0.9 keeps 70 tokens     75.7%  ' I'
```

In the corpus, `KING RICHARD` is followed by ` II:` 98 times and ` III:` 138 times, and never by anything else.
Both start with the token ` I` (the tokenizer spells `III:` as `' I' 'I' 'I' ':'`), so the model is right to be sure. Watch top-p at 1.5: the top token drops to 75.7%, below 0.9, so top-p has
to add the long tail until it reaches 90%, and keeps 70 tokens. That's top-p adapting to confidence: 1 token where
the model is sure, dozens where it isn't, and why it's usually paired with a moderate temperature.

**P3.**

```text
temperature 0.3:
ROMEO:
I would I do, my lord, I'll tell thee myself.

JULIET:
So, my lord, I'll not, I'll prove myself.

JULIET:
I will not call thee, I'll not

temperature 1.5, top-p off:
ROMEO:
Put tke of answin but what hath a meEven honourable.

Nurse:
A slow the fle! thou shouldst height pardon thy.-hine
```

Low temperature is fluent and repetitive (`I'll ... myself` twice, `JULIET:` twice). High temperature samples from
the tail and invents words (`tke`, `answin`, `meEven`). The default, 0.8 with top-p 0.95, sits between.

**P4.**

```text
with KV cache   :  0.40s     620 tok/s
without KV cache:  1.21s     207 tok/s
speed-up        : 3.0x
greedy outputs identical for the first 250 tokens: True
```

3x, with identical output. Without the cache, every new token reruns all earlier positions through all 6 layers.
The gap grows with context length and model size; for a production model serving 8 k-token conversations, decoding
without a cache would be hopeless.

**P5.** One test fails, sampling doesn't crash, and the text *almost* looks fine:

```text
FAILED tests/test_tinyllm.py::test_kv_cache_matches_full_forward - AssertionE...
1 failed, 9 passed

ROMEO:
Pister for then say to subet therely to ruck, ple of jure? impul, weariet I do he dishought. VOMELI:
SItising,
```

```text
with KV cache   :  0.34s     742 tok/s
without KV cache:  1.21s     206 tok/s
speed-up        : 3.6x
greedy outputs identical for the first 250 tokens: False
```

With `pos` stuck at 0, every decode step writes its K and V into slot 0, gets RoPE position 0, and attends to
`k[:, :, :1]`: only itself. The model has become a bigram model, predicting each token from the previous one alone.
That gives word-like fragments with no grammar and no line structure. Compare the fixed output for the same seed:

```text
ROMEO:
Put thanks, that will I not mine enemies,
If heild a ruthless the field,--a here prevents
```

And it got *faster* (742 tok/s), because attending to one token is cheaper than attending to all of them. A bug that
makes things faster and still produces plausible output is the dangerous kind; the equivalence test is the only thing
that caught it.

## What to remember

- Temperature rescales confidence without changing the ranking; top-p keeps a confidence-dependent number of tokens.
- The KV cache gives identical results at a fraction of the cost. Test that equivalence, because a broken cache
  still generates text.

Next: [Lab 07 · Production scale](07-production-scale.md).
