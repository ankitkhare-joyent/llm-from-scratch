# Lab 01 · The tokenizer: what the model actually sees

One sitting, about 30 minutes. Background: [docs/01](../docs/01-tokenizer.md). Code:
[tokenizer.py](../src/tinyllm/tokenizer.py), [lab/tok.py](tok.py).

The model never sees letters, only the ids this tokenizer produces. After this lab you can explain:

- why `ROMEO` and `Romeo` cost a different number of tokens
- what vocabulary size trades against, with measured numbers
- why the regex pre-split exists
- why streaming output must decode the whole sequence, not one token at a time

## Before you start

```bash
mise run -q lab:tok             # prints bytes/token for the trained 512-token tokenizer
vim +/'def train' src/tinyllm/tokenizer.py
```

## 1. Same word, different costs

**Predict (P1):** how many tokens for each of `ROMEO:`, `Romeo:`, ` romeo` (leading space), `1599`, and
`Wherefore art thou`? Which is the most expensive per byte?

```bash
mise run -q lab:tok -- "ROMEO:" "Romeo:" " romeo" "1599" "Wherefore art thou"
```

## 2. Vocabulary size

The lab's tokenizer has 512 tokens: 256 raw bytes plus 256 learned merges.

**Predict (P2):** for 256 tokens (bytes only, no merges) and for 1,024 tokens, estimate the corpus size in tokens
(it's 547,276 at 512) and how many tokens `Wherefore art thou` becomes. What does training a 1,024-token tokenizer
cost in time compared with 512 (about 13 s)?

```bash
mise run -q lab:tok -- --vocab 256 "Wherefore art thou"
mise run -q lab:tok -- --vocab 1024 "Wherefore art thou"
```

These train a fresh tokenizer in memory only; `data/tokenizer.json` is not touched.

## Break it, then fix it

**Break 1: no pre-split.** `--no-split` trains on whole lines instead of the GPT-4 regex chunks, so merges can cross
spaces and punctuation. Both runs use the first 200,000 characters and 300 tokens to keep it fast.

**Predict (P3):** which tokenizer compresses better? What do the learned tokens look like without the split?

```bash
mise run -q lab:tok -- --vocab 300 --chars 200000 "and the king"
mise run -q lab:tok -- --no-split --vocab 300 --chars 200000 "and the king"
```

**Fix:** keep the pre-split (it's the default). The point to take away is in the answer.

**Break 2: decoding one token at a time.** A streaming chat UI receives tokens one by one. `--emoji` encodes
`Romeo 🌹 Juliet`, then decodes each id separately and joins the strings, and also decodes all ids at once.

**Predict (P4):** do the two decodes match? How many tokens is the rose?

```bash
mise run -q lab:tok -- --emoji
```

**Fix:** decode the running sequence and print only the new suffix. Read the loop at the bottom of
[sample.py](../src/tinyllm/sample.py) (`shown = text`): that's what `mise run sample` does.

## Check your predictions

**P1.**

```text
'ROMEO:': 6 bytes -> 6 tokens        ['R', 'O', 'M', 'E', 'O', ':']
'Romeo:': 6 bytes -> 4 tokens        ['R', 'ome', 'o', ':']
' romeo': 6 bytes -> 4 tokens        [' ', 'r', 'ome', 'o']
'1599': 4 bytes -> 4 tokens          ['1', '5', '9', '9']
'Wherefore art thou': 18 bytes -> 9 tokens
                                     ['W', 'he', 're', 'f', 'ore', ' a', 'r', 't', ' thou']
```

`ROMEO:` and `1599` cost one token per byte: the 256 merges went to common lower-case pieces (`' t'`, `'he'`,
`' thou'`), and capitals and digits are rarer. Speaker names in capitals open every speech in the corpus, so the
model spends a lot of its context on them. Digits also show the pre-split: GPT-4's pattern groups digits in runs of up
to three, so `1599` would be `'159'` + `'9'` in a big vocabulary, never one token for the whole year. This is one
reason small models are bad at arithmetic.

**P2.**

| vocab | train time | corpus tokens | bytes/token | `Wherefore art thou` |
|---|---|---|---|---|
| 256 | 0.2 s | 1,115,394 | 1.00 | 18 tokens |
| 512 | ~13 s | 547,276 | 2.04 | 9 tokens |
| 1,024 | 31.3 s | 428,114 | 2.61 | 4: `'Where' 'fore' ' art' ' thou'` |

Doubling the vocabulary from 512 to 1,024 cut the corpus by only 22%: each merge saves less than the one before
(Zipf's law). The longest learned tokens at 1,024 are character names: `GLOUCESTER`, `' VINCENTIO'`, `CORIOLANUS`.
The trade-off: a 1,024 vocabulary doubles `wte` and `lm_head` (196,608 more parameters, 8% of this model) and splits
the training signal across more classes. At production scale the embedding share is tiny, so vocabularies are large
(64 k to 256 k).

**P3.**

```text
with pre-split:   1.42 bytes/token   longest: ' the', ' you', '.\n\n', ' to', 'hat', ' he'
no pre-split:     1.41 bytes/token   longest: ' the ', 'US:\n', 'and ', ' th', 'you', 'S:\n'
'and the king' -> ['a', 'nd', ' the', ' ', 'k', 'in', 'g']   vs   ['and ', 'th', 'e ', 'k', 'ing']
```

Compression is almost identical, so the split isn't about compression. Without it, tokens absorb the space *after*
a word (`'and '`, `'e '`) or glue a name's tail to the newline (`'US:\n'`, from `CORIOLANUS:` and friends).
The same word then gets different ids depending on what follows it, and the model has to learn every variant
separately. The pre-split makes `' the'` mean "the word the", wherever it appears.

**P4.** They don't match:

```text
token ids       [82, 352, 111, 32, 240, 159, 140, 185, 32, 74, 445, 105, 318]
decode each id  'Romeo ���� Juliet'
decode all ids  'Romeo 🌹 Juliet'
```

The rose is 4 tokens, `240 159 140 185`: its 4 UTF-8 bytes, because no merge covers them. None of those bytes is a
valid character alone, so each decodes to the replacement character `�`. Every tokenizer with byte fallback has this
property, including the ones behind production chat APIs, which is why streaming clients buffer partial characters.

## What to remember

- Token count, not character count, is what costs context and compute; it depends on case, digits and language.
- Bigger vocabularies shorten sequences with diminishing returns and cost embedding parameters.
- The pre-split keeps a word's token the same wherever it appears.
- Decode sequences, not single tokens.

Next: [Lab 02 · Data and batching](02-data-and-batching.md).
