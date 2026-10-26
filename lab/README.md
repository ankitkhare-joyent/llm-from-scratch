# Hands-on lab: the model, one component per sitting

## Fresh start

Run these from the repo root before Lab 00, and any time you want a clean slate. The whole sequence takes about
seven minutes on an M1, most of it training.

```bash
cd ~/workstuff/argocd-terraform-opentofu/llm-from-scratch
docker ps --format '{{.Names}}'   # nothing heavy should be running: training shares the Mac with Docker
mise run reset                    # deletes .venv, the download, tokenizer, token files, checkpoints and runs
mise install                      # Python 3.13 + uv, pinned in mise.toml
mise run setup                    # uv sync: torch, numpy, regex, pytest, ruff
mise run data                     # download TinyShakespeare (SHA-256 checked), train the tokenizer, tokenise
mise run train                    # 400 steps, ~5-6 min; writes checkpoints/shakespeare.pt
mise run test                     # ruff + 10 CPU tests; all should pass before you start breaking things
```

What I got from that sequence on 25 October 2026 (Apple M1, 16 GB):

```text
reset: removed .venv, data downloads, tokenizer, checkpoints and runs
tokenised 547,276 tokens -> train 492,548 / val 54,728
device=mps params=2,555,904 train_tokens=492,548
step     0 | val 6.2385 | ppl  512.09 | saved
step   100 | val 3.5772 | ppl   35.77 | saved
step   200 | val 3.1408 | ppl   23.12 | saved
step   300 | val 3.0608 | ppl   21.34 | saved
step   400 | val 3.0057 | ppl   20.20 | saved
done in 385s; best val loss 3.0057 -> checkpoints/shakespeare.pt
10 passed
```

Your last digits will differ a little (Metal kernels aren't bit-for-bit deterministic). Step 0 should be
`6.2385` exactly, and the final val loss should land around 3.00 to 3.06.

`mise run reset` never touches your source. If a "break it" step is still in place, it says so and tells you to
run `git restore src tests`.

## How the labs work

The lessons in [docs/](../docs) explain the design. This track has you poke each component until you could debug it
yourself. It follows the data: the whole model first, then text to tokens, batches, positions, attention, training,
generation, and finally the numbers at production scale.

Every lab has the same shape:

1. **Look**: run a helper against the real code and the trained checkpoint, and see what each part stores,
   computes and prints.
2. **Predict** before you run: each step asks a question (P1, P2, …). Write your guess down first. The answers, with
   the output I got from the fresh start above, are at the bottom of each file.
3. **Break it, then fix it**: at least one failure on purpose, so you've seen what it looks like before it happens
   for real. Some breaks are flags on a helper; others have you edit the source, watch a test fail, then
   `git restore` it.

```bash
glow lab/00-the-whole-model.md     # or: vim lab/00-the-whole-model.md
```

A tmux split works well: the lab on one side, a shell on the other.

| Lab | Component | You'll be able to explain | Changes anything? |
|---|---|---|---|
| [00](00-the-whole-model.md) | The whole model | every tensor shape; where 2.56 M parameters live (not where you'd guess); what GQA saves | no |
| [01](01-tokenizer.md) | Tokenizer | why `ROMEO` costs more than `Romeo`; vocab size vs compression; why the regex split exists; broken emoji when streaming | no |
| [02](02-data-and-batching.md) | Data and batching | inputs vs targets; epochs and tokens per parameter; what an off-by-one target bug looks like | trains 100 steps |
| [03](03-embeddings-and-rope.md) | Embeddings and RoPE | why scores depend only on distance; what `rope_base` sets; how much position information matters | trains 100 steps |
| [04](04-attention.md) | Attention | what the heads look at; heads sharing a KV group; what a missing causal mask does | edits `model.py`, trains 200 steps, then restores it |
| [05](05-training.md) | Training | why step 0 is 6.2385; bad initialisation; a learning rate that diverges | trains 100 steps, twice |
| [06](06-sampling-and-kv-cache.md) | Sampling and the KV cache | temperature and top-p as numbers; the cache's speed-up; a cache bug the tests catch | edits `model.py`, then restores it |
| [07](07-production-scale.md) | Production scale | 6ND compute, H100-days, why the KV cache limits serving; how far this Mac is from a real run | trains 20 steps of a bigger model |

Each lab is one sitting of 20 to 40 minutes. Do them in order: later labs reuse what you saw earlier.

## Helpers

All run on CPU except `lab:train`, which uses the GPU like `mise run train`.

| Command | What | Source |
|---|---|---|
| `mise run -q lab:shapes [-- --n-head 8 ...]` | tensor shapes through the model; parameter split; KV bytes | [shapes.py](shapes.py) |
| `mise run -q lab:tok -- [--vocab N] [--no-split] [--emoji] TEXT…` | encode text; train other tokenizers in memory | [tok.py](tok.py) |
| `mise run -q lab:batch` | one batch decoded as input/target pairs; the data budget | [batch.py](batch.py) |
| `mise run -q lab:rope [-- --base 100]` | RoPE scores at different absolute positions, same distance | [rope_check.py](rope_check.py) |
| `mise run -q lab:attn [-- --layer N --text "..."]` | per-head attention statistics from the checkpoint | [attn_map.py](attn_map.py) |
| `mise run -q lab:next -- "PROMPT"` | next-token probabilities, entropy and top-p size per temperature | [next_token.py](next_token.py) |
| `mise run -q lab:train -- --run NAME [flags]` | 100-step training; any `tinyllm.train` flag, plus `--copy-targets`, `--no-rope`, `--head-std X` | [quick_train.py](quick_train.py) |
| `mise run -q lab:scale [-- --tok-per-s N]` | compute and KV-cache arithmetic for real models | [scale.py](scale.py) |
| `mise run bench-kv` | generation with and without the KV cache | [../scripts/bench_kv.py](../scripts/bench_kv.py) |

Lab runs write `checkpoints/lab-*.pt` and `runs/lab-*/`; they never overwrite `shakespeare.pt`. Everything they
create is git-ignored and removed by `mise run reset`.

## If something is off

- `FileNotFoundError: .../data/train.bin` or `checkpoints/shakespeare.pt`: run the fresh start; the helpers need
  the data and the trained checkpoint.
- Training is much slower than ~1 step per second: something else is using the GPU or memory. Check `docker ps` and
  Activity Monitor.
- A test fails before you've broken anything: `git status`. A break from an earlier lab is probably still in place;
  `git restore src tests`.
- Your numbers differ from the answers in the last digit or two: expected. Different by more than a few percent,
  or a different shape of result: worth a look.

## The model, with the lab that covers each part

```text
  "ROMEO:\nBut, soft!"                                      Lab 01  byte-level BPE, 512 tokens
          │ encode
          ▼
  [82, 79, 77, 69, 79, 58, …]   uint16 in train.bin / val.bin   Lab 02  x = window[:-1], y = window[1:]
          │ batch (32 x 256)
          ▼
  ┌───────────────────────────────────────────────┐
  │ wte 512 x 192 → RMSNorm                       │      Lab 00  shapes and parameters
  ├───────────────────────────────────────────────┤
  │ 6 x block                                     │
  │   RMSNorm → attention: 6 q heads, 2 kv heads  │      Lab 03  RoPE on q and k
  │             RoPE, QK-norm, causal mask        │      Lab 04  what the heads look at
  │   RMSNorm → MLP 192 → 768 → 192, ReLU²        │
  ├───────────────────────────────────────────────┤
  │ RMSNorm → lm_head 192 x 512 → soft-cap        │
  └───────────────┬───────────────────────┬───────┘
                  │ logits vs y           │ last position's logits
                  ▼                       ▼
     cross-entropy → Muon/AdamW     temperature, top-p, KV cache       Lab 06
     Lab 05  training                     │
                  │                       ▼
                  ▼                 generated text
     checkpoints/shakespeare.pt
                                                     Lab 07  the same, at 8 B to 70 B parameters
```
