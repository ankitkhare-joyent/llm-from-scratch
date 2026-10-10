"""Fast CPU tests for every stage of the pipeline. No dataset download needed."""

from __future__ import annotations

import math

import pytest
import torch

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.model import GPT, KVCache
from tinyllm.sample import generate, sample_next
from tinyllm.tokenizer import BPETokenizer
from tinyllm.train import build_optimizers, lr_multiplier

TEXT = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n" * 20
TINY = ModelConfig(vocab_size=300, seq_len=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(TEXT, 300)


def test_tokenizer_round_trips_unseen_text(tok, tmp_path):
    for s in [TEXT, "unseen wörds ✓ 12345\r\n", ""]:
        assert tok.decode(tok.encode(s)) == s
    tok.save(tmp_path / "t.json")
    again = BPETokenizer.load(tmp_path / "t.json")
    assert again.encode(TEXT) == tok.encode(TEXT)


def test_tokenizer_compresses(tok):
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2


def test_initial_loss_is_uniform_guess():
    torch.manual_seed(0)
    model = GPT(TINY)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.seq_len))
    _, loss = model(x, x)
    assert loss.item() == pytest.approx(math.log(TINY.vocab_size), abs=0.01)


def test_model_is_causal():
    """Changing a future token must not change predictions for earlier positions."""
    torch.manual_seed(0)
    model = GPT(TINY)
    for p in model.parameters():  # zero-init projections would hide leaks; randomise
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randint(0, TINY.vocab_size, (1, TINY.seq_len))
    y = x.clone()
    y[0, 20] = (y[0, 20] + 1) % TINY.vocab_size
    a, _ = model(x)
    b, _ = model(y)
    torch.testing.assert_close(a[:, :20], b[:, :20])
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randint(0, TINY.vocab_size, (2, 24))
    full, _ = model(x)
    cache = KVCache(TINY, 2, torch.device("cpu"), torch.float32)
    prefill, _ = model(x[:, :16], kv_cache=cache)
    steps = [model(x[:, t : t + 1], kv_cache=cache)[0] for t in range(16, 24)]
    torch.testing.assert_close(torch.cat([prefill, *steps], dim=1), full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("optimizer", ["muon", "adamw"])
def test_model_can_overfit_one_batch(optimizer):
    torch.manual_seed(0)
    model = GPT(TINY)
    tc = TrainConfig(optimizer=optimizer)
    opts = build_optimizers(model, tc)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.seq_len + 1))
    first = None
    for _ in range(60):
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        for o in opts:
            o.step()
        model.zero_grad()
        first = first if first is not None else loss.item()
    assert loss.item() < first / 2


def test_lr_schedule_shape():
    tc = TrainConfig(max_steps=100, warmup_steps=10, warmdown_ratio=0.4)
    assert lr_multiplier(0, tc) == pytest.approx(0.1)
    assert lr_multiplier(30, tc) == 1.0
    assert lr_multiplier(80, tc) == pytest.approx(0.5)  # halfway through warmdown (60..100)
    assert lr_multiplier(100, tc) == pytest.approx(0.0)


def test_sampling_rules():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.9])
    assert sample_next(logits, temperature=0) == 1
    gen = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, top_k=2, generator=gen) for _ in range(200)}
    assert picks <= {1, 3}
    picks = {sample_next(logits, top_p=0.3, generator=gen) for _ in range(200)}
    assert picks == {1}


def test_generate_runs_past_the_context_window():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    out = list(generate(model, [1, 2, 3], max_new_tokens=TINY.seq_len * 2, seed=0))
    assert len(out) == TINY.seq_len * 2
    assert all(0 <= t < TINY.vocab_size for t in out)
