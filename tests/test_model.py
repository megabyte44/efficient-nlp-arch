"""Minimal correctness checks — run with: python -m pytest tests/ -q
(from project root, with src/ on PYTHONPATH)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from model import GPT


def make_cfg(**overrides):
    cfg = dict(
        attn_type="full",
        n_layer=2,
        n_head=2,
        n_embd=16,
        block_size=8,
        dropout=0.0,
        bias=False,
        vocab_size=13,
    )
    cfg.update(overrides)
    return cfg


def test_forward_shape():
    cfg = make_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg["block_size"], cfg["vocab_size"])
    assert loss is None


def test_loss_computed_with_targets():
    cfg = make_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    targets = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0


def test_causality_future_tokens_dont_affect_earlier_logits():
    cfg = make_cfg()
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_unimplemented_attn_type_raises():
    cfg = make_cfg(attn_type="sparse")
    try:
        GPT(cfg)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
