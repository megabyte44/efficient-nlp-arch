"""Minimal correctness checks — run with: python -m pytest tests/ -q
(from project root, with src/ on PYTHONPATH)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from model import GPT, resolve_layer_cfg


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


def test_mamba_forward_shape():
    cfg = make_cfg(attn_type="mamba", d_state=4, expand=2, d_conv=3)
    model = GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg["block_size"], cfg["vocab_size"])
    assert loss is None


def test_mamba_is_causal():
    # The selective scan only looks backward in time, so changing the last
    # token must not move any earlier position's logits, same as attention.
    cfg = make_cfg(attn_type="mamba", d_state=4, expand=2, d_conv=3)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_chunked_full_attention_matches_non_chunked():
    # Chunked (online-softmax, blocked) attention is exact -- same math,
    # different execution strategy -- so it must match bit-for-bit-close
    # given identical weights. block_size=8, chunk_size=3 exercises an
    # uneven last block (3,3,2).
    torch.manual_seed(0)
    cfg = make_cfg(attn_type="full")
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    with torch.no_grad():
        logits_plain, _ = model(idx)
        for block in model.blocks:
            block.attn.chunked = True
            block.attn.chunk_size = 3
        logits_chunked, _ = model(idx)
    assert torch.allclose(logits_plain, logits_chunked, atol=1e-5)


def test_chunked_local_attention_matches_non_chunked():
    # Same equivalence check for the windowed variant, with window_size not
    # a multiple of chunk_size to exercise the key-block-skipping logic.
    torch.manual_seed(0)
    cfg = make_cfg(attn_type="local", window_size=3)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    with torch.no_grad():
        logits_plain, _ = model(idx)
        for block in model.blocks:
            block.attn.chunked = True
            block.attn.chunk_size = 3
        logits_chunked, _ = model(idx)
    assert torch.allclose(logits_plain, logits_chunked, atol=1e-5)


def test_unimplemented_attn_type_raises():
    cfg = make_cfg(attn_type="sparse")
    try:
        GPT(cfg)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_local_attn_is_causal():
    cfg = make_cfg(attn_type="local", window_size=3)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_local_attn_ignores_tokens_outside_window():
    # Changing a token further back than window_size should not move a later
    # position's logits at all -- that's the entire point of windowing.
    # Single layer, so the receptive field is exactly window_size (with
    # multiple layers it grows by ~window_size per layer, like dilated
    # convs, which would make this direct a check invalid).
    cfg = make_cfg(attn_type="local", window_size=3, n_layer=1)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_edited = idx.clone()
        idx_edited[:, 0] = (idx_edited[:, 0] + 1) % cfg["vocab_size"]
        logits_edited, _ = model(idx_edited)
    # position 0 is the edited token itself, so only compare from position
    # window_size onward, which should be fully unaffected by position 0.
    window_size = cfg["window_size"]
    assert torch.allclose(
        logits_full[:, window_size:], logits_edited[:, window_size:], atol=1e-5
    )


def test_dilated_attn_is_causal():
    cfg = make_cfg(attn_type="dilated", segment_length=4, dilation_rate=2)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_dilated_attn_ignores_tokens_outside_segment_and_residue():
    # Position 1 is in segment [0,3] (segment_length=4), residue 1 mod 2.
    # Editing position 0 (different residue, same segment) or position 4+
    # (different segment) must not move position 1's logits at all.
    # Single layer, so the receptive field is exactly one segment/residue
    # class (with multiple layers it would compound across layers).
    cfg = make_cfg(attn_type="dilated", segment_length=4, dilation_rate=2, n_layer=1)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_edited = idx.clone()
        idx_edited[:, 0] = (idx_edited[:, 0] + 1) % cfg["vocab_size"]
        logits_edited, _ = model(idx_edited)
    # position 1 shares dilation_rate=2's residue-1 class with position 0
    # only if 0 % 2 == 1 % 2, which is false -- so position 1 is unaffected.
    assert torch.allclose(
        logits_full[:, 1:2], logits_edited[:, 1:2], atol=1e-5
    )


def test_dilated_attn_matches_full_when_segment_covers_block_and_no_dilation():
    # Sanity check: segment_length == block_size and dilation_rate == 1
    # degenerates to plain full causal attention (same mask), so outputs
    # should match exactly given identical weights.
    torch.manual_seed(0)
    cfg_full = make_cfg(attn_type="full")
    torch.manual_seed(0)
    cfg_dilated = make_cfg(
        attn_type="dilated", segment_length=cfg_full["block_size"], dilation_rate=1
    )

    torch.manual_seed(42)
    model_full = GPT(cfg_full)
    torch.manual_seed(42)
    model_dilated = GPT(cfg_dilated)
    model_full.eval()
    model_dilated.eval()

    idx = torch.randint(0, cfg_full["vocab_size"], (1, cfg_full["block_size"]))
    with torch.no_grad():
        logits_full, _ = model_full(idx)
        logits_dilated, _ = model_dilated(idx)
    assert torch.allclose(logits_full, logits_dilated, atol=1e-5)


def test_local_attn_matches_full_when_window_covers_block():
    # Sanity check: a window as wide as the whole sequence degenerates to
    # full causal attention (same mask), so outputs should match exactly
    # given identical weights.
    torch.manual_seed(0)
    cfg_full = make_cfg(attn_type="full")
    torch.manual_seed(0)
    cfg_local = make_cfg(attn_type="local", window_size=cfg_full["block_size"])

    torch.manual_seed(42)
    model_full = GPT(cfg_full)
    torch.manual_seed(42)
    model_local = GPT(cfg_local)
    model_full.eval()
    model_local.eval()

    idx = torch.randint(0, cfg_full["vocab_size"], (1, cfg_full["block_size"]))
    with torch.no_grad():
        logits_full, _ = model_full(idx)
        logits_local, _ = model_local(idx)
    assert torch.allclose(logits_full, logits_local, atol=1e-5)


def test_s4_forward_shape():
    cfg = make_cfg(attn_type="s4", d_state=4)
    model = GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg["block_size"], cfg["vocab_size"])
    assert loss is None


def test_s4_is_causal():
    # The SSM recurrence only runs forward in time, so it should have the
    # same causality property as attention: changing the last token must
    # not move any earlier position's logits.
    cfg = make_cfg(attn_type="s4", d_state=4)
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)


def test_layer_recipe_backward_compatible():
    # Without layer_recipe, every layer must resolve to cfg unchanged --
    # the no-op guard that keeps all six existing single-mixer-type configs
    # behaving exactly as before this seam was added.
    cfg = make_cfg(attn_type="s4", d_state=4)
    for i in range(cfg["n_layer"]):
        assert resolve_layer_cfg(cfg, i) is cfg


def test_hybrid_forward_shape():
    cfg = make_cfg(
        n_layer=3,
        layer_recipe=["mamba", "full", "s4"],
        d_state=4,
        expand=2,
        d_conv=3,
    )
    model = GPT(cfg)
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    logits, loss = model(idx)
    assert logits.shape == (2, cfg["block_size"], cfg["vocab_size"])
    assert loss is None


def test_hybrid_is_causal():
    # Same causality check applied to every other mixer, run across a mixed
    # stack -- a regression guard that interleaving different mixer types
    # doesn't break causality end to end.
    cfg = make_cfg(
        n_layer=3,
        layer_recipe=["mamba", "full", "s4"],
        d_state=4,
        expand=2,
        d_conv=3,
    )
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg["vocab_size"], (1, cfg["block_size"]))
    with torch.no_grad():
        logits_full, _ = model(idx)
        idx_truncated = idx.clone()
        idx_truncated[:, -1] = (idx_truncated[:, -1] + 1) % cfg["vocab_size"]
        logits_changed, _ = model(idx_truncated)
    assert torch.allclose(logits_full[:, :-1], logits_changed[:, :-1], atol=1e-5)
