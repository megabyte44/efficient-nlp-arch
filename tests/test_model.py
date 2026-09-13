"""Minimal correctness checks — run with: python -m pytest tests/ -q
(from project root, with src/ on PYTHONPATH)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from model import GPT, resolve_layer_cfg, mamba_sequential_scan, mamba_chunked_scan


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


# --- Phase 1 of the Mamba-scan diagnostic experiment (see ROADMAP.md): does
# the experimental mamba_chunked_scan compute the same thing as the existing
# mamba_sequential_scan? Not wired into MambaMixer.forward -- these tests
# compare the two functions directly. ---


def _random_mamba_scan_inputs(seed, batch=2, T=17, d_inner=6, d_state=4, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    delta = F.softplus(torch.randn(batch, T, d_inner, generator=g))
    A_log = torch.log(torch.arange(1, d_state + 1).float().expand(d_inner, d_state).clone())
    A = -torch.exp(A_log)
    B_seq = torch.randn(batch, T, d_state, generator=g)
    x_in = torch.randn(batch, T, d_inner, generator=g)
    C_seq = torch.randn(batch, T, d_state, generator=g)
    D = torch.ones(d_inner)
    if requires_grad:
        for t in (delta, B_seq, x_in, C_seq):
            t.requires_grad_(True)
    return delta, A, B_seq, x_in, C_seq, D


def test_chunked_mamba_scan_matches_sequential_across_seqlens_and_chunk_sizes():
    # mamba_chunked_scan is a closed-form reorganization of the exact same
    # recurrence mamba_sequential_scan steps through one timestep at a time
    # -- not an approximation -- so it must agree within float32 tolerance
    # for any (sequence length, chunk size) pair, including ones where the
    # last chunk is uneven (T not a multiple of chunk_size).
    atol = 1e-4
    for seed in (0, 1, 2):
        for T in (8, 17, 33, 64):
            for chunk_size in (8, 16, 32, 64):
                delta, A, B_seq, x_in, C_seq, D = _random_mamba_scan_inputs(seed, T=T)
                with torch.no_grad():
                    y_seq = mamba_sequential_scan(delta, A, B_seq, x_in, C_seq, D)
                    y_chunked = mamba_chunked_scan(delta, A, B_seq, x_in, C_seq, D, chunk_size)
                max_diff = (y_seq - y_chunked).abs().max().item()
                assert torch.allclose(y_seq, y_chunked, atol=atol), (
                    f"seed={seed} T={T} chunk_size={chunk_size}: max abs diff={max_diff}"
                )


def test_chunked_mamba_scan_gradients_match_sequential():
    # Training backprops through the scan, so the two implementations must
    # agree on gradients, not just forward outputs.
    atol = 1e-3
    delta, A, B_seq, x_in, C_seq, D = _random_mamba_scan_inputs(0, T=20, requires_grad=True)

    y_seq = mamba_sequential_scan(delta, A, B_seq, x_in, C_seq, D)
    y_seq.sum().backward()
    grads_seq = {name: t.grad.clone() for name, t in zip(
        ("delta", "B_seq", "x_in", "C_seq"), (delta, B_seq, x_in, C_seq)
    )}
    for t in (delta, B_seq, x_in, C_seq):
        t.grad = None

    y_chunked = mamba_chunked_scan(delta, A, B_seq, x_in, C_seq, D, chunk_size=8)
    y_chunked.sum().backward()
    grads_chunked = {name: t.grad.clone() for name, t in zip(
        ("delta", "B_seq", "x_in", "C_seq"), (delta, B_seq, x_in, C_seq)
    )}

    for name in grads_seq:
        max_diff = (grads_seq[name] - grads_chunked[name]).abs().max().item()
        assert torch.allclose(grads_seq[name], grads_chunked[name], atol=atol), (
            f"gradient mismatch for {name}: max abs diff={max_diff}"
        )


def test_chunked_mamba_scan_matches_sequential_after_training_steps():
    # Correctness at random init isn't sufficient: once weights (and thus
    # delta) have moved during training, decay magnitudes change -- exactly
    # where a numerically unstable chunked implementation would first show
    # divergence. Trains a real MambaMixer for a few steps, then reproduces
    # its pre-scan computation (embeddings -> ln1 -> in_proj -> conv ->
    # x_proj -> softplus) to get real delta/B/C/x_in tensors, and checks the
    # two scans still agree on those.
    cfg = make_cfg(attn_type="mamba", d_state=4, expand=2, d_conv=3, n_layer=1)
    torch.manual_seed(0)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(10):
        idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
        targets = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
        _, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    block = model.blocks[0]
    mixer = block.attn
    idx = torch.randint(0, cfg["vocab_size"], (2, cfg["block_size"]))
    with torch.no_grad():
        pos = torch.arange(cfg["block_size"])
        x = model.tok_emb(idx) + model.pos_emb(pos)
        mixer_in = block.ln1(x)

        x_in, _z = mixer.in_proj(mixer_in).chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = F.pad(x_in, (mixer.d_conv - 1, 0))
        x_in = F.silu(mixer.conv1d(x_in))
        x_in = x_in.transpose(1, 2)
        delta, B_seq, C_seq = torch.split(
            mixer.x_proj(x_in), [mixer.d_inner, mixer.d_state, mixer.d_state], dim=-1
        )
        delta = F.softplus(delta)
        A = -torch.exp(mixer.A_log)

        y_seq = mamba_sequential_scan(delta, A, B_seq, x_in, C_seq, mixer.D)
        y_chunked = mamba_chunked_scan(delta, A, B_seq, x_in, C_seq, mixer.D, chunk_size=4)

    max_diff = (y_seq - y_chunked).abs().max().item()
    assert torch.allclose(y_seq, y_chunked, atol=1e-4), f"post-training mismatch: max abs diff={max_diff}"
