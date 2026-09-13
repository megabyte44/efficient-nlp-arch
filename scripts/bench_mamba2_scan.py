"""Phase 4 of the Mamba-scan investigation (see ROADMAP.md): benchmarks
`mamba2_sequential_scan` vs `mamba2_chunked_scan` (the Mamba-2/SSD-style,
scalar-per-head-A recurrence -- a distinct architecture from MambaMixer's
Mamba-1/S6, not a modification of it) at the same sizes Phase 2's
`bench_mamba_scan.py` found broke down for the full per-(channel, state)
diagonal-A chunked scan (wider model, larger chunk sizes), to check whether
restricting A to scalar-per-head actually removes that blowup.

Run: python scripts/bench_mamba2_scan.py
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import mamba2_sequential_scan, mamba2_chunked_scan  # noqa: E402
from utils import get_device  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Same three points as scripts/bench_mamba_scan.py (Phase 2), so results are
# directly comparable: mamba_tiny's actual shape, a longer-sequence variant,
# and the wider-model point where Phase 2's chunked scan blew up badly
# (chunk=16/32 forward: 1735ms / 4226ms, vs sequential's 26.4ms).
SIZE_CONFIGS = [
    {"name": "mamba_tiny (T=128, d_model=128)", "batch": 8, "T": 128, "n_embd": 128, "expand": 2, "n_heads": 4, "d_state": 16},
    {"name": "longer seq (T=512, d_model=128)", "batch": 8, "T": 512, "n_embd": 128, "expand": 2, "n_heads": 4, "d_state": 16},
    {"name": "wider model (T=128, d_model=256)", "batch": 8, "T": 128, "n_embd": 256, "expand": 2, "n_heads": 4, "d_state": 16},
]
CHUNK_SIZES = [8, 16, 32, 64]
N_WARMUP = 3
N_ITERS = 10


def make_inputs(device, batch, T, n_heads, d_inner, d_state, seed, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    delta = F.softplus(torch.randn(batch, T, n_heads, generator=g)).to(device)
    A = (-torch.exp(torch.arange(1, n_heads + 1).float())).to(device)  # (n_heads,)
    B_seq = torch.randn(batch, T, d_state, generator=g).to(device)
    x_in = torch.randn(batch, T, d_inner, generator=g).to(device)
    C_seq = torch.randn(batch, T, d_state, generator=g).to(device)
    D = torch.ones(d_inner, device=device)
    if requires_grad:
        for t in (delta, B_seq, x_in, C_seq):
            t.requires_grad_(True)
    return delta, A, B_seq, x_in, C_seq, D


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_forward_latency(fn, args, device, n_warmup=N_WARMUP, n_iters=N_ITERS):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(*args)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(*args)
        sync(device)
        elapsed = time.perf_counter() - t0
    return (elapsed / n_iters) * 1000


def measure_forward_backward_latency(fn, make_args, device, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        args = make_args()
        fn(*args).sum().backward()

    fwd_total, bwd_total = 0.0, 0.0
    for _ in range(n_iters):
        args = make_args()
        sync(device)
        t0 = time.perf_counter()
        y = fn(*args)
        sync(device)
        t1 = time.perf_counter()
        y.sum().backward()
        sync(device)
        t2 = time.perf_counter()
        fwd_total += t1 - t0
        bwd_total += t2 - t1
    return (fwd_total / n_iters) * 1000, (bwd_total / n_iters) * 1000


def measure_peak_memory_mb(fn, make_args, device):
    if device.type != "cuda":
        return None
    args = make_args()
    torch.cuda.reset_peak_memory_stats(device)
    fn(*args).sum().backward()
    sync(device)
    return torch.cuda.max_memory_allocated(device) / 1e6


def _is_oom(exc):
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return "out of memory" in str(exc).lower()


def run():
    device = get_device()
    print(f"device: {device}", flush=True)
    rows = []

    for size_cfg in SIZE_CONFIGS:
        batch, T, n_embd, expand, n_heads, d_state = (
            size_cfg["batch"], size_cfg["T"], size_cfg["n_embd"], size_cfg["expand"],
            size_cfg["n_heads"], size_cfg["d_state"],
        )
        d_inner = expand * n_embd

        print(f"\n=== {size_cfg['name']} (batch={batch}, d_inner={d_inner}, n_heads={n_heads}) ===", flush=True)
        print("  sequential ... ", end="", flush=True)

        seq_args = make_inputs(device, batch, T, n_heads, d_inner, d_state, seed=0)
        y_seq_ref = mamba2_sequential_scan(*seq_args, n_heads).detach()

        seq_fwd_ms = measure_forward_latency(
            lambda *a: mamba2_sequential_scan(*a, n_heads), seq_args, device
        )
        _, seq_bwd_ms = measure_forward_backward_latency(
            lambda *a: mamba2_sequential_scan(*a, n_heads),
            lambda: make_inputs(device, batch, T, n_heads, d_inner, d_state, seed=0, requires_grad=True),
            device,
        )
        seq_peak_mem = measure_peak_memory_mb(
            lambda *a: mamba2_sequential_scan(*a, n_heads),
            lambda: make_inputs(device, batch, T, n_heads, d_inner, d_state, seed=0, requires_grad=True),
            device,
        )
        tokens_per_sec = (batch * T) / (seq_fwd_ms / 1000)
        print(f"fwd={seq_fwd_ms:.2f}ms bwd={seq_bwd_ms:.2f}ms", flush=True)

        rows.append({
            "size": size_cfg["name"], "impl": "sequential", "chunk_size": None,
            "seq_len": T, "batch_size": batch, "d_model": n_embd, "n_heads": n_heads, "d_state": d_state,
            "forward_ms": seq_fwd_ms, "backward_ms": seq_bwd_ms,
            "peak_vram_mb": seq_peak_mem, "tokens_per_sec": tokens_per_sec,
            "numerical_error_max_abs": 0.0,
        })

        for chunk_size in CHUNK_SIZES:
            print(f"  chunked (chunk_size={chunk_size}) ... ", end="", flush=True)
            try:
                y_chunked = mamba2_chunked_scan(*seq_args, n_heads, chunk_size).detach()
                error = (y_seq_ref - y_chunked).abs().max().item()

                chunk_fwd_ms = measure_forward_latency(
                    mamba2_chunked_scan, seq_args + (n_heads, chunk_size), device
                )
                _, chunk_bwd_ms = measure_forward_backward_latency(
                    lambda *a: mamba2_chunked_scan(*a, n_heads, chunk_size),
                    lambda: make_inputs(device, batch, T, n_heads, d_inner, d_state, seed=0, requires_grad=True),
                    device,
                )
                chunk_peak_mem = measure_peak_memory_mb(
                    lambda *a: mamba2_chunked_scan(*a, n_heads, chunk_size),
                    lambda: make_inputs(device, batch, T, n_heads, d_inner, d_state, seed=0, requires_grad=True),
                    device,
                )
                chunk_tokens_per_sec = (batch * T) / (chunk_fwd_ms / 1000)
                print(f"fwd={chunk_fwd_ms:.2f}ms bwd={chunk_bwd_ms:.2f}ms err={error:.2e}", flush=True)

                rows.append({
                    "size": size_cfg["name"], "impl": "chunked", "chunk_size": chunk_size,
                    "seq_len": T, "batch_size": batch, "d_model": n_embd, "n_heads": n_heads, "d_state": d_state,
                    "forward_ms": chunk_fwd_ms, "backward_ms": chunk_bwd_ms,
                    "peak_vram_mb": chunk_peak_mem, "tokens_per_sec": chunk_tokens_per_sec,
                    "numerical_error_max_abs": error,
                })
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                print("OOM -- skipped", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append({
                    "size": size_cfg["name"], "impl": "chunked", "chunk_size": chunk_size,
                    "seq_len": T, "batch_size": batch, "d_model": n_embd, "n_heads": n_heads, "d_state": d_state,
                    "forward_ms": None, "backward_ms": None,
                    "peak_vram_mb": None, "tokens_per_sec": None,
                    "numerical_error_max_abs": None, "note": "OOM",
                })

    return rows


def _fmt(value, spec):
    return format(value, spec) if value is not None else "n/a"


def print_and_save(rows):
    table = [
        [
            r["size"], r["impl"], r["chunk_size"],
            _fmt(r["forward_ms"], ".3f"), _fmt(r["backward_ms"], ".3f"),
            _fmt(r["peak_vram_mb"], ".2f"),
            _fmt(r["tokens_per_sec"], ",.0f"), _fmt(r["numerical_error_max_abs"], ".2e"),
            r.get("note", ""),
        ]
        for r in rows
    ]
    headers = ["size", "impl", "chunk_size", "fwd_ms", "bwd_ms", "peak_vram_mb", "tokens/sec", "max_abs_err", "note"]
    print(tabulate(table, headers=headers))

    out_dir = os.path.join(ROOT, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mamba2_scan_bench.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = [
        "# Mamba-2/SSD-style scan benchmark (Phase 4 diagnostic)",
        "",
        "`mamba2_sequential_scan` vs `mamba2_chunked_scan` (scalar-per-head A), "
        "in isolation. Same sizes as `experiments/mamba_scan_bench.md` (Phase 2, "
        "full per-(channel,state) diagonal A) for direct comparison. See "
        "ROADMAP.md for the experiment this is part of.",
        "",
        "| size | impl | chunk_size | fwd_ms | bwd_ms | peak_vram_mb | tokens/sec | max_abs_err | note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {size} | {impl} | {chunk_size} | {fwd} | {bwd} | {mem} | {tok} | {err} | {note} |".format(
                size=r["size"], impl=r["impl"],
                chunk_size=r["chunk_size"] if r["chunk_size"] is not None else "-",
                fwd=_fmt(r["forward_ms"], ".3f"), bwd=_fmt(r["backward_ms"], ".3f"),
                mem=_fmt(r["peak_vram_mb"], ".2f"),
                tok=_fmt(r["tokens_per_sec"], ",.0f"), err=_fmt(r["numerical_error_max_abs"], ".2e"),
                note=r.get("note", ""),
            )
        )
    lines.append("")
    with open(os.path.join(out_dir, "mamba2_scan_bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nsaved to {os.path.join(out_dir, 'mamba2_scan_bench.json')} and .md")


if __name__ == "__main__":
    rows = run()
    print_and_save(rows)
