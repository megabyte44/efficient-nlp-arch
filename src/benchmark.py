"""Stage-1 benchmark harness: quality / size / compute / latency / memory.

This is the comparison tool the whole project depends on — every later
variant (Stage 2+) gets run through the same script so numbers are
comparable. Keep it variant-agnostic: it only assumes a GPT-compatible
model with a `.forward(idx, targets)` signature and a `get_num_params()`.
"""

import argparse
import json
import math
import os
import time

import psutil
import torch
from tabulate import tabulate

from data import load_dataset
from model import GPT
from utils import get_device, load_config


def attention_span(cfg):
    """Effective number of keys/queries the attention term scales with.

    Full attention: each token attends over the whole sequence -> T.
    Local/windowed attention: capped at window_size regardless of T.
    This is what makes the O(T) attention term in the FLOPs formula below
    actually shrink for windowed attention instead of just replicating
    full attention's estimate under a different name.
    """
    if cfg["attn_type"] == "local":
        return min(cfg["window_size"], cfg["block_size"])
    return cfg["block_size"]


def analytical_flops_per_token_forward(cfg, n_params):
    """Approximate forward-pass FLOPs/token.

    Derived from the standard 6N + 12*L*H*Q*T total (forward+backward)
    approximation used in nanoGPT/PaLM/Chinchilla scaling-law work, divided
    by 3 (forward:backward FLOPs ratio is ~1:2). This is an analytical
    estimate, not a profiler trace — treat absolute values as approximate,
    but relative comparisons between variants (same formula, different
    architecture) are meaningful. T is replaced by the effective attention
    span (see `attention_span`) so variants that restrict attention (e.g.
    local/windowed) show the corresponding drop in this term.
    """
    L, H = cfg["n_layer"], cfg["n_head"]
    Q, T = cfg["n_embd"] // cfg["n_head"], attention_span(cfg)
    total_fwd_bwd = 6 * n_params + 12 * L * H * Q * T
    return total_fwd_bwd / 3


@torch.no_grad()
def measure_latency(model, dataset, cfg, device, batch_size=8, n_warmup=5, n_iters=20):
    model.eval()
    x, _ = dataset.get_batch("val", batch_size, device)

    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_call = batch_size * cfg["block_size"]
    total_tokens = tokens_per_call * n_iters
    return {
        "ms_per_forward": (elapsed / n_iters) * 1000,
        "ms_per_token": (elapsed / total_tokens) * 1000,
        "tokens_per_sec": total_tokens / elapsed,
    }


@torch.no_grad()
def measure_memory(model, dataset, cfg, device, batch_size=8):
    x, _ = dataset.get_batch("val", batch_size, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        model(x)
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        return {"peak_memory_mb": peak_bytes / 1e6, "source": "torch.cuda.max_memory_allocated"}
    else:
        model(x)
        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        return {"peak_memory_mb": rss_bytes / 1e6, "source": "psutil RSS (CPU proxy, not true peak)"}


@torch.no_grad()
def measure_quality(model, dataset, cfg, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = dataset.get_batch("val", cfg["batch_size"], device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    val_loss = losses.mean().item()
    return {"val_loss": val_loss, "val_perplexity": math.exp(val_loss)}


def run_benchmark(cfg, checkpoint_path=None):
    device = get_device()
    dataset = load_dataset(cfg)
    cfg["vocab_size"] = dataset.tokenizer.vocab_size

    model = GPT(cfg).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded checkpoint: {checkpoint_path}")
    else:
        print("no checkpoint found — benchmarking an untrained (random-init) model")

    n_params_total = model.get_num_params(non_embedding=False)
    n_params_non_emb = model.get_num_params(non_embedding=True)

    results = {
        "run_name": cfg.get("run_name", "unnamed"),
        "attn_type": cfg["attn_type"],
        "params_total": n_params_total,
        "params_non_embedding": n_params_non_emb,
        "flops_per_token_forward": analytical_flops_per_token_forward(cfg, n_params_non_emb),
    }
    results.update(measure_quality(model, dataset, cfg, device))
    results.update(measure_latency(model, dataset, cfg, device))
    results.update(measure_memory(model, dataset, cfg, device))
    return results


def print_results(results):
    rows = [(k, f"{v:,.4f}" if isinstance(v, float) else v) for k, v in results.items()]
    print(tabulate(rows, headers=["metric", "value"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_ckpt = os.path.join(
            os.path.dirname(__file__), "..", cfg["out_dir"], "ckpt.pt"
        )
        checkpoint_path = os.path.abspath(default_ckpt)

    results = run_benchmark(cfg, checkpoint_path)
    print_results(results)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cfg["out_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {os.path.join(out_dir, 'benchmark.json')}")


if __name__ == "__main__":
    main()
