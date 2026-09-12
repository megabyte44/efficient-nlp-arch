# Efficient NLP Architecture Research

Open-ended research project: find an architectural mechanism for language
modeling that sits **up and to the left** on the quality-vs-compute Pareto
frontier compared to a standard Transformer — same or better quality at
meaningfully lower inference FLOPs and/or memory.

## The three goals (attacked independently)

| Goal | Strategy | Directions to investigate |
|---|---|---|
| 1. Preserve/improve quality | Don't lose important token relationships | attention, selective state (Mamba-style), routing, retrieval |
| 2. Reduce compute | Don't perform unnecessary operations | sparse/local/linear attention, adaptive depth, dynamic routing |
| 3. Reduce memory/latency | Minimize data movement + precision | quantization, KV-cache reduction, low-rank state, kernel IO efficiency |

The point is not to pick a mechanism upfront. It's to build a baseline and a
benchmark harness, then **measure where each existing approach actually
fails**, and design around that specific gap.

## Research method (the actual project)

```
baseline Transformer
        |
benchmark: quality / FLOPs / memory / latency / tokens-per-sec
        |
attack O(N^2) attention        (Stage 2)
        |
attack per-token uniform compute (Stage 3)
        |
attack memory movement / precision (Stage 4)
        |
ablations + comparison vs baseline -> Pareto frontier plot
```

Success is demonstrated experimentally: e.g. "same perplexity as baseline,
60% lower inference FLOPs" or "same compute, 8% better long-context
retrieval." Not assumed.

## Status

**Stage 1 (current): Baseline.** Minimal GPT-style decoder-only Transformer,
trained from scratch, with a benchmark harness measuring: validation
perplexity, parameter count, analytical FLOPs/token, wall-clock
latency/tokens-per-sec, peak memory.

Local dev is CPU-only (small char-level model on tiny Shakespeare, for fast
iteration and correctness checking). Real benchmark runs at meaningful scale
happen on Colab/cloud GPU — see `notebooks/`.

See [ROADMAP.md](ROADMAP.md) for the full staged plan and running research log.

## Layout

```
src/
  model.py       config-driven decoder-only Transformer (attention is a pluggable seam)
  data.py        char-level tokenizer + dataset
  train.py       training loop, checkpointing
  benchmark.py   quality / FLOPs / memory / latency measurement suite
  utils.py       config loading, seeding, device selection
configs/         YAML configs (model + training + benchmark)
scripts/         one-off utilities (data download, etc.)
experiments/     results per run (JSON + markdown tables), the actual research log
notebooks/       Colab notebooks for GPU-scale benchmarking
```

## Setup

```
pip install -r requirements.txt
python scripts/download_data.py
python src/train.py --config configs/baseline_tiny.yaml
python src/benchmark.py --config configs/baseline_tiny.yaml --checkpoint experiments/baseline_tiny/ckpt.pt
```
