# Roadmap / Research Log

This is a living document. Update it as stages complete — it's the actual
record of what was tried, what worked, and what the next hypothesis is.

## Stage 1 — Baseline

Goal: a correct, minimal decoder-only Transformer + a benchmark harness that
can compare any future variant against it on equal footing.

Metrics captured by `src/benchmark.py`:
- quality: validation loss / perplexity
- size: total params, non-embedding params
- compute: analytical FLOPs/token (standard 6N + 12·L·H·Q·T approximation,
  same formula used in nanoGPT/PaLM/Chinchilla scaling-law work — approximate
  but consistent, so relative comparisons between variants are meaningful)
- latency: ms/token, tokens/sec (measured wall-clock, CPU now / GPU later)
- memory: peak RSS (CPU) or peak CUDA memory (GPU)

Status: scaffolded. Run a short training job locally to confirm the loop is
correct before trusting any later comparison.

## Stage 2 — Attack O(N²) attention

Question: which computations in full attention are actually unnecessary, and
can "what's relevant" be found for less than the cost of computing everything?

Candidates to implement behind the same `attn_type` config seam in
`src/model.py`:
- local/windowed attention
- sparse/strided attention
- linear attention (kernel-based)
- a selective-state (Mamba-style) block as an alternative to attention entirely

For each: benchmark vs baseline, note where quality degrades and why. The
gap each one fails to close is the actual research opening.

## Stage 3 — Attack uniform per-token compute

Question: does every token need the same amount of computation?

Candidates: early exit, adaptive depth, learned token routing / MoE.
Key cost to watch: routing/communication overhead can eat the savings —
measure it, don't assume it away.

## Stage 4 — Attack memory movement / precision

Only after an algorithmic variant is working and measured:
- quantization (INT8/INT4)
- KV-cache / state compression
- kernel-level IO optimization (FlashAttention-style)

## Log

- 2026-09-12: project scaffolded (baseline model, data pipeline, training
  loop, benchmark harness). Local dev is CPU-only; real-scale runs deferred
  to Colab/cloud GPU.
