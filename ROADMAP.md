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
- 2026-09-12: Stage 2, local (windowed) causal attention added
  (`attn_type: local`). window=32 shows no quality degradation vs baseline;
  window=8 config exists to find where it starts to.
- 2026-09-12: Stage 2, dilated segment attention added (`attn_type:
  dilated`), extracting LongNet's core primitive (fixed segment + dilation
  sparse pattern). Same effective span as `local_w8_tiny` (8) gives an
  identical analytical FLOPs estimate, spread across a wider segment
  instead of a tight trailing window -- the quality comparison between the
  two at equal compute is the open question.
- 2026-09-12: Stage 2, diagonal state-space mixer added (`attn_type: s4`),
  extracting S4D's closed-form convolutional view (FFT-based, O(T log T),
  no qkv/attention term at all). Non-selective baseline for the SSM family.
- 2026-09-12: Stage 2, selective state-space mixer added (`attn_type:
  mamba`), extending S4D with input-dependent Delta/B/C (Mamba's S6) plus
  the paper's gating/conv block shape. Implemented as a plain sequential
  scan (no fused kernel) since selectivity breaks S4D's FFT shortcut --
  measured ms_per_forward on the tiny CPU config is ~9x baseline's, the
  expected cost of an unfused O(T) scan; a real hardware-aware scan is a
  Stage-4-style follow-up if this variant's quality justifies it.
- 2026-09-12: Stage 4 (early), chunked/online-softmax exact attention added
  as a `chunked: true` toggle on `full`/`local` (the FlashAttention tiling
  trick, pure PyTorch). Verified numerically exact (matches the plain path
  within 1e-5). Measured a full T sweep (window=8, chunk_size=16, CPU) to
  find where it actually pays off, rather than assuming: **slower** at
  small T (2.8x at T=128, 2.0x at T=512) where per-block Python-loop
  overhead dominates, crossing over between T=2048 (~parity) and T=4096
  (0.83x), then decisively faster as T grows -- 0.50x at T=8192, 0.25x
  (4x faster) at T=16384, since the plain mask path is still a full T^2
  matmul while the chunked path's cost stays tied to window_size once
  block-skipping kicks in. Exactly the roadmap's own caveat playing out:
  "only after an algorithmic variant is measured" -- the trick is real,
  but only past a measured crossover length; a fused kernel (Triton/CUDA)
  would push that crossover much earlier.
