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
- 2026-09-13: all six Stage 1/2 variants (full, local x2, dilated, s4,
  mamba) trained for real (2000 iters, tinyshakespeare, RTX 2050 CUDA --
  not just benchmarked at random init) and compared on trained quality for
  the first time. Headline: s4 strictly dominates the full-attention
  baseline (fewer params, fewer FLOPs, lower val_loss); local(w=8) beats
  both local(w=32) and full attention on quality, not just compute (char-
  level next-token prediction leans on very recent context); dilated
  matches local(w=8)'s FLOPs/token exactly but loses badly on quality
  (2.1x worse ppl) -- contiguous local context beats sparse-but-wider
  context here; mamba has the best raw quality of the six but at ~71x the
  measured latency of any attention variant (unfused sequential scan, not
  the algorithm). Full comparison: `experiments/summary.{json,md}` and the
  published artifact (see chat).
- 2026-09-13: Stage 2 extension -- hybrid (interleaved attention + SSM)
  architectures added via a new `layer_recipe` seam (`resolve_layer_cfg` in
  `src/model.py`), motivated by the Jamba/Griffin/Zamba line of research
  the user pointed at. Two recipes trained: `[mamba, mamba, mamba, full]`
  and `[s4, s4, s4, full]` (attention as the last layer, position not yet
  swept). Result: **hybrid-mamba beats pure mamba on every axis at once**
  -- fewer params (1.13M vs 1.25M), fewer FLOPs/token (2.36M vs 2.52M),
  and better val_loss (1.5444 vs 1.5562, ppl 4.69 vs 4.74) -- now the best
  of all eight variants tested, and 25% faster than pure mamba (92.4ms vs
  121.9ms/forward) though still dominated by the three remaining
  sequential-scan layers. hybrid-s4 similarly edges out pure s4 (ppl 4.97
  vs 5.02) for a small compute cost, at S4D-like speed (5.3ms/forward) --
  the practical pick if latency matters more than the last bit of quality.
  Confirms the hybrid hypothesis on this toy setup. Not yet explored:
  attention-layer position within the recipe, and the interleaving ratio.
- 2026-09-13: Mamba scan diagnostic, Phases 1-2 of a scoped experiment
  (`mamba_sequential_scan`/`mamba_chunked_scan` in `src/model.py`,
  `scripts/bench_mamba_scan.py`) -- checking whether mamba's ~9x-latency
  cost (above) is the Python loop's fault or inherent to the recurrence.
  Phase 1: derived a chunked/parallel scan of the exact same S6 recurrence
  (a numerically-stable cumulative-decay-difference trick, adapted from the
  official Mamba-2/SSD reference's `segsum` but keeping our `A`'s full
  `(d_inner, d_state)` diagonal rather than Mamba-2's per-head-scalar
  restriction) and verified it matches the sequential reference within
  1e-4 across sequence lengths, chunk sizes, gradients, and after real
  training steps (`tests/test_model.py`). Phase 2: benchmarked the two
  scans in isolation (not the full model) at three sizes. Result: **a
  narrow, real win, not a general one**. At mamba_tiny's actual shape
  (d_inner=256) with the smallest chunk size tested (8), chunked beats
  sequential on both forward (17.9ms vs 27.0ms) and backward (37.9ms vs
  98.0ms) -- the loop really is costing something at production scale. But
  it doesn't generalize: larger chunk sizes get progressively worse
  (chunk=64: 138.6ms fwd / 1543ms bwd, both worse than sequential) because
  this adaptation's intra-chunk decay tensor scales as
  `O(chunk_size^2 * d_inner * d_state)` -- exactly the cost Mamba-2/SSD's
  per-head-scalar `A` restriction exists to avoid. Doubling model width
  (d_inner=512) breaks even chunk_size=8: forward stays roughly flat
  (34.5ms vs 26.4ms) but backward regresses badly (810ms vs 105ms); the
  chunk=16/32 points at that width were wildly non-monotonic (1735ms,
  4226ms fwd) and look like a measurement artifact (thermal/allocator
  noise from a prior heavy run) rather than a clean trend -- worth a
  rerun in a fresh process before trusting those two numbers specifically,
  though the qualitative scaling problem holds regardless. FLOPs are
  unchanged either way (same math, fewer loop iterations), so this is a
  latency finding, not a quality/compute-frontier improvement by itself.
  Sharpens the Phase 4 question from "make our chunked scan faster" to:
  how does Mamba-2 get chunked-scan parallelism without paying this
  `O(chunk^2 * width * state)` cost, and does that require its scalar-A
  restriction or something less restrictive? Full numbers:
  `experiments/mamba_scan_bench.{json,md}`.
- 2026-09-13: Mamba scan diagnostic, Phase 3 (model-level validation) --
  trained `mamba_tiny_chunked` and `hybrid_mamba_attn_tiny_chunked`
  (`scan_type: chunked`, `chunk_size: 8`, the one point Phase 2 flagged as
  a real win) for real, 2000 iters, on a Colab GPU rather than the local
  RTX 2050 (see below for why). Params and FLOPs/token match the
  unchunked originals exactly (architecture unchanged, as expected), and
  val_loss/perplexity land within noise (mamba: 1.5633 vs 1.5562; hybrid:
  1.5668 vs 1.5444) -- correctness holds over a full training run, not
  just short unit tests. For latency, ran the *unchunged* configs'
  `benchmark.py` on the same Colab GPU with no checkpoint (latency doesn't
  depend on trained weights, so random-init is fine for this) to get a
  same-hardware sequential baseline, since comparing against the original
  RTX 2050 numbers would have conflated the hardware change with the
  algorithm change. Result: chunked is **2.6x faster** for mamba_tiny
  (39.0ms vs 101.5ms/forward) and **3.4x faster** for
  hybrid_mamba_attn_tiny (27.8ms vs 95.2ms/forward) on the same GPU --
  better than Phase 2's isolated-scan estimate (~1.5x), plausibly because
  training's larger batch size (32 vs Phase 2's 8) amortizes the chunked
  computation's fixed overhead further.

  This directly contradicts an earlier local attempt: training
  `mamba_tiny_chunked` on the RTX 2050 at batch_size=32 was ~7x *slower*
  per iteration than sequential (interrupted before completion). Likely
  explanation: the RTX 2050's limited VRAM (4GB) hits the same
  `O(chunk_size^2 * d_inner * d_state)` intermediate-tensor cost Phase 2
  already flagged, but on a memory-constrained card that manifests as
  severe slowdown (allocator thrashing / possible thermal throttling
  under sustained load) rather than just "somewhat worse." Not confirmed
  with a controlled rerun -- noted as the likely cause, not a proven one.

  Net conclusion: the chunked scan is a **real, substantial win on
  adequately-resourced hardware, but hardware-dependent enough that it
  should stay an opt-in (`scan_type: chunked`) rather than becoming the
  default** -- the same setting that helps 2.6-3.4x on a Colab GPU
  regressed 7x on a memory-constrained laptop GPU. Gates Phase 4 open:
  the next question is whether Mamba-2/SSD's scalar-per-head `A` (or some
  less restrictive variant) removes this hardware sensitivity entirely by
  avoiding the `O(chunk^2 * width * state)` blowup in the first place,
  rather than just being fast when there happens to be enough VRAM.
