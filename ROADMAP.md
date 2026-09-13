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
- 2026-09-13: Mamba-scan investigation, Phase 4 (`experiment/mamba2-scalar-a-ssd`
  branch; Phases 1-3 are on `experiment/mamba-chunked-scan` -- see that
  branch's ROADMAP.md for the full log). Phases 1-3 built a chunked scan
  for MambaMixer's existing Mamba-1/S6 recurrence (full per-(channel,
  state) diagonal `A`) and found it real but hardware-sensitive: a win on
  a Colab GPU, a 7x regression on the local RTX 2050, traced to an
  intra-chunk decay tensor scaling as `O(chunk_size^2 * d_inner *
  d_state)`. Phase 4 asks the question that sharpened to: does official
  Mamba-2/SSD's restriction -- `A` a single scalar per head, shared across
  every channel-within-head and every state dim, instead of independent
  per (channel, state) -- actually remove that blowup, or was it a red
  herring? Added `mamba2_sequential_scan`/`mamba2_chunked_scan` (`src/
  model.py`) implementing that restricted recurrence and its own chunked
  form (intra-chunk decay tensor now `(B, chunk_size, chunk_size,
  n_heads)` -- no channel/state dims at all), verified equivalent to its
  own sequential reference within 1e-4 across sequence lengths, chunk
  sizes, n_heads, and gradients (`tests/test_model.py`). Deliberately a
  separate architecture, not a modification of MambaMixer -- the point is
  to isolate what the restriction itself buys.

  Benchmarked at the exact sizes that broke Phase 2 (`scripts/
  bench_mamba2_scan.py`, `experiments/mamba2_scan_bench.{json,md}`, same
  RTX 2050 Phase 2 used). Result: **the blowup is gone, cleanly**. Larger
  chunk sizes now *monotonically improve* speed instead of catastrophically
  regressing -- chunk_size=64 is the best point at every size tested, e.g.
  mamba_tiny's shape: 1.94ms fwd / 5.92ms bwd vs sequential's 40.25ms /
  162.93ms (a 21x/28x speedup), and peak VRAM grows gently (42MB to 86MB)
  instead of exploding (Phase 2 saw 77MB to 2.8GB at the same chunk size).
  The wider-model point that hit 1735ms/4226ms forward in Phase 2 is now
  6.55ms/3.86ms at the same chunk sizes -- confirms the scalar-per-head
  restriction is exactly the mechanism that makes Mamba-2's chunked scan
  hardware-robust, not an unrelated implementation difference.

  Open question this doesn't answer: what does this restriction cost in
  quality? Scalar-per-head `A` is a real reduction in selectivity (one
  decay rate per head instead of per channel-and-state) -- per the
  project's actual objective (quality per unit compute, not speed for its
  own sake), this needs a real trained comparison (Mamba-1/S6-style
  MambaMixer vs a Mamba-2-style equivalent, same param budget) before
  concluding the restriction is worth adopting, not just that it's fast.
  Not yet built -- these functions aren't wired into a trainable mixer.
