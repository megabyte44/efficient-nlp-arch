# Colab / cloud GPU benchmarking

Local dev is CPU-only and intentionally tiny (char-level, ~4 layers) — just
enough to verify correctness fast. Real quality/FLOPs/memory comparisons
between architecture variants (Stage 2+) need GPU scale to mean anything.

Plan for this folder: a Colab notebook that
1. clones this repo
2. installs `requirements.txt`
3. runs `src/train.py` with a larger config (bigger `n_embd`/`n_layer`/`block_size`,
   BPE tokenizer instead of char-level)
4. runs `src/benchmark.py` for each variant and collects results into
   `experiments/` for the Pareto-frontier comparison

Not built yet — add the notebook once Stage 2 has a second variant to
actually compare against the baseline.
