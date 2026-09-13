# Mamba-2/SSD-style scan benchmark (Phase 4 diagnostic)

`mamba2_sequential_scan` vs `mamba2_chunked_scan` (scalar-per-head A), in isolation. Same sizes as `experiments/mamba_scan_bench.md` (Phase 2, full per-(channel,state) diagonal A) for direct comparison. See ROADMAP.md for the experiment this is part of.

| size | impl | chunk_size | fwd_ms | bwd_ms | peak_vram_mb | tokens/sec | max_abs_err | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mamba_tiny (T=128, d_model=128) | sequential | - | 40.252 | 162.931 | 42.04 | 25,440 | 0.00e+00 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 8 | 12.567 | 29.197 | 62.13 | 81,483 | 3.05e-05 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 16 | 6.537 | 17.732 | 65.53 | 156,638 | 9.54e-05 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 32 | 3.502 | 9.107 | 72.35 | 292,445 | 1.14e-04 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 64 | 1.941 | 5.918 | 86.00 | 527,685 | 1.14e-04 |  |
| longer seq (T=512, d_model=128) | sequential | - | 189.310 | 643.899 | 117.64 | 21,637 | 0.00e+00 |  |
| longer seq (T=512, d_model=128) | chunked | 8 | 47.669 | 135.601 | 189.24 | 85,925 | 3.39e-05 |  |
| longer seq (T=512, d_model=128) | chunked | 16 | 25.160 | 66.228 | 191.72 | 162,798 | 8.20e-05 |  |
| longer seq (T=512, d_model=128) | chunked | 32 | 13.266 | 34.577 | 200.11 | 308,765 | 1.11e-04 |  |
| longer seq (T=512, d_model=128) | chunked | 64 | 7.652 | 24.252 | 216.91 | 535,264 | 2.92e-04 |  |
| wider model (T=128, d_model=256) | sequential | - | 43.649 | 156.836 | 70.62 | 23,460 | 0.00e+00 |  |
| wider model (T=128, d_model=256) | chunked | 8 | 13.495 | 31.942 | 106.30 | 75,882 | 2.22e-05 |  |
| wider model (T=128, d_model=256) | chunked | 16 | 6.550 | 15.916 | 112.85 | 156,344 | 7.15e-05 |  |
| wider model (T=128, d_model=256) | chunked | 32 | 3.858 | 12.619 | 125.96 | 265,451 | 1.14e-04 |  |
| wider model (T=128, d_model=256) | chunked | 64 | 3.659 | 9.491 | 152.19 | 279,889 | 2.06e-04 |  |
