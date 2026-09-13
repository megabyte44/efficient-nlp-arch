# Mamba scan benchmark (Phase 2 diagnostic)

`mamba_sequential_scan` vs `mamba_chunked_scan`, in isolation (not the full MambaMixer/GPT model). See ROADMAP.md for the experiment this is part of.

| size | impl | chunk_size | fwd_ms | bwd_ms | peak_vram_mb | tokens/sec | max_abs_err | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mamba_tiny (T=128, d_model=128) | sequential | - | 27.010 | 97.984 | 77.11 | 37,912 | 0.00e+00 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 8 | 17.908 | 37.897 | 370.98 | 57,181 | 4.10e-05 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 16 | 29.727 | 49.724 | 666.74 | 34,446 | 6.48e-05 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 32 | 64.602 | 93.140 | 1308.60 | 15,851 | 1.93e-04 |  |
| mamba_tiny (T=128, d_model=128) | chunked | 64 | 138.649 | 1542.928 | 2793.65 | 7,386 | 3.40e-04 |  |
| longer seq (T=512, d_model=128) | sequential | - | 101.107 | 640.477 | 258.64 | 40,511 | 0.00e+00 |  |
| longer seq (T=512, d_model=128) | chunked | 8 | 71.014 | 305.681 | 1400.64 | 57,679 | 1.20e-04 |  |
| longer seq (T=512, d_model=128) | chunked | 16 | 118.994 | 210.278 | 2501.69 | 34,422 | 9.73e-05 |  |
| longer seq (T=512, d_model=128) | chunked | 32 | 258.336 | 4542.800 | 4754.16 | 15,855 | 1.79e-04 |  |
| longer seq (T=512, d_model=128) | chunked | 64 | 554.448 | 20399.386 | 9460.45 | 7,388 | 5.80e-04 |  |
| wider model (T=128, d_model=256) | sequential | - | 26.410 | 105.448 | 140.97 | 38,774 | 0.00e+00 |  |
| wider model (T=128, d_model=256) | chunked | 8 | 34.476 | 810.438 | 724.58 | 29,702 | 9.16e-05 |  |
| wider model (T=128, d_model=256) | chunked | 16 | 1735.422 | 3567.859 | 1316.10 | 590 | 1.24e-04 |  |
| wider model (T=128, d_model=256) | chunked | 32 | 4226.230 | 7697.044 | 2599.82 | 242 | 1.87e-04 |  |
| wider model (T=128, d_model=256) | chunked | 64 | 278.850 | 6015.736 | 5569.92 | 3,672 | 4.39e-04 |  |
