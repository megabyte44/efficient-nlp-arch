# Stage 2 results summary

Generated from `experiments/*/benchmark.json` (all trained checkpoints, 2000 iters, tinyshakespeare char-level, RTX 2050). Sorted by validation perplexity (best first). See `ROADMAP.md` for the narrative and `configs/*.yaml` for exact hyperparameters.

| Run | attn_type | layer_recipe | Params (non-emb) | FLOPs/token | Val loss | Val ppl | ms/forward | tokens/sec | Peak mem (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid_mamba_attn_tiny | hybrid | mamba, mamba, mamba, full | 1,133,440 | 2,356,992 | 1.5444 | 4.6854 | 92.35 | 11,088 | 28.76 |
| mamba_tiny | mamba | - | 1,245,568 | 2,523,904 | 1.5562 | 4.7409 | 121.86 | 8,403 | 29.66 |
| hybrid_s4_attn_tiny | hybrid | s4, s4, s4, full | 674,944 | 1,440,000 | 1.6042 | 4.9740 | 5.29 | 193,715 | 20.69 |
| s4_tiny | s4 | - | 634,240 | 1,301,248 | 1.6140 | 5.0229 | 3.86 | 265,145 | 20.40 |
| local_w8_tiny | local | - | 797,056 | 1,610,496 | 1.6518 | 5.2163 | 1.72 | 596,804 | 20.88 |
| local_tiny | local | - | 797,056 | 1,659,648 | 1.6878 | 5.4074 | 1.64 | 624,886 | 20.88 |
| baseline_tiny | full | - | 797,056 | 1,856,256 | 1.7133 | 5.5475 | 1.72 | 596,490 | 20.81 |
| dilated_tiny | dilated | - | 797,056 | 1,610,496 | 2.4094 | 11.1268 | 1.62 | 630,245 | 20.88 |

