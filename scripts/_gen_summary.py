"""One-off generator for experiments/summary.{json,md} from the per-run
benchmark.json files. Not part of the benchmark pipeline itself -- run by
hand after a batch of runs to refresh the consolidated summary.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = [
    "baseline_tiny", "local_tiny", "local_w8_tiny", "dilated_tiny",
    "s4_tiny", "mamba_tiny", "hybrid_s4_attn_tiny", "hybrid_mamba_attn_tiny",
]

results = []
for r in RUNS:
    with open(os.path.join(ROOT, "experiments", r, "benchmark.json"), encoding="utf-8") as f:
        results.append(json.load(f))
results.sort(key=lambda d: d["val_perplexity"])

summary_json_path = os.path.join(ROOT, "experiments", "summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

lines = []
lines.append("# Stage 2 results summary")
lines.append("")
lines.append(
    "Generated from `experiments/*/benchmark.json` (all trained checkpoints, "
    "2000 iters, tinyshakespeare char-level, RTX 2050). Sorted by validation "
    "perplexity (best first). See `ROADMAP.md` for the narrative and "
    "`configs/*.yaml` for exact hyperparameters."
)
lines.append("")
lines.append(
    "| Run | attn_type | layer_recipe | Params (non-emb) | FLOPs/token | "
    "Val loss | Val ppl | ms/forward | tokens/sec | Peak mem (MB) |"
)
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
for d in results:
    recipe = ", ".join(d["layer_recipe"]) if d.get("layer_recipe") else "-"
    lines.append(
        "| {run_name} | {attn_type} | {recipe} "
        "| {params:,} | {flops:,.0f} "
        "| {loss:.4f} | {ppl:.4f} "
        "| {ms:.2f} | {tok:,.0f} "
        "| {mem:.2f} |".format(
            run_name=d["run_name"],
            attn_type=d["attn_type"],
            recipe=recipe,
            params=d["params_non_embedding"],
            flops=d["flops_per_token_forward"],
            loss=d["val_loss"],
            ppl=d["val_perplexity"],
            ms=d["ms_per_forward"],
            tok=d["tokens_per_sec"],
            mem=d["peak_memory_mb"],
        )
    )
lines.append("")

summary_md_path = os.path.join(ROOT, "experiments", "summary.md")
with open(summary_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("wrote", summary_json_path)
print("wrote", summary_md_path)
