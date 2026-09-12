import argparse
import json
import math
import os
import time

import torch

from data import load_dataset
from model import GPT
from utils import get_device, load_config, set_seed


@torch.no_grad()
def estimate_loss(model, dataset, cfg, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = dataset.get_batch(split, cfg["batch_size"], device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device: {device}")

    dataset = load_dataset(cfg)
    cfg["vocab_size"] = dataset.tokenizer.vocab_size
    print(f"vocab_size: {cfg['vocab_size']}")

    model = GPT(cfg).to(device)
    n_params = model.get_num_params()
    print(f"params (non-embedding): {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    out_dir = os.path.join(os.path.dirname(__file__), "..", cfg["out_dir"])
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    history = []
    t0 = time.time()
    best_val_loss = float("inf")

    for it in range(cfg["max_iters"] + 1):
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            losses = estimate_loss(model, dataset, cfg, device)
            val_ppl = math.exp(losses["val"])
            elapsed = time.time() - t0
            print(
                f"iter {it}: train_loss {losses['train']:.4f} "
                f"val_loss {losses['val']:.4f} val_ppl {val_ppl:.2f} "
                f"({elapsed:.1f}s)"
            )
            history.append({"iter": it, **losses, "val_ppl": val_ppl, "elapsed_s": elapsed})

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {"model_state_dict": model.state_dict(), "cfg": cfg},
                    os.path.join(out_dir, "ckpt.pt"),
                )

            if it == cfg["max_iters"]:
                break

        x, y = dataset.get_batch("train", cfg["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"done. best val_loss={best_val_loss:.4f}, checkpoint + history saved to {out_dir}")


if __name__ == "__main__":
    main()
