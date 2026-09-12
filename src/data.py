"""Character-level tokenizer + dataset over a single text file.

Char-level keeps the vocab tiny (~65 for tinyshakespeare) so the whole
baseline trains fast on CPU. Swap for a BPE tokenizer once real-scale runs
move to GPU.
"""

import os

import torch


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class CharDataset:
    def __init__(self, data_path, block_size, val_fraction=0.1):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = CharTokenizer(text)
        ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        n = len(ids)
        split = int(n * (1 - val_fraction))
        self.train_ids = ids[:split]
        self.val_ids = ids[split:]
        self.block_size = block_size

    def get_batch(self, split, batch_size, device):
        data = self.train_ids if split == "train" else self.val_ids
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


def load_dataset(cfg):
    data_path = os.path.join(os.path.dirname(__file__), "..", cfg["data_path"])
    return CharDataset(os.path.abspath(data_path), cfg["block_size"], cfg.get("val_fraction", 0.1))
