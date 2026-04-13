"""
train.py - Training loop
nanoGPT: https://github.com/karpathy/nanoGPT

Usage:
    python data.py     # prepare data first
    python train.py    # start training
"""

import os
import math
import pickle
import time
import numpy as np
import torch
from model import GPT, GPTConfig

# hyperparameters
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# model (GPU recommended: N_EMBD=384, N_LAYER=6, N_HEAD=6, BLOCK_SIZE=256)
BLOCK_SIZE = 256
N_EMBD     = 384
N_LAYER    = 6
N_HEAD     = 6
DROPOUT    = 0.2

# training
BATCH_SIZE    = 64
MAX_ITERS     = 5000
EVAL_INTERVAL = 500
EVAL_ITERS    = 200
LOG_INTERVAL  = 100

# optimizer
LR           = 1e-3
WEIGHT_DECAY = 1e-1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0

# LR schedule (Cosine with Warmup)
WARMUP_ITERS   = 100
LR_DECAY_ITERS = MAX_ITERS
MIN_LR         = LR / 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


def load_data():
    train_data = np.fromfile(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16)
    val_data   = np.fromfile(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16)
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return train_data, val_data, meta


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (LR - MIN_LR)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = [model(*get_batch(data, BLOCK_SIZE, BATCH_SIZE))[1].item()
                  for _ in range(EVAL_ITERS)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train():
    print(f"device: {DEVICE}")
    train_data, val_data, meta = load_data()
    vocab_size = meta["vocab_size"]
    print(f"vocab_size: {vocab_size}")

    config = GPTConfig(
        vocab_size=vocab_size, n_positions=BLOCK_SIZE,
        n_embd=N_EMBD, n_layer=N_LAYER, n_head=N_HEAD, dropout=DROPOUT,
    )
    model = GPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(LR, WEIGHT_DECAY, (BETA1, BETA2))

    best_val_loss = float('inf')
    t0 = time.time()
    x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)

    for it in range(1, MAX_ITERS + 1):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {time.time()-t0:.1f}s")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({"model": model.state_dict(), "config": config,
                            "iter": it, "val_loss": best_val_loss},
                           os.path.join(OUT_DIR, "best_ckpt.pt"))
                print(f"  -> checkpoint saved (val_loss={best_val_loss:.4f})")
            t0 = time.time()

        _, loss = model(x, y)
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            print(f"  step {it:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    print(f"\nDone. best val_loss = {best_val_loss:.4f}")
    print(f"checkpoint: {OUT_DIR}/best_ckpt.pt")


if __name__ == "__main__":
    train()
