"""
generate.py - Text generation with trained model
nanoGPT: https://github.com/karpathy/nanoGPT

Usage:
    python generate.py
    python generate.py --prompt "ROMEO:" --max_tokens 300 --temperature 0.8
"""

import os
import argparse
import pickle
import torch
from model import GPT, GPTConfig

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "out")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    ckpt_path = os.path.join(OUT_DIR, "best_ckpt.pt")
    assert os.path.exists(ckpt_path), \
        f"No checkpoint: {ckpt_path}\nRun python train.py first."
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (iter={ckpt['iter']}, val_loss={ckpt['val_loss']:.4f})")
    return model, ckpt["config"]


def generate(prompt, max_new_tokens, temperature, top_k):
    model, config = load_model()
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    print(f"\nPrompt: {repr(prompt)}\n" + "-" * 60)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(out[0].tolist()))
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default="\n")
    parser.add_argument("--max_tokens",  type=int,   default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=200)
    args = parser.parse_args()
    generate(args.prompt, args.max_tokens, args.temperature, args.top_k)
