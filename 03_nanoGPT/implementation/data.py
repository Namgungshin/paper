"""
data.py - Shakespeare data preparation
nanoGPT: https://github.com/karpathy/nanoGPT
"""

import os
import urllib.request
import numpy as np
import pickle

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_FILE  = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN   = os.path.join(DATA_DIR, "val.bin")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")
VAL_RATIO = 0.1

_SAMPLE = """First Citizen:
Before we proceed any further, hear me speak.
ROMEO:
But, soft! what light through yonder window breaks?
HAMLET:
To be, or not to be, that is the question.
MACBETH:
Is this a dagger which I see before me.
""" * 80

def prepare():
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        try:
            urllib.request.urlretrieve(DATA_URL, RAW_FILE)
            with open(RAW_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            text = _SAMPLE
            with open(RAW_FILE, "w", encoding="utf-8") as f:
                f.write(text)

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n_val = int(len(data) * VAL_RATIO)
    data[:-n_val].tofile(TRAIN_BIN)
    data[-n_val:].tofile(VAL_BIN)
    with open(META_FILE, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)
    print(f"vocab_size = {vocab_size}")
    return vocab_size

if __name__ == "__main__":
    prepare()
