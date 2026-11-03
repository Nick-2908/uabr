"""Write synthetic train.bin/val.bin (uint16) so the training code can be tested before real data exists.

Tokens follow a random first-order Markov chain, so loss *should* fall well below log(vocab).
"""
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="data/processed/synthetic")
ap.add_argument("--vocab_size", type=int, default=12000)
ap.add_argument("--n_train", type=int, default=5_000_000)
ap.add_argument("--n_val", type=int, default=200_000)
ap.add_argument("--branching", type=int, default=4, help="possible next tokens per token")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
succ = rng.integers(0, a.vocab_size, size=(a.vocab_size, a.branching))
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for name, n in (("train.bin", a.n_train), ("val.bin", a.n_val)):
    choice = rng.integers(0, a.branching, size=n)
    toks = np.empty(n, dtype=np.uint16)
    cur = int(rng.integers(0, a.vocab_size))
    for i in range(n):
        toks[i] = cur
        cur = succ[cur, choice[i]]
    toks.tofile(out / name)
    print(f"wrote {out/name}: {n:,} tokens")
print(f"ideal loss = ln({a.branching}) = {np.log(a.branching):.3f}; random = ln({a.vocab_size}) = {np.log(a.vocab_size):.3f}")
