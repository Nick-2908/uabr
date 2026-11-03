"""Generate text from fixed prompts so checkpoints can be compared by eye (handbook: do not judge by loss alone).

    python scripts/sample_text.py --ckpt runs/pretrain_40m --data /kaggle/input/.../data_v1
"""
import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.model.load import load_model  # noqa: E402

PROMPTS = ["The history of the city", "In mathematics, a prime number", "The river flows through",
           "During the Second World War,", "The album was released in", "Photosynthesis is the process",
           "The university was founded", "In 1998, the team", "The species is found in", "The capital of France is"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.data) / "tokenizer" / "tokenizer.json"))
    model, _, info = load_model(a.ckpt, a.device)
    print(f"step {info['step']}")
    torch.manual_seed(0)
    for p in PROMPTS:
        ids = torch.tensor([[formats.BOS] + tok.encode(p).ids], device=a.device)
        g = model.generate(ids, a.max_new, eos_id=formats.EOS, temperature=0.0)[0, ids.shape[1]:].tolist()
        s = model.generate(ids, a.max_new, eos_id=formats.EOS, temperature=0.8)[0, ids.shape[1]:].tolist()
        print(f"\n> {p}\n  greedy: {tok.decode(g)!r}\n  t=0.8 : {tok.decode(s)!r}")


if __name__ == "__main__":
    main()
