"""Shared fixtures: a tiny offline Phase 2 dataset built once per test session."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORDS = ("the of and in to was is for on as with by he at from his an were are which this be its first new "
         "city river school war year state national century music film team university known also after "
         "government population church world album series season league island village district").split()


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("p2")
    rng = random.Random(0)
    wiki = d / "wiki.jsonl"
    with open(wiki, "w", encoding="utf-8") as f:
        for i in range(600):
            text = "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(60, 140))).capitalize() + "."
                             for _ in range(4)) + "\nReferences\nshould be cut"
            f.write(json.dumps({"text": text}) + "\n")
    sft = d / "sft_local.jsonl"
    with open(sft, "w", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps({"id": f"x{i}", "source": "toy", "instruction": f"Explain topic {i} briefly.",
                                "context": "", "response": " ".join(rng.choice(WORDS) for _ in range(30))}) + "\n")
    data = d / "processed"
    run("-m", "src.data.vocab", "--offline", "--out", str(d / "vocab.json"))
    run("-m", "src.data.build_facts", "--vocab", str(d / "vocab.json"), "--out", str(data / "facts"), "--n_entities", "150")  # v2: ~146 exposures per fact
    run("-m", "src.data.sft_data", "--local", str(sft), "--out", str(d / "sft_raw.jsonl"))
    run("-m", "src.data.tokenizer", "--facts", str(data / "facts"), "--sft", str(d / "sft_raw.jsonl"),
        "--out", str(data / "tokenizer"), "--vocab_size", "2000", "--sample_chars", "300000",
        "--wiki_source", "local", "--wiki_local", str(wiki))
    run("-m", "src.data.pack", "--data", str(data), "--sft_raw", str(d / "sft_raw.jsonl"), "--target_tokens", "1500000",
        "--val_tokens", "20000", "--sft_val", "50", "--wiki_source", "local", "--wiki_local", str(wiki))
    return data
