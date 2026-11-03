"""Knowledge-based delegation labels, measured in the DEPLOYED delegation format (v2 change C7).

v1 labelled a fact ANSWER from a pretraining-style *cloze* probe ("X was born in the city of ___").
That scored 0.733 on dev facts whose real question-format accuracy was only 0.222, so delegation
training was handed ANSWER targets the model could not actually satisfy once the question was phrased
as a question. v2 labels from the same format the router is deployed in:

    <BOS><USER> What is the occupation of {e}?<ASSISTANT>   ->   greedy answer, exact-matched

For each non-test fact we ask its q_main question templates (the family seen in training and at
deployment; q_alt is held out for evaluation only), greedy-decode, and take knows = 1 when the model
answers correctly on at least `know_threshold` of those templates. Delegation targets are then ANSWER
if knows else CALL, so ANSWER means "the model reliably answers this, in the format it will be asked".

    python scripts/label_knowledge.py --base /kaggle/input/.../base.pt --data /kaggle/input/.../data \
        --out /kaggle/working/facts_knowledge.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.data.build_facts import MAIN_FAMILY, QUESTIONS  # noqa: E402
from src.evaluation.generate import greedy  # noqa: E402
from src.model.load import load_model  # noqa: E402


def label(model, tok, ledger, n_templates=len(MAIN_FAMILY), know_threshold=0.5, batch_size=256, max_new=10):
    """One row per fact. knows = 1 if greedy EM >= know_threshold across q_main question templates."""
    fams = MAIN_FAMILY[:n_templates]
    prefixes, owner = [], []
    for ridx, r in enumerate(ledger.itertuples()):
        for t in fams:
            q = QUESTIONS[r.relation][t].format(e=r.pseudonym)
            prefixes.append(tok.encode(formats.delegation_prompt(q)).ids)
            owner.append(ridx)
    gens = greedy(model, prefixes, max_new=max_new, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    aliases = [json.loads(a) for a in ledger.accepted_aliases]

    n = len(ledger)
    em_sum, cnt = np.zeros(n), np.zeros(n)
    first_pred = [""] * n
    for g, ri in zip(gens, owner):
        txt = tok.decode(g).strip().split("\n")[0].strip()
        if cnt[ri] == 0:
            first_pred[ri] = txt
        em_sum[ri] += formats.exact_match(txt, aliases[ri])
        cnt[ri] += 1
    em_rate = em_sum / np.maximum(cnt, 1)

    out = ledger[["fact_id", "entity_id", "split", "relation", "exposure_status", "exposure_count"]].copy()
    out["qa_em_rate"] = em_rate.round(4)
    out["qa_pred"] = first_pred
    out["n_templates"] = cnt.astype(int)
    out["knows"] = (em_rate >= know_threshold).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="delegation_train,replay_construction,dev")
    ap.add_argument("--n_templates", type=int, default=len(MAIN_FAMILY),
                    help=f"q_main templates asked per fact (1-{len(MAIN_FAMILY)}); knows = EM rate over them")
    ap.add_argument("--know_threshold", type=float, default=0.5, help="EM rate at or above which knows = 1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    assert 1 <= a.n_templates <= len(MAIN_FAMILY), f"--n_templates must be 1-{len(MAIN_FAMILY)}"
    t0 = time.time()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    model, _, info = load_model(a.base, a.device)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    splits = a.splits.split(",")
    assert "test" not in splits, "test facts are never labelled here"
    led = led[led.split.isin(splits)].reset_index(drop=True)
    out = label(model, tok, led, n_templates=a.n_templates, know_threshold=a.know_threshold)
    out["base_step"] = info["step"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)

    tab = out.pivot_table(index="split", columns="exposure_count", values="knows", aggfunc="mean").round(3)
    print(f"labelled {len(out):,} facts in {time.time() - t0:.0f}s with base step {info['step']} "
          f"({a.n_templates} q_main templates/fact, knows = EM rate >= {a.know_threshold})")
    print("\nshare the base model knows (1 = ANSWER label), by split x exposure tier:")
    print(tab.to_string())
    print("\nmean QA EM rate by split x exposure tier (what the label threshold is applied to):")
    print(out.pivot_table(index="split", columns="exposure_count", values="qa_em_rate", aggfunc="mean").round(3).to_string())
    print("\nlabel balance:", out.groupby("split").knows.agg(["mean", "size"]).round(3).to_dict(orient="index"))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
