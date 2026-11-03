"""Phase 3 fact probe: did pretraining actually store the controlled facts?

Cloze test per exposure tier (0 = withheld, 5, 20, 80 statements): greedy completion exact match, plus the
teacher-forced log-probability of the gold answer (a smoother signal early in training).
Uses delegation_train + dev facts only; the test split is never touched here.

    python scripts/probe_facts.py --ckpt runs/pretrain_40m --data /kaggle/input/.../data_v1 --per_tier 300
Expected after full pretraining: EM and logprob rise with tier (80 > 20 > 5 > withheld).
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.model.load import load_model  # noqa: E402

# statement templates that end with the answer (subset of build_facts.STATEMENTS)
CLOZE = {
    "born_in": "{e} was born in the city of",
    "citizen_of": "{e} is a citizen of",
    "occupation": "The occupation of {e} is",
    "language": "{e} speaks",
    "birth_year": "{e} was born in the year",
}


@torch.no_grad()
def greedy_batch(model, prompts, max_new=10, stop_ids=()):
    """prompts: list of equal-length id lists."""
    x = torch.tensor(prompts, device=next(model.parameters()).device)
    outs = [[] for _ in prompts]
    done = [False] * len(prompts)
    for _ in range(max_new):
        logits, _ = model(x[:, -model.cfg.max_seq_len:])
        nxt = logits[:, -1].argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if not done[i]:
                if t in stop_ids:
                    done[i] = True
                else:
                    outs[i].append(t)
        if all(done):
            break
        x = torch.cat([x, nxt[:, None]], 1)
    return outs


@torch.no_grad()
def gold_logprob(model, prefix_ids, gold_ids):
    dev = next(model.parameters()).device
    ids = torch.tensor([prefix_ids + gold_ids], device=dev)
    logits, _ = model(ids)
    lp = torch.log_softmax(logits[0].float(), -1)
    pos = range(len(prefix_ids) - 1, len(prefix_ids) + len(gold_ids) - 1)
    return sum(lp[p, ids[0, p + 1]].item() for p in pos) / len(gold_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir, checkpoint dir or exported .pt")
    ap.add_argument("--data", required=True, help="data_v1 folder")
    ap.add_argument("--per_tier", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    model, cfg, info = load_model(a.ckpt, a.device)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    led = led[led.split.isin(["delegation_train", "dev"])]
    sample = pd.concat([g.sample(min(a.per_tier, len(g)), random_state=0) for _, g in led.groupby("exposure_count")])

    stop = {formats.EOS, formats.BOS}
    rows, by_len = [], defaultdict(list)
    for r in sample.itertuples():
        prefix = [formats.BOS] + tok.encode(CLOZE[r.relation].format(e=r.pseudonym)).ids
        by_len[len(prefix)].append((r, prefix))
    for L, items in by_len.items():
        for i in range(0, len(items), 64):
            chunk = items[i : i + 64]
            gens = greedy_batch(model, [p for _, p in chunk], stop_ids=stop)
            for (r, prefix), g in zip(chunk, gens):
                pred = re.split(r"\.(?:\s|$)", tok.decode(g).strip(), maxsplit=1)[0]
                aliases = json.loads(r.accepted_aliases)
                gold = tok.encode(" " + r.value_text).ids
                rows.append({"fact_id": r.fact_id, "relation": r.relation, "tier": r.exposure_count, "pred": pred,
                             "gold": r.value_text, "em": formats.exact_match(pred, aliases),
                             "gold_logprob": gold_logprob(model, prefix, gold)})
    df = pd.DataFrame(rows)
    summ = df.groupby("tier").agg(n=("em", "size"), em=("em", "mean"), gold_logprob=("gold_logprob", "mean")).round(4)
    by_rel = df.pivot_table(index="relation", columns="tier", values="em", aggfunc="mean").round(3)
    print(f"checkpoint step {info['step']} ({(info['tokens_seen'] or 0) / 1e6:,.0f}M tokens) | {time.time() - t0:.0f}s")
    print("\nExact match by exposure tier (0 = withheld):")
    print(summ.to_string())
    print("\nExact match by relation x tier:")
    print(by_rel.to_string())
    em = summ["em"].to_dict()
    ok = em.get(80, 0) > em.get(0, 0) + 0.05 and summ["gold_logprob"].is_monotonic_increasing
    print("\nGATE:", "PASS - recall rises with exposure" if ok else "NOT YET - no clear exposure gradient")
    print("\nExamples (tier 80):")
    print(df[df.tier == 80].head(8)[["gold", "pred", "em"]].to_string(index=False))
    out = Path(a.out) if a.out else Path(a.ckpt if Path(a.ckpt).is_dir() else Path(a.ckpt).parent) / f"probe_step{info['step']}.json"
    json.dump({"step": info["step"], "tokens_seen": info["tokens_seen"], "by_tier": summ.reset_index().to_dict(orient="records"),
               "by_relation_tier": by_rel.to_dict(), "gate_pass": bool(ok)}, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
