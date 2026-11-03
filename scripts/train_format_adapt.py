"""Phase 3.5: format adaptation (v2 change C8).

Between pretraining (Phase 3) and delegation (Phase 4), briefly fine-tune the base model on the
instruction / QA answer format so the ANSWER path actually works when a fact is asked as a question.
v1 had no such stage: the base model scored cloze EM 0.747 but QA-format EM 0.165, so "answer from
memory" could never win and the whole delegation task degenerated into always-CALL.

Data = Dolly/Alpaca SFT (data/sft/sft_train.jsonl, already tokenized) + QA-format facts built from
replay_construction entities only. delegation_train and dev facts are deliberately left to pretraining
alone, so their knowledge (and therefore the routing measurement in Phase 4) is not inflated here;
replay_construction is the pool the Phase 5-6 replay study draws from, so this is also where the
"policy that can be forgotten" is first established.

The output checkpoint (format_adapted.pt) is what Phase 3.6 (label_knowledge.py) and Phase 4
(train_delegation.py) take as their --base.

    python scripts/train_format_adapt.py --base base.pt --data DATA --out runs/format_adapt
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.data.build_facts import MAIN_FAMILY, QUESTIONS  # noqa: E402
from src.evaluation.generate import greedy  # noqa: E402
from src.evaluation.predict import load_prompts  # noqa: E402
from src.model.load import load_model  # noqa: E402
from src.train.sft import save_weights, train  # noqa: E402


def sft_examples(data):
    """(ids, labels) from the packed Dolly/Alpaca rows; loss only on the response (after prompt_len)."""
    out = []
    for line in open(data / "sft" / "sft_train.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ids, pl = r["ids"], r["prompt_len"]
        labels = [-100] * len(ids)
        for t in range(len(ids) - 1):
            if t + 1 >= pl:
                labels[t] = ids[t + 1]
        out.append((ids, labels))
    return out


def fact_examples(tok, ledger, q_templates):
    """QA-format examples for replay_construction facts, same format as the SFT rows (no action token)."""
    fams = MAIN_FAMILY[:q_templates]
    out = []
    for r in ledger.itertuples():
        for t in fams:
            q = QUESTIONS[r.relation][t].format(e=r.pseudonym)
            out.append(formats.build_example(tok, formats.delegation_prompt(q), formats.sft_target(r.value_text)))
    return out


def epoch_lists(sft, facts, epochs, fact_frac, seed, max_examples=None):
    """Each epoch: all fact examples + SFT examples resampled to the target fact fraction (seeded)."""
    out = []
    for ep in range(epochs):
        rng = random.Random(seed + ep)
        n_fact = len(facts)
        want_sft = round(n_fact * (1 - fact_frac) / fact_frac) if fact_frac > 0 else len(sft)
        n_sft = min(len(sft), want_sft)
        if n_sft < want_sft and fact_frac < 1:  # fewer SFT than needed: downsample facts to keep the ratio
            n_fact = min(len(facts), round(n_sft * fact_frac / (1 - fact_frac)))
        ex = rng.sample(facts, n_fact) + rng.sample(sft, n_sft)
        rng.shuffle(ex)
        out.append(ex[:max_examples] if max_examples else ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/format_adapt.yaml")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from OUT/resume.pt (same data, seed, epochs)")
    ap.add_argument("--max_minutes", type=float, default=None, help="stop cleanly, saving full state for --resume")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    ft, dc = dict(cfg["finetune"]), dict(cfg["data"])
    if a.seed is not None:
        ft["seed"] = a.seed
    max_examples = None
    if a.quick:
        ft["epochs"] = cfg["quick"]["epochs"]
        dc["dev_eval_prompts"] = cfg["quick"]["dev_eval_prompts"]
        max_examples = cfg["quick"]["max_train_examples"]
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    tok_hash = json.load(open(data / "manifest.json", encoding="utf-8"))["tokenizer_sha256"]
    model, base_config, info = load_model(a.base, a.device)

    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    rc = led[led.split == "replay_construction"].reset_index(drop=True)
    sft = sft_examples(data)
    facts = fact_examples(tok, rc, dc["q_templates"])
    assert facts, "no replay_construction facts to adapt on"
    epochs = epoch_lists(sft, facts, ft["epochs"], dc["fact_frac"], ft["seed"], max_examples)
    print(f"base model step {info['step']} | {len(sft):,} SFT + {len(facts):,} QA-fact examples "
          f"({len(rc):,} replay_construction facts x {dc['q_templates']} templates); "
          f"{len(epochs[0]):,} per epoch ({dc['fact_frac']:.0%} facts) x {len(epochs)} epochs")

    dev = load_prompts(data / "facts", "dev", limit=dc["dev_eval_prompts"], seed=0)

    def eval_fn(m, step):
        prefixes = [tok.encode(formats.delegation_prompt(p["question"])).ids for p in dev]
        gens = greedy(m, prefixes, max_new=12, stop_ids=[formats.EOS, formats.BOS], batch_size=256)
        em = np.array([formats.exact_match(tok.decode(g).strip().split("\n")[0].strip(), p["aliases"])
                       for g, p in zip(gens, dev)], dtype=float)
        fam = np.array([p["template_family"] for p in dev])
        res = {"dev_qa_em": float(em.mean()),
               "dev_qa_em_qmain": float(em[fam == "q_main"].mean()) if (fam == "q_main").any() else float("nan"),
               "dev_qa_em_qalt": float(em[fam == "q_alt"].mean()) if (fam == "q_alt").any() else float("nan")}
        print("  dev: " + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))
        return res

    if not a.resume:
        print("before adaptation (base model):")
        eval_fn(model, 0)
    summ = train(model, epochs, ft, base_config, out, eval_fn=eval_fn, select_metric=dc["select_metric"],
                 resume=a.resume, max_minutes=a.max_minutes)
    if summ["status"] != "completed":
        print(f"stopped at step {summ['steps_done']}/{summ['total_steps']}; rerun the same command with --resume")
        return
    print(f"trained {summ['total_steps']} steps, {summ['tokens_seen']:,} tokens; best step {summ['best_step']}")

    chosen = out / "best.pt" if (out / "best.pt").exists() else out / "final.pt"
    if chosen.name == "final.pt":
        print("WARNING: no dev evaluation produced a valid selection score; using the final checkpoint")
    best, bconfig, binfo = load_model(chosen, a.device)
    eval_fn(best, binfo["step"])
    save_weights(best, out / "format_adapted.pt", bconfig, binfo["step"], binfo["tokens_seen"],
                 {"name": "format_adapted", "tokenizer_sha256": tok_hash, "seed": ft["seed"],
                  "base_step": info["step"]})
    rep = {"train": summ, "quick": a.quick, "seed": ft["seed"], "base_step": info["step"],
           "n_sft": len(sft), "n_fact_examples": len(facts), "n_replay_construction_facts": len(rc),
           "fact_frac": dc["fact_frac"], "q_templates": dc["q_templates"]}
    json.dump(rep, open(out / "format_adapt_report.json", "w", encoding="utf-8"), indent=1, default=float)
    print(f"saved {out / 'format_adapted.pt'} | total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
