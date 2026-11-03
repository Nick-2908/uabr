"""Phase 4: delegation training + validation gate .

    python scripts/train_delegation.py --base base_v1.pt --data data_v1 --knowledge facts_knowledge.parquet \
        --out runs/delegation [--quick]

Writes: best.pt / final.pt, delegation_v1.pt (frozen hand-off), dev_predictions.parquet, dev_summary.json,
gate_report.json, audit_sample.csv (100 dev rows to check by hand).
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
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.predict import load_prompts, predict  # noqa: E402
from src.model.load import load_model  # noqa: E402
from src.retrieval.evidence import EvidenceIndex  # noqa: E402
from src.train.sft import save_weights, train  # noqa: E402


def build_examples(tok, prompts, knows, evidence):
    """ANSWER examples for facts the base model knows, CALL examples for the rest.

    CALL is trained only where the gold record was actually retrieved. Supervising the gold answer on
    a noisy retrieval would teach the model to produce an answer its evidence does not support - the
    opposite of the behaviour under test - so those prompts are skipped and the unsupported-answer
    rate stays a held-out measurement rather than something trained away.
    """
    ans, call, skipped = [], [], 0
    for p in prompts:
        prompt = formats.delegation_prompt(p["question"])
        if knows[p["fact_id"]]:
            ans.append(formats.build_example(tok, prompt, formats.answer_target(p["answer"])))
        else:
            # same frozen retrieval the scorer uses, so training and evaluation see identical evidence
            ev, m = evidence.retrieve(p["question"], p["prompt_id"], p["evidence_doc_id"])
            if not m["gold_retrieved"]:
                skipped += 1
                continue
            call.append(formats.build_example(tok, prompt, formats.call_target(ev, p["answer"])))
    return ans, call, skipped


def epoch_lists(ans, call, epochs, answer_frac, seed, max_examples=None):
    out = []
    for ep in range(epochs):
        rng = random.Random(seed + ep)
        n_call = min(len(call), round(len(ans) * (1 - answer_frac) / answer_frac)) if ans else len(call)
        n_ans = len(ans)
        if n_call < round(n_ans * (1 - answer_frac) / answer_frac):  # fewer CALL than needed: downsample ANSWER
            n_ans = min(len(ans), round(n_call * answer_frac / (1 - answer_frac)))
        ex = rng.sample(ans, n_ans) + rng.sample(call, n_call)
        rng.shuffle(ex)
        if max_examples:
            ex = ex[:max_examples]
        out.append(ex)
    return out


def gate(df, knows_dev, call_ci):
    d = df.copy()
    d["label"] = np.where(d.fact_id.map(knows_dev) == 1, "ANSWER", "CALL")
    conf = pd.crosstab(d.label, d.predicted_action).reindex(index=["ANSWER", "CALL"], columns=["ANSWER", "CALL"], fill_value=0)
    recall = {lab: float(conf.loc[lab, lab] / max(conf.loc[lab].sum(), 1)) for lab in ["ANSWER", "CALL"]}
    pred_share = d.predicted_action.value_counts(normalize=True).to_dict()
    direct_known = float(d[d.label == "ANSWER"].reward_answer.mean()) if (d.label == "ANSWER").any() else float("nan")
    direct_unknown = float(d[d.label == "CALL"].reward_answer.mean())
    s = M.summarize(d)
    checks = {
        "both_actions_learned": min(pred_share.get("ANSWER", 0), pred_share.get("CALL", 0)) >= 0.05
                                and min(recall.values()) >= 0.5,
        "useful_ranking": call_ci[0] > 0.55,
        "direct_path_works": direct_known - direct_unknown >= 0.2,
        "call_path_works": s["acc_always_call"] >= 0.9,
    }
    rep = {
        "confusion_label_x_predicted": conf.to_dict(orient="index"), "recall_by_label": recall,
        "predicted_share": pred_share, "auroc": s["auroc"], "auroc_95ci": call_ci,
        "direct_accuracy_known_facts": direct_known, "direct_accuracy_unknown_facts": direct_unknown,
        "call_path_accuracy": s["acc_always_call"], "summary": s,
        "by_template_family": {fam: {"n": len(g), "auroc": M.auroc(g.call_score, g.benefit > 0), "default_acc": float(
            np.where(g.predicted_action == "CALL", g.reward_call, g.reward_answer).mean())} for fam, g in d.groupby("template_family")},
        "checks": checks, "all_pass": all(checks.values()),
    }
    return rep, conf, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/delegation.yaml")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--knowledge", required=True)
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
    if a.quick:
        ft["epochs"] = cfg["quick"]["epochs"]; dc["dev_eval_prompts"] = cfg["quick"]["dev_eval_prompts"]
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    tok_hash = json.load(open(data / "manifest.json", encoding="utf-8"))["tokenizer_sha256"]
    evidence = EvidenceIndex(data / "facts")
    model, base_config, info = load_model(a.base, a.device)
    print(f"base model step {info['step']} | evidence {evidence.version} ({len(evidence):,} records)")

    kn = pd.read_parquet(a.knowledge)
    knows = dict(zip(kn.fact_id, kn.knows))
    train_prompts = load_prompts(data / "facts", "delegation_train")
    missing = {p["fact_id"] for p in train_prompts} - set(knows)
    assert not missing, f"{len(missing)} delegation_train facts have no knowledge label"
    ans, call, skipped = build_examples(tok, train_prompts, knows, evidence)
    epochs = epoch_lists(ans, call, ft["epochs"], dc["answer_frac"], ft["seed"],
                         cfg["quick"]["max_train_examples"] if a.quick else None)
    print(f"delegation_train: {len(ans):,} ANSWER + {len(call):,} CALL examples available "
          f"({skipped:,} CALL prompts skipped for ungrounded retrieval); "
          f"{len(epochs[0]):,} per epoch ({dc['answer_frac']:.0%} ANSWER) x {len(epochs)} epochs")

    dev_small = load_prompts(data / "facts", "dev", limit=dc["dev_eval_prompts"], seed=0)

    def eval_fn(m, step):
        df = predict(m, tok, dev_small, evidence, batch_size=256)
        s = M.summarize(df)
        lab = np.where(df.fact_id.map(knows) == 1, "ANSWER", "CALL")
        label_auroc = M.auroc(df.call_score, lab == "CALL")
        # select on the counterfactual routing AUROC; fall back to AUROC against the knowledge labels if undefined
        select = s["auroc"] if s["auroc"] == s["auroc"] else label_auroc
        res = {"dev_auroc": s["auroc"], "dev_label_auroc": label_auroc, "select_score": select,
               "dev_default_acc": s["default_decision"]["accuracy"],
               "dev_default_call_rate": s["default_call_rate"], "dev_action_acc_vs_label": float((df.predicted_action == lab).mean()),
               "dev_call_path_acc": s["acc_always_call"], "dev_direct_acc": s["acc_always_answer"]}
        print("  dev: " + " | ".join(f"{k} {v:.3f}" for k, v in res.items() if k != "select_score"))
        return res

    if not a.resume:
        print("before training (base model):")
        eval_fn(model, 0)
    summ = train(model, epochs, ft, base_config, out, eval_fn=eval_fn, select_metric="select_score",
                 resume=a.resume, max_minutes=a.max_minutes)
    if summ["status"] != "completed":
        print(f"stopped at step {summ['steps_done']}/{summ['total_steps']}; rerun the same command with --resume")
        return
    print(f"trained {summ['total_steps']} steps, {summ['tokens_seen']:,} tokens; best step {summ['best_step']}")

    chosen = out / "best.pt" if (out / "best.pt").exists() else out / "final.pt"
    if chosen.name == "final.pt":
        print("WARNING: no dev evaluation produced a valid selection score; using the final checkpoint")
    best, _, binfo = load_model(chosen, a.device)
    dev = load_prompts(data / "facts", "dev")
    df = predict(best, tok, dev, evidence, run_id="delegation", checkpoint=f"best@{binfo['step']}", tokenizer_hash=tok_hash)
    df.to_parquet(out / "dev_predictions.parquet", index=False)
    ci = M.bootstrap_ci(df, lambda d: M.auroc(d.call_score, d.benefit > 0), n_boot=500)
    rep, conf, d = gate(df, knows, ci)
    rep["train"] = summ; rep["quick"] = a.quick; rep["seed"] = ft["seed"]
    rep["labels"] = "knowledge-based (base_v1 cloze probe); declared deviation from exposure-status labels"
    json.dump(rep, open(out / "gate_report.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(rep["summary"], open(out / "dev_summary.json", "w", encoding="utf-8"), indent=1, default=float)
    d.sample(min(100, len(d)), random_state=0)[["question", "gold", "label", "predicted_action", "call_score",
        "answer_text", "reward_answer", "call_answer_text", "reward_call", "exposure_count"]].to_csv(
        out / "audit_sample.csv", index=False, encoding="utf-8-sig")
    save_weights(best, out / "delegation_v1.pt", torch.load(chosen, map_location="cpu", weights_only=False)["config"],
                 binfo["step"], binfo["tokens_seen"], {"name": "delegation_v1", "evidence_index": evidence.version,
                                                       "tokenizer_sha256": tok_hash, "seed": ft["seed"]})

    s = rep["summary"]
    print(f"\n=== Delegation gate (full dev, {s['n']:,} prompts) ===")
    print("confusion (rows = knowledge label, columns = predicted first token):")
    print(conf.to_string())
    print(f"routing AUROC {s['auroc']:.3f}  (95% CI {ci[0]:.3f}-{ci[1]:.3f}) | average precision {s['average_precision']:.3f} "
          f"| AUROC vs knowledge label {M.auroc(d.call_score, d.label == 'CALL'):.3f}")
    print(f"direct answer accuracy: known facts {rep['direct_accuracy_known_facts']:.3f} vs unknown {rep['direct_accuracy_unknown_facts']:.3f}")
    print(f"CALL-path accuracy {s['acc_always_call']:.3f} | call-beneficial share {s['call_beneficial_rate']:.3f}")
    print(f"end-to-end accuracy: always answer {s['acc_always_answer']:.3f} | default decision {s['default_decision']['accuracy']:.3f} "
          f"(call rate {s['default_call_rate']:.3f}) | always call {s['acc_always_call']:.3f} | oracle {s['acc_oracle']:.3f}")
    for b in (5, 10, 20):
        o = s[f"budget_{b}"]
        print(f"  call budget {b:>2}%: accuracy {o['accuracy']:.3f} | harmful answers {o['harmful_answer_rate']:.3f}")
    for k, v in rep["checks"].items():
        print(("PASS " if v else "FAIL ") + k)
    print("DELEGATION GATE PASS" if rep["all_pass"] else "DELEGATION GATE NOT PASSED - see gate_report.json")
    print(f"saved {out / 'delegation_v1.pt'} | total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
