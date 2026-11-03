"""Preflight gate: is the delegation task non-degenerate, BEFORE the expensive run? (v2 change C9)

v1 spent ~5 GPU-hours and reached Phase 4 before noticing that always-CALL was optimal (R(call) 1.000,
R(answer) 0.068, 93% of prompts call-beneficial), which made |b_i| a constant and UABR identical to BPR.
QUICK=True only proved the code ran. This gate checks that the *metric* can separate the conditions.

Three levels, cheapest first. Each runs everything above it:

  data        no model. Retrieval ceiling (share of dev prompts whose gold record is in the top-k) and
              the withheld share that bounds the 10% call-rate endpoint.
  base        a pretrained checkpoint that has never seen <ANSWER>/<CALL>. R(answer) is measured in the
              QA pretraining format, R(call) is the retrieval ceiling (CALL can copy at best what it
              retrieved), and the router is proxied by the model's own answer confidence: if its
              confidence cannot rank call-beneficial prompts, delegation training is unlikely to.
  delegation  a real prediction table from train_delegation.py (dev_predictions.parquet): the actual
              R(call), R(answer) and routing AUROC.

At pilot scale the dev mix follows the pilot ladder (equal weights over 8 tiers), not production.
--project resamples dev rows per tier to the production mix so the numbers predict the full run.

    python scripts/preflight_gate.py --data DATA                                  # data only
    python scripts/preflight_gate.py --data DATA --base runs/pilot --project      # pilot model
    python scripts/preflight_gate.py --data DATA --predictions runs/delegation/dev_predictions.parquet

Exit code 0 only if every check passes. Writes preflight_report.json next to --out.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.predict import load_prompts  # noqa: E402
from src.retrieval.evidence import EvidenceIndex  # noqa: E402

# docs/v2_design.md section 1 - (low, high); None = unbounded
TARGETS = {
    "r_call": (0.80, 0.90),
    "r_answer": (0.60, None),
    "call_beneficial_rate": (0.30, 0.60),
    "oracle_gap": (0.08, None),           # acc_oracle - acc_always_call
    "answer_wins_rate": (0.05, None),     # share with b_i < 0: the two-sided benefit v1 lacked entirely
    "auroc": (0.75, None),
    # Share with 0 < |b_i| < 1 on the graded per-fact reward. Below this floor |b| is effectively
    # binary, so UABR's s = exp(-m/tau)*|b| degenerates into boundary-only BPR with the b = 0
    # prompts masked out and the two conditions stop being different rules. Same class of
    # degeneracy that killed v1 (|b| constant), caught before the GPU spend this time.
    "graded_interior_rate": (0.15, None),
}
MAX_WITHHELD = 0.12                       # >= this share is call-necessary by construction; endpoint is 10%
PROD_TIERS, PROD_WEIGHTS, PROD_P_EXPOSED = [40, 120, 300], [0.35, 0.35, 0.30], 0.90


def in_band(v, band):
    lo, hi = band
    return v == v and (lo is None or v >= lo) and (hi is None or v <= hi)


def band_str(band):
    lo, hi = band
    return f">= {lo:.2f}" if hi is None else f"{lo:.2f} - {hi:.2f}"


# ---------------------------------------------------------------- level 1: data
def retrieval_table(prompts, evidence):
    rows = []
    for p in prompts:
        _, m = evidence.retrieve(p["question"], p["prompt_id"], p["evidence_doc_id"])
        rows.append({"prompt_id": p["prompt_id"], "gold_retrieved": m["gold_retrieved"],
                     "evidence_noisy": m["evidence_noisy"]})
    return pd.DataFrame(rows)


def data_checks(prompts, ret):
    withheld = float(np.mean([p["exposure_status"] == "withheld" for p in prompts]))
    clean = ret[ret.evidence_noisy == 0]
    ceiling = float(ret.gold_retrieved.mean())
    stats = {"n_prompts": len(prompts), "retrieval_ceiling": ceiling, "noisy_share": float(ret.evidence_noisy.mean()),
             "gold_retrieved_when_not_withheld": float(clean.gold_retrieved.mean()) if len(clean) else float("nan"),
             "withheld_share": withheld}
    checks = {"retrieval_ceiling_in_band": in_band(ceiling, TARGETS["r_call"]),
              "endpoint_reachable": withheld < MAX_WITHHELD}
    return stats, checks


# ---------------------------------------------------------------- level 2: base model
def base_table(model, tok, prompts, ret, batch_size=256, max_new=12):
    """One row per prompt in the predict.py schema, so metrics.summarize applies unchanged."""
    from src.data import formats
    from src.evaluation.generate import greedy, sequence_logprob
    # QA pretraining format (formats.qa_pretrain): the base model has never seen an action token
    prefixes = [tok.encode(formats.delegation_prompt(p["question"])).ids for p in prompts]
    gens = greedy(model, prefixes, max_new, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    # confidence = mean log-prob of the model's own greedy answer; empty answers count as least confident
    has = [i for i, g in enumerate(gens) if g]
    conf = np.full(len(gens), -np.inf)
    if has:
        conf[has] = sequence_logprob(model, [prefixes[i] for i in has], [gens[i] for i in has], batch_size)
    gold_ret = ret.set_index("prompt_id").gold_retrieved
    rows = []
    for p, g, c in zip(prompts, gens, conf):
        txt = tok.decode(g).strip()
        ra = formats.exact_match(txt, p["aliases"])
        rc = int(gold_ret[p["prompt_id"]])          # ceiling: assumes Phase 4 learns to copy (v1 did, 1.000)
        rows.append({"prompt_id": p["prompt_id"], "fact_id": p["fact_id"], "relation": p["relation"],
                     "template_family": p["template_family"], "exposure_count": p["exposure_count"],
                     "answer_text": txt, "gold": p["answer"], "reward_answer": ra, "reward_call": rc,
                     "benefit": rc - ra, "call_score": float(-c) if np.isfinite(c) else 1e9,
                     "predicted_action": "ANSWER", "both_fail": int(ra == 0 and rc == 0)})
    return M.add_graded_reward(pd.DataFrame(rows))


def project(df, tiers, weights, p_exposed, n=5000, seed=0):
    """Resample rows so the tier mix matches production (withheld = 1 - p_exposed)."""
    have = set(df.exposure_count.unique())
    missing = [t for t in [0, *tiers] if t not in have]
    assert not missing, f"tiers {missing} not in this build's ladder {sorted(have)}; pick production tiers from it"
    rng = np.random.default_rng(seed)
    share = {0: 1 - p_exposed, **{t: p_exposed * w for t, w in zip(tiers, weights)}}
    parts = []
    for t, s in share.items():
        g = df[df.exposure_count == t]
        parts.append(g.iloc[rng.integers(0, len(g), int(round(n * s)))])
    return pd.concat(parts, ignore_index=True)


def tier_table(df):
    g = df.groupby("exposure_count")
    return pd.DataFrame({"n": g.size(), "r_answer": g.reward_answer.mean(), "r_call": g.reward_call.mean(),
                         "call_beneficial": g.benefit.apply(lambda b: (b > 0).mean())}).round(3)


# ---------------------------------------------------------------- shared: task checks on a prediction table
def task_checks(df, n_boot=500):
    s = M.summarize(df)
    vals = {"r_call": s["acc_always_call"], "r_answer": s["acc_always_answer"],
            "call_beneficial_rate": s["call_beneficial_rate"],
            "oracle_gap": s["acc_oracle"] - s["acc_always_call"],
            "answer_wins_rate": float((df.benefit < 0).mean()), "auroc": s["auroc"],
            "graded_interior_rate": M.graded_interior_rate(df)}
    ci = M.bootstrap_ci(df, lambda d: M.auroc(d.call_score, d.benefit > 0), n_boot=n_boot)
    checks = {k: in_band(v, TARGETS[k]) for k, v in vals.items()}
    return vals, checks, {"auroc_95ci": ci, "budget_10": s["budget_10"], "acc_oracle": s["acc_oracle"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed data dir (facts/, tokenizer/)")
    ap.add_argument("--base", default=None, help="pretrained checkpoint: run the base-model level")
    ap.add_argument("--predictions", default=None, help="dev_predictions.parquet: run the delegation level")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None, help="score a fixed-seed subset of prompts")
    ap.add_argument("--project", action="store_true", help="reweight a pilot build to the production tier mix")
    ap.add_argument("--tiers", default=",".join(map(str, PROD_TIERS)))
    ap.add_argument("--weights", default=",".join(map(str, PROD_WEIGHTS)))
    ap.add_argument("--p_exposed", type=float, default=PROD_P_EXPOSED)
    ap.add_argument("--out", default="preflight_report.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    evidence = EvidenceIndex(data / "facts")
    prompts = load_prompts(data / "facts", a.split, limit=a.limit, seed=0)
    assert prompts, f"no prompts in split {a.split}"
    ret = retrieval_table(prompts, evidence)
    stats, checks = data_checks(prompts, ret)
    report = {"evidence_index": evidence.version, "split": a.split, "data": stats, "checks": dict(checks)}
    level = "data"

    df = None
    if a.predictions:
        level = "delegation"
        df = pd.read_parquet(a.predictions)
        assert set(df.prompt_id) <= set(ret.prompt_id) or a.limit, "prediction table is not from this data build"
    elif a.base:
        import torch
        from tokenizers import Tokenizer
        from src.model.load import load_model
        level = "base"
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
        model, _, info = load_model(a.base, device)
        report["base_step"] = info["step"]
        df = base_table(model, tok, prompts, ret)
        df.to_parquet(Path(a.out).with_name("preflight_base_table.parquet"), index=False)

    if df is not None:
        report["by_tier"] = tier_table(df).to_dict(orient="index")
        report["by_template_family"] = df.groupby("template_family").reward_answer.mean().round(3).to_dict()
        scored = df
        if a.project:
            tiers, weights = [int(x) for x in a.tiers.split(",")], [float(x) for x in a.weights.split(",")]
            scored = project(df, tiers, weights, a.p_exposed)
            report["projected_to"] = {"tiers": tiers, "weights": weights, "p_exposed": a.p_exposed}
        vals, tchecks, extra = task_checks(scored)
        report.update({"task": vals, "task_extra": extra})
        report["checks"].update(tchecks)
    report["level"] = level
    report["all_pass"] = all(report["checks"].values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w", encoding="utf-8"), indent=1, default=float)

    # ---------------------------------------------------------------- print
    d = report["data"]
    print(f"=== Preflight gate: level '{level}', {d['n_prompts']:,} {a.split} prompts, {evidence.version} ===")
    print(f"retrieval ceiling {d['retrieval_ceiling']:.3f} (target {band_str(TARGETS['r_call'])}) | "
          f"noisy share {d['noisy_share']:.3f} | gold found when not withheld {d['gold_retrieved_when_not_withheld']:.3f}")
    print(f"withheld share {d['withheld_share']:.3f} (must be < {MAX_WITHHELD:.2f} for the 10% call-rate endpoint)")
    if df is not None:
        print("\nper exposure tier:")
        print(tier_table(df).to_string())
        print(f"R(answer) by template family: {report['by_template_family']}")
        if a.project:
            print(f"\nprojected to production mix {report['projected_to']}")
        auroc_name = "routing AUROC" if level == "delegation" else "confidence AUROC (router proxy)"
        names = {"r_call": "R(call)" + (" ceiling" if level == "base" else ""), "r_answer": "R(answer)",
                 "call_beneficial_rate": "call-beneficial rate", "oracle_gap": "oracle - always_call",
                 "answer_wins_rate": "answer-wins rate (b<0)", "auroc": auroc_name}
        print(f"\n{'quantity':<34}{'value':>8}   target")
        for k, v in report["task"].items():
            print(f"{names[k]:<34}{v:>8.3f}   {band_str(TARGETS[k])}")
        lo, hi = report["task_extra"]["auroc_95ci"]
        print(f"AUROC 95% CI {lo:.3f} - {hi:.3f} | accuracy at 10% call budget "
              f"{report['task_extra']['budget_10']['accuracy']:.3f}")
    print()
    for k, v in report["checks"].items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"\nPREFLIGHT {'PASS' if report['all_pass'] else 'FAIL'} ({level} level) - {a.out} - "
          f"{time.time() - t0:.0f}s")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
