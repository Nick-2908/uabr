"""the seven data audits, as one pass/fail report.

    python scripts/audit_ledger.py --data data/processed
Writes data/processed/audit_report.json and audit_alias_sample.csv (100 rows to check by hand).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_facts import QA_FRACTION  # noqa: E402
from src.data.formats import normalize_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--min_class", type=int, default=50,
                    help="min facts per split and class (v2: p_exposed 0.90 leaves ~72 withheld dev facts)")
    a = ap.parse_args()
    data = Path(a.data)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    prompts = pd.read_json(data / "facts" / "prompts.jsonl", lines=True)
    ev = pd.read_json(data / "facts" / "evidence.jsonl", lines=True)
    rep, ok = {}, {}

    # 1. duplicate facts across splits (and entities split cleanly)
    fact_splits = led.groupby("fact_id").split.nunique()
    ent_splits = led.groupby("entity_id").split.nunique()
    rep["1_duplicate_facts"] = {"facts_in_multiple_splits": int((fact_splits > 1).sum()),
                                "entities_in_multiple_splits": int((ent_splits > 1).sum()),
                                "duplicate_fact_ids": int(led.fact_id.duplicated().sum())}
    ok["1_duplicate_facts"] = all(v == 0 for v in rep["1_duplicate_facts"].values())

    # 2. normalized prompt overlap across splits
    prompts["norm"] = prompts.question.map(normalize_answer)
    cross = prompts.groupby("norm").split.nunique()
    rep["2_prompt_overlap"] = {"normalized_prompts_crossing_splits": int((cross > 1).sum()),
                               "prompt_split_matches_fact_split": bool(
                                   (prompts.merge(led[["fact_id", "split"]], on="fact_id", suffixes=("", "_f"))
                                    .eval("split == split_f")).all())}
    ok["2_prompt_overlap"] = rep["2_prompt_overlap"]["normalized_prompts_crossing_splits"] == 0 and \
        rep["2_prompt_overlap"]["prompt_split_matches_fact_split"]

    # 3. class balance (report counts; fail only if a split lacks either class)
    bal = led.groupby(["split", "exposure_status"]).size().unstack(fill_value=0)
    tiers = led[led.exposure_count > 0].groupby(["split", "exposure_count"]).size().unstack(fill_value=0)
    rep["3_class_balance"] = {"facts": bal.to_dict(orient="index"), "exposed_tiers": tiers.to_dict(orient="index"),
                              "relations": led.relation.value_counts().to_dict()}
    ok["3_class_balance"] = bool((bal.min(axis=1) >= a.min_class).all())

    # 4. evidence completeness: exactly one evidence record per fact
    ev_per_fact = ev.groupby("fact_id").size()
    missing = set(led.fact_id) - set(ev.fact_id)
    rep["4_evidence"] = {"facts_missing_evidence": len(missing), "facts_with_multiple_evidence": int((ev_per_fact > 1).sum()),
                         "evidence_contains_answer": float(
                             ev.drop(columns="value_text", errors="ignore").merge(led[["fact_id", "value_text"]], on="fact_id")
                             .apply(lambda r: r.value_text in r.text, axis=1).mean())}
    ok["4_evidence"] = rep["4_evidence"]["facts_missing_evidence"] == 0 and \
        rep["4_evidence"]["facts_with_multiple_evidence"] == 0 and rep["4_evidence"]["evidence_contains_answer"] == 1.0

    # 5. alias audit: automatic checks + 100-row sample for a manual look
    aliases = led.accepted_aliases.map(json.loads)
    short = aliases.map(lambda al: any(len(normalize_answer(x)) <= 1 for x in al))
    shared = Counter(normalize_answer(x) for al in led.drop_duplicates("value_id").accepted_aliases.map(json.loads) for x in al)
    ambiguous = sorted(k for k, c in shared.items() if c > 1)
    led.sample(100, random_state=0)[["fact_id", "relation", "value_text", "accepted_aliases"]].to_csv(
        data / "audit_alias_sample.csv", index=False, encoding="utf-8-sig")  # BOM so Excel shows accents
    rep["5_aliases"] = {"facts_with_1char_alias": int(short.sum()), "aliases_shared_by_different_values": ambiguous[:20],
                        "manual_sample": "audit_alias_sample.csv - check 100 rows by hand and record the result"}
    ok["5_aliases"] = int(short.sum()) == 0

    # 6. corpus leakage: pseudonyms found in the Wikipedia sample (from the tokenizer audit)
    tr_path = data / "tokenizer" / "tokenizer_report.json"
    hits = json.load(open(tr_path, encoding="utf-8")).get("pseudonym_hits_in_wiki_sample", None) if tr_path.exists() else None
    rep["6_leakage"] = {"pseudonym_hits_in_wiki_sample": hits,
                        "shard_ids_recorded_for_exposed": float((led[led.exposure_count > 0].pretrain_shard_ids != "").mean())
                        if "pretrain_shard_ids" in led else None,
                        "withheld_with_shard_ids": int((led[led.exposure_count == 0].pretrain_shard_ids != "").sum())}
    ok["6_leakage"] = hits == [] and rep["6_leakage"]["withheld_with_shard_ids"] == 0

    # 7. exposure composition (v2 C2): statement + QA counts are consistent and the QA share is as designed
    if {"exposure_count_statement", "exposure_count_qa"} <= set(led.columns):
        sums_ok = bool((led.exposure_count_statement + led.exposure_count_qa == led.exposure_count).all())
        withheld = led[led.exposure_count == 0]
        withheld_zero = bool(((withheld.exposure_count_statement == 0) & (withheld.exposure_count_qa == 0)).all())
        exposed = led[led.exposure_count > 0]
        qa_frac = float(exposed.exposure_count_qa.sum() / max(exposed.exposure_count.sum(), 1))
        rep["7_exposure_composition"] = {"statement_plus_qa_equals_total": sums_ok,
                                         "withheld_have_no_exposure": withheld_zero,
                                         "qa_fraction": round(qa_frac, 3), "qa_fraction_target": QA_FRACTION}
        ok["7_exposure_composition"] = sums_ok and withheld_zero and abs(qa_frac - QA_FRACTION) < 0.05
    else:
        rep["7_exposure_composition"] = "ledger has no exposure_count_statement/qa columns (pre-C2 build)"
        ok["7_exposure_composition"] = False

    rep["pass"] = ok
    rep["all_pass"] = all(ok.values())
    json.dump(rep, open(data / "audit_report.json", "w", encoding="utf-8"), indent=1, default=str)
    for k, v in ok.items():
        print(("PASS " if v else "FAIL ") + k)
    print("ALL PASS" if rep["all_pass"] else "SOME AUDITS FAILED - see audit_report.json")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
