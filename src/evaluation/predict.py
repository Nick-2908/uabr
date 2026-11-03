"""Score both actions once per prompt .

For every prompt this records the routing log-probabilities, the forced direct answer, the forced
evidence-assisted answer, their rewards and the oracle analysis. Every metric is computed from this one
table, so different scripts cannot disagree.
"""
import time

import numpy as np
import pandas as pd

from src.data import formats
from src.evaluation import metrics
from src.evaluation.generate import greedy, next_token_logprobs


def predict(model, tok, prompts, evidence, run_id="", checkpoint="", tokenizer_hash="", batch_size=256, max_new=12):
    """prompts: list of dicts from prompts.jsonl (question, aliases, evidence_doc_id, ...)."""
    t0 = time.time()
    enc = lambda s: tok.encode(s).ids  # noqa: E731
    base = [enc(formats.delegation_prompt(p["question"])) for p in prompts]
    route = next_token_logprobs(model, base, [formats.ANSWER, formats.CALL], batch_size)
    t_route = time.time()

    ans_prompts = [b + [formats.ANSWER] for b in base]
    ans = greedy(model, ans_prompts, max_new, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    t_ans = time.time()

    # v2: frozen top-k retrieval, not an id lookup, so CALL can fail (docs/v2_design.md section 4)
    retrieved = [evidence.retrieve(p["question"], p["prompt_id"], p["evidence_doc_id"]) for p in prompts]
    ev_text = [t for t, _ in retrieved]
    ev_meta = [m for _, m in retrieved]
    call_prompts = [enc(formats.delegation_prompt(p["question"]) + f"<CALL><EVIDENCE> {e} </EVIDENCE>")
                    for p, e in zip(prompts, ev_text)]
    calls = greedy(model, call_prompts, max_new, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    t_call = time.time()

    rows = []
    for p, (la, lc), a, c, em, ev in zip(prompts, route, ans, calls, ev_meta, ev_text):
        a_txt, c_txt = tok.decode(a).strip(), tok.decode(c).strip()
        ra, rc = formats.exact_match(a_txt, p["aliases"]), formats.exact_match(c_txt, p["aliases"])
        # unsupported: the CALL branch asserted something the retrieved evidence does not contain
        unsupported = int(bool(c_txt) and formats.normalize_answer(c_txt) not in
                          formats.normalize_answer(ev))
        pred = "CALL" if lc > la else "ANSWER"
        b = rc - ra
        rows.append({
            "run_id": run_id, "prompt_id": p["prompt_id"], "fact_id": p["fact_id"], "split": p["split"],
            "relation": p["relation"], "template_family": p["template_family"],
            "exposure_count": p["exposure_count"], "question": p["question"], "gold": p["answer"],
            "logp_answer": la, "logp_call": lc, "call_score": lc - la, "predicted_action": pred,
            "answer_text": a_txt, "normalized_answer": formats.normalize_answer(a_txt), "reward_answer": ra,
            "evidence_doc_id": p["evidence_doc_id"], "call_answer_text": c_txt,
            "normalized_call_answer": formats.normalize_answer(c_txt), "reward_call": rc,
            "benefit": b, "preferred_action": "CALL" if b > 0 else "ANSWER",
            "harmful_answer": int(pred == "ANSWER" and ra == 0 and rc == 1),
            "wasteful_call": int(pred == "CALL" and ra == 1),
            "both_fail": int(ra == 0 and rc == 0), "unsupported_call_answer": unsupported,
            "model_checkpoint": checkpoint, "tokenizer_hash": tokenizer_hash, "evidence_index_hash": evidence.version,
            **em,  # retrieved_doc_ids, evidence_noisy, gold_retrieved, gold_rank
        })
    df = metrics.add_graded_reward(pd.DataFrame(rows))  # graded |b| for Phase 6 replay selection
    n = max(len(prompts), 1)
    df.attrs["latency_ms_per_prompt"] = {"route": 1000 * (t_route - t0) / n, "answer": 1000 * (t_ans - t_route) / n,
                                         "call": 1000 * (t_call - t_ans) / n}
    return df


def load_prompts(facts_dir, split, limit=None, seed=0):
    import json
    from pathlib import Path
    rows = [json.loads(l) for l in open(Path(facts_dir) / "prompts.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == split]
    if limit and limit < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), limit, replace=False))]
    return rows
