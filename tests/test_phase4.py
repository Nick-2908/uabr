"""Phase 4: evaluation primitives + the delegation pipeline end to end on a tiny model """
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.generate import greedy  # noqa: E402
from src.model.model import DecoderLM, ModelConfig  # noqa: E402


def test_auroc_ap_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    s, y = np.round(rng.normal(size=400), 1), (rng.random(400) < 0.3).astype(int)
    assert abs(M.auroc(s, y) - roc_auc_score(y, s)) < 1e-9
    s2 = rng.normal(size=400)
    assert abs(M.average_precision(s2, y) - average_precision_score(y, s2)) < 1e-9


def test_budget_routing_and_aurc():
    df = pd.DataFrame({"call_score": [3, 2, 1, 0], "reward_answer": [0, 0, 1, 1], "reward_call": [1, 1, 1, 0],
                       "benefit": [1, 1, 0, -1], "predicted_action": ["CALL", "CALL", "ANSWER", "ANSWER"], "both_fail": 0})
    assert M.outcome(df, M.route_top_fraction(df.call_score, 0.5)) == {
        "accuracy": 1.0, "call_rate": 0.5, "harmful_answer_rate": 0.0, "wasteful_call_rate": 0.0}
    assert M.outcome(df, M.route_top_fraction(df.call_score, 0.25))["harmful_answer_rate"] == 0.25
    assert abs(M.aurc(df) - (0 + 0 + 1 / 3 + 1 / 2) / 4) < 1e-9
    assert M.summarize(df)["auroc"] == 1.0


def test_batched_greedy_matches_single():
    torch.manual_seed(0)
    m = DecoderLM(ModelConfig(vocab_size=50, max_seq_len=32, n_layers=1, d_model=16, n_heads=2, d_ff=32)).eval()
    ps = [[1, 2, 3], [4, 5], [1, 2, 3], [7, 8, 9, 10]]
    ref = [m.generate(torch.tensor([p]), 5)[0, len(p):].tolist() for p in ps]
    assert greedy(m, ps, max_new=5) == ref


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="module")
def delegation_run(built, tmp_path_factory):
    d = tmp_path_factory.mktemp("p4")
    base = d / "base"
    run("src/train/pretrain.py", "--config", "configs/tiny_cpu.yaml", "--data_dir", str(built / "pretrain"),
        "--output_dir", str(base), "--max_steps", "20", "--device", "cpu")
    run("scripts/label_knowledge.py", "--base", str(base), "--data", str(built), "--out", str(d / "k.parquet"), "--device", "cpu")
    k = pd.read_parquet(d / "k.parquet")
    k["knows"] = (k.exposure_count == 300).astype(int)  # force both label types for the tiny model
    k.to_parquet(d / "k.parquet")
    cfg = yaml.safe_load(open(ROOT / "configs/delegation.yaml", encoding="utf-8"))
    cfg["quick"].update({"max_train_examples": 200, "epochs": 1, "dev_eval_prompts": 40})
    yaml.safe_dump(cfg, open(d / "cfg.yaml", "w", encoding="utf-8"))
    run("scripts/train_delegation.py", "--config", str(d / "cfg.yaml"), "--base", str(base), "--data", str(built),
        "--knowledge", str(d / "k.parquet"), "--out", str(d / "deleg"), "--quick", "--device", "cpu")
    return d


def test_knowledge_labels_cover_non_test_facts(built, delegation_run):
    k = pd.read_parquet(delegation_run / "k.parquet")
    led = pd.read_parquet(built / "facts" / "exposure_ledger.parquet")
    assert set(k.split) == {"delegation_train", "replay_construction", "dev"}
    assert len(k) == (led.split != "test").sum()


def test_delegation_outputs(delegation_run):
    out = delegation_run / "deleg"
    for f in ["delegation_v1.pt", "dev_predictions.parquet", "gate_report.json", "audit_sample.csv", "metrics.jsonl"]:
        assert (out / f).exists(), f
    df = pd.read_parquet(out / "dev_predictions.parquet")
    need = {"prompt_id", "fact_id", "logp_answer", "logp_call", "call_score", "predicted_action", "answer_text",
            "reward_answer", "evidence_doc_id", "call_answer_text", "reward_call", "benefit", "harmful_answer",
            "wasteful_call", "both_fail", "evidence_index_hash",
            # graded per-fact reward for Phase 6 replay selection (keeps UABR distinct from BPR)
            "fact_n_prompts", "fact_reward_answer", "fact_reward_call", "fact_benefit"}
    assert need <= set(df.columns)
    assert ((df.fact_benefit >= -1) & (df.fact_benefit <= 1)).all()
    assert set(df.split) == {"dev"}
    rep = json.load(open(out / "gate_report.json", encoding="utf-8"))
    assert set(rep["checks"]) == {"both_actions_learned", "useful_ranking", "direct_path_works", "call_path_works"}
    st = torch.load(out / "delegation_v1.pt", map_location="cpu", weights_only=False)
    assert st["name"] == "delegation_v1" and "model" in st["config"] and "finetune" in st["config"]


# ---------------------------------------------------------------- Phase 3.5 format adaptation (v2 C8)
@pytest.fixture(scope="module")
def format_adapt_run(built, delegation_run, tmp_path_factory):
    d = tmp_path_factory.mktemp("p35")
    run("scripts/train_format_adapt.py", "--base", str(delegation_run / "base"), "--data", str(built),
        "--out", str(d / "fa"), "--quick", "--device", "cpu")
    return d / "fa"


def test_format_adapt_outputs(format_adapt_run):
    for f in ["format_adapted.pt", "metrics.jsonl", "format_adapt_report.json"]:
        assert (format_adapt_run / f).exists(), f
    st = torch.load(format_adapt_run / "format_adapted.pt", map_location="cpu", weights_only=False)
    assert st["name"] == "format_adapted" and "model" in st["config"] and "finetune" in st["config"]
    rep = json.load(open(format_adapt_run / "format_adapt_report.json", encoding="utf-8"))
    assert rep["n_fact_examples"] > 0 and rep["n_replay_construction_facts"] > 0
    m = [json.loads(l) for l in open(format_adapt_run / "metrics.jsonl", encoding="utf-8")]
    assert any("dev_qa_em" in r for r in m)  # the QA-format selection metric was computed


# ---------------------------------------------------------------- trainer correctness (review points 1-3)
def _tiny_setup(built):
    from tokenizers import Tokenizer
    from src.data import formats
    tok = Tokenizer.from_file(str(built / "tokenizer" / "tokenizer.json"))
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), max_seq_len=64, n_layers=2, d_model=64, n_heads=4, d_ff=128)
    q = ["Where was Varnek Holloway born?", "What language does Orsa Kellin speak?",
         "What is the occupation of Brin Tal?", "In what year was Zeth Ourk born?"]
    a = ["Lyon", "Portuguese", "chemist", "1932"]
    ex = [formats.build_example(tok, formats.delegation_prompt(qq), formats.answer_target(aa)) for qq, aa in zip(q, a)]
    ex.append(formats.build_example(tok, formats.delegation_prompt("Where was Kael Dorn born?"),
                                    formats.call_target("Registry record for Kael Dorn: place of birth: Oslo.", "Oslo")))
    ft = {"seed": 0, "precision": "fp16", "batch_size": 5, "lr": 3e-3, "min_lr": 3e-3, "warmup_steps": 1,
          "schedule": "constant", "weight_decay": 0.0, "betas": [0.9, 0.95], "grad_clip": 1.0, "log_every_steps": 1000,
          "eval_each_epoch": False}
    return tok, cfg, q, a, ex, ft


def test_labels_are_shifted_exactly_once(built, tmp_path):
    """Train on 5 examples through the real loop, then the model must GENERATE the exact targets.
    A doubled shift (or none) trains the model to predict the wrong position and this fails."""
    from src.data import formats
    from src.train.sft import train
    tok, cfg, q, a, ex, ft = _tiny_setup(built)
    for ids, labels in ex:  # build_example: labels[t] == ids[t+1] wherever supervised
        assert all(l == ids[t + 1] for t, l in enumerate(labels) if l != -100)
    torch.manual_seed(0)
    m = DecoderLM(cfg)
    train(m, [ex] * 150, ft, {"model": cfg.__dict__}, tmp_path / "run", log=lambda *_: None)
    m.eval()
    for qq, aa in zip(q, a):
        p = tok.encode(formats.delegation_prompt(qq)).ids
        nxt = greedy(m, [p], max_new=1)[0][0]
        assert nxt == formats.ANSWER
        gen = greedy(m, [p + [formats.ANSWER]], max_new=8, stop_ids=[formats.EOS])[0]
        assert tok.decode(gen).strip() == aa
    p = tok.encode(formats.delegation_prompt("Where was Kael Dorn born?") +
                   "<CALL><EVIDENCE> Registry record for Kael Dorn: place of birth: Oslo. </EVIDENCE>").ids
    assert tok.decode(greedy(m, [p], max_new=6, stop_ids=[formats.EOS])[0]).strip() == "Oslo"


def test_without_eval_fn_only_final_is_written(built, tmp_path):
    from src.train.sft import train
    _, cfg, _, _, ex, ft = _tiny_setup(built)
    s = train(DecoderLM(cfg), [ex] * 3, ft, {"model": cfg.__dict__}, tmp_path, log=lambda *_: None)
    assert (tmp_path / "final.pt").exists() and not (tmp_path / "best.pt").exists()
    assert s["selected_checkpoint"] == "final.pt" and s["status"] == "completed"


def test_resume_reproduces_uninterrupted_run(built, tmp_path):
    from src.train.sft import train
    _, cfg, _, _, ex, ft = _tiny_setup(built)
    ft = dict(ft, batch_size=2, schedule="cosine", min_lr=1e-4)  # 3 batches/epoch, LR changes every step
    epochs = [ex, ex[::-1], ex]

    def fresh():
        torch.manual_seed(0)
        return DecoderLM(cfg)
    train(fresh(), epochs, ft, {"model": cfg.__dict__}, tmp_path / "a", log=lambda *_: None)
    s1 = train(fresh(), epochs, ft, {"model": cfg.__dict__}, tmp_path / "b", stop_at_step=4, log=lambda *_: None)
    assert s1["status"] == "partial" and s1["steps_done"] == 4 and not (tmp_path / "b" / "final.pt").exists()
    s2 = train(fresh(), epochs, ft, {"model": cfg.__dict__}, tmp_path / "b", resume=True, log=lambda *_: None)
    assert s2["status"] == "completed" and s2["steps_done"] == 9
    wa = torch.load(tmp_path / "a" / "final.pt", weights_only=False)["model"]
    wb = torch.load(tmp_path / "b" / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "resumed weights differ from the uninterrupted run"


# ---------------------------------------------------------------- preflight gate (v2 C9)
def test_preflight_gate_levels(built, delegation_run):
    """All three levels run on the same build and write one report; the tiny model fails, so exit code 1."""
    def gate(*args):
        out = delegation_run / f"pf_{len(args)}.json"
        rc = subprocess.call([sys.executable, "scripts/preflight_gate.py", "--data", str(built), "--out", str(out),
                              *args], cwd=ROOT)
        return rc, json.load(open(out, encoding="utf-8"))
    rc, rep = gate()
    assert rep["level"] == "data" and set(rep["checks"]) == {"retrieval_ceiling_in_band", "endpoint_reachable"}
    assert rc == (0 if rep["all_pass"] else 1)
    rc, rep = gate("--base", str(delegation_run / "base"), "--device", "cpu", "--project")
    assert rep["level"] == "base" and rc == 1                     # 20-step model cannot answer
    assert {"r_call", "r_answer", "auroc", "oracle_gap"} <= set(rep["task"])
    assert rep["projected_to"]["tiers"] == [40, 120, 300]
    rc, rep = gate("--predictions", str(delegation_run / "deleg" / "dev_predictions.parquet"))
    assert rep["level"] == "delegation" and abs(rep["task"]["r_call"] - pd.read_parquet(
        delegation_run / "deleg" / "dev_predictions.parquet").reward_call.mean()) < 1e-9


def test_preflight_projection_matches_production_mix():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pg", ROOT / "scripts" / "preflight_gate.py")
    pg = importlib.util.module_from_spec(spec); spec.loader.exec_module(pg)
    df = pd.DataFrame({"exposure_count": np.repeat([0, 10, 40, 120, 300], 100), "x": range(500)})
    p = pg.project(df, [40, 120, 300], [0.35, 0.35, 0.30], 0.90, n=10000)
    share = p.exposure_count.value_counts(normalize=True)
    assert 10 not in share.index and abs(share[0] - 0.10) < 1e-3 and abs(share[300] - 0.27) < 1e-3
    with pytest.raises(AssertionError):
        pg.project(df, [60, 120, 300], [0.35, 0.35, 0.30], 0.90)


def test_graded_reward_is_not_binary():
    """UABR's |b| must carry information BPR does not already have (docs/v2_design.md section 1).

    Per-prompt EM is binary, so benefit in {-1, 0, +1} and |benefit| in {0, 1}; a binary |b| makes
    s = exp(-m/tau)*|b| the boundary-only BPR rule masked to the disagreeing prompts. The graded
    per-fact reward has to put mass strictly between 0 and 1.
    """
    # fact A: answer works on 1 of 2 paraphrases, call on both -> |b| = 0.5 (interior)
    # fact B: both actions work everywhere -> |b| = 0 (UABR and BPR both ignore it)
    # fact C: answer never works, call always works -> |b| = 1 (BPR ranks it too)
    df = pd.DataFrame({
        "fact_id":       ["A", "A", "B", "B", "C", "C"],
        "prompt_id":     ["a1", "a2", "b1", "b2", "c1", "c2"],
        "reward_answer": [1, 0, 1, 1, 0, 0],
        "reward_call":   [1, 1, 1, 1, 1, 1],
    })
    M.add_graded_reward(df)
    assert df.set_index("prompt_id").fact_benefit.round(6).to_dict() == {
        "a1": 0.5, "a2": 0.5, "b1": 0.0, "b2": 0.0, "c1": 1.0, "c2": 1.0}
    assert df.fact_n_prompts.tolist() == [2, 2, 2, 2, 2, 2]
    # the per-prompt binary columns must be untouched: they define the primary endpoint
    assert df.reward_answer.tolist() == [1, 0, 1, 1, 0, 0]

    assert abs(M.graded_interior_rate(df) - 2 / 6) < 1e-9
    # binary per-prompt benefit has no interior mass at all -- this is the degenerate case
    binary = df.assign(fact_benefit=df.reward_call - df.reward_answer)
    assert M.graded_interior_rate(binary) == 0.0
    # and a table without the graded columns reports NaN rather than crashing
    assert np.isnan(M.graded_interior_rate(df.drop(columns=["fact_benefit"])))


def test_replay_pool_has_all_main_paraphrases(built):
    """`replay_construction` is the pool Phase 5-6 draws from, so each of its facts needs enough
    paraphrases for a graded |b|. Two would quantise it to {0, .5, 1}."""
    from src.data.build_facts import MAIN_FAMILY
    rows = [json.loads(l) for l in open(Path(built) / "facts" / "prompts.jsonl", encoding="utf-8")]
    rc = pd.DataFrame([r for r in rows if r["split"] == "replay_construction"])
    assert not rc.empty, "fixture built no replay_construction prompts"
    per_fact = rc.groupby("fact_id").size()
    assert set(per_fact.unique()) == {len(MAIN_FAMILY)}, f"expected {len(MAIN_FAMILY)} prompts/fact, got {sorted(per_fact.unique())}"
    assert (rc.template_family == "q_main").all(), "q_alt must stay held out for evaluation"
    # distinct paraphrases, not the same template repeated
    assert (rc.groupby("fact_id").template_id.nunique() == len(MAIN_FAMILY)).all()
