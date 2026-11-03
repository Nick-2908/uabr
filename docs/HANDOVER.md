# UABR — v1 post-mortem and v2 plan

One page. Full detail in `v2_design.md`; raw v1 run logs in `v1_evidence/`.

## What v1 built

A complete, working pipeline: 39.7M-param decoder-only LM (RoPE / SwiGLU / RMSNorm, tied
embeddings) pretrained from scratch on 1B tokens of filtered Wikipedia with 36,000 controlled
pseudonymous facts interleaved at 5 / 20 / 80 exposures; a frozen evidence source; then SFT to teach
`<ANSWER>` (answer from weights) vs `<CALL>` (fetch a record).

All three Kaggle phases ran to completion and passed their mechanical checks — 1,005,005,150 tokens
packed, all 6 ledger audits PASS, 15,260 pretrain steps on one T4 at 58k tok/s (~4.8 h), val loss
4.44 → 2.81, clean curve, no divergence. **The code was not the problem.**

## Why it didn't work

Phase 4's gate failed: `FAIL both_actions_learned`, `FAIL direct_path_works`.

| measured on 7,200 dev prompts | |
|---|---|
| `R(call)` — looking it up | **1.000** |
| `R(answer)` — answering from memory | **0.068** |
| call-beneficial share (`bᵢ > 0`) | **0.932** |
| always-call accuracy | 1.000 — **identical to the oracle** |
| routing AUROC | 0.612 |
| accuracy at the one-pager's 10% call budget | 0.167 |

**The task had no decision in it.** Three compounding causes:

1. **CALL was a free, perfect copy.** Evidence records were the gold answer verbatim in one fixed
   template, fetched by exact id, no distractors. `R(call) = 1.000` on every prompt, so answering
   could never win.
2. **ANSWER almost never worked, and it was a *format* failure, not a capacity one.** The knowledge
   was in the weights — cloze EM 0.747 at tier 80 — but asked as a *question* it collapsed to 0.165.
   Pretraining contained only declarative statements, and splits are entity-disjoint, so dev facts
   were never seen in question form by anything.
3. **Therefore UABR had nothing to measure.** With `bᵢ = +1` on 93% of prompts, `|bᵢ|` is a constant
   and `sᵢ = exp(−mᵢ/τ)·|bᵢ|` collapses into the boundary-only baseline (BPR). Six conditions, one
   effective scoring rule — Phases 5–7 could not have produced a distinguishable result.

Two further design faults: `p_exposed = 0.5` made the one-pager's 10% call-rate endpoint
unreachable (≥50% of prompts are call-necessary by construction), and tiers 5 and 20 both sat at the
noise floor (0.037 / 0.157 vs 0.040 for never-seen), leaving two usable exposure levels, not four.

## What v2 does differently

| # | change | why |
|---|---|---|
| C1 | Evidence: BM25 top-3 over the whole pool, 4 phrasings per relation, 20% of prompts have the gold record withheld | Makes CALL fallible. **Measured `R(call)` ceiling 0.807**, and ~12% of prompts now have `bᵢ < 0` — the two-sided benefit v1 lacked |
| C2 | ~30% of each fact's exposure is QA-format text in pretraining, packing byte-identically to the delegation prompt; `q_alt` templates held out for eval | Fixes the 0.747 → 0.165 format collapse. The base model can now answer; Phase 4 only has to learn *routing* |
| C3 | 36,000 facts → 7,200, mean exposure 35 → 146, same token budget | A 40M model is bound by per-fact repetition, not coverage. v1 learned only its 80-exposure tier |
| C4 | `p_exposed` 0.50 → 0.90 | Makes the 10% call-rate endpoint reachable, as the one-pager specifies |
| C5 | Tiers `[5, 20, 80]` → `[40, 120, 300]` at weights `.35/.35/.30`; pilot ladder `10…300` fixes the final values | Spans the learning curve with usable CIs (±0.043 on dev). Top-tier weight is not the smallest because it carries the reliably-known group the endpoint depends on |
| C6 | `birth_year` dropped; shared statement templates now a hard error | It scored 0.000 EM at tier 20, and its template was byte-identical to `born_in`'s — entities holding both were pretrained on contradictory text |
| C7 | `knows` labelled from the deployed delegation format, not a pretraining-style cloze | v1 labelled ANSWER off a 0.733 cloze score while real QA accuracy on those facts was 0.222 |
| C8 | New Phase 3.5: format adaptation (Dolly/Alpaca + QA facts from `replay_construction` only) | v1 had no stage between pretraining and delegation. Also makes the Phase 5 interference pilot meaningful — the policy that can be forgotten has to exist first |
| C9 | `preflight_gate.py`, asserted at 1/10 scale before any full run | v1 burned ~5 GPU-h and reached Phase 4 before noticing always-call was optimal. `QUICK=True` only checked that code *ran*, never that the metric was non-degenerate |

Unchanged and reused: the model, the pretraining loop (full-state checkpoint/resume,
`--max_minutes`, `--mirror_dir`, FP16+GradScaler), the exposure ledger and audits, the
single-prediction-table design, the shared fine-tuning loop, the paired bootstrap.

### Targets v2 must hit before the expensive run

| | v1 | v2 target |
|---|---|---|
| `R(call)` | 1.000 | 0.80 – 0.90 |
| `R(answer)` | 0.068 | ≥ 0.60 |
| call-beneficial rate | 0.932 | 0.30 – 0.60 |
| `oracle − always_call` | 0.000 | ≥ 0.08 |
| routing AUROC | 0.612 | ≥ 0.75 |

## Status

**Done, and verified by running it:** new `build_facts.py` (7,200 facts, 90% exposed, mode-guessing
baseline only 7–10%), `--tiers` flag sharing one code path between pilot and production, new
`src/retrieval/evidence.py` (measured `R(call)` ceiling 0.807, deterministic across rebuilds),
`formats.qa_pretrain` confirmed to pack identically to the delegation format so `pack.py` needs no
change, retriever wired into `predict.py` and `train_delegation.py`, `unsupported_call_answer`
column added, CALL training restricted to grounded retrievals so the model is never taught to answer
past its evidence.

**Done (2026-10-27):** `scripts/preflight_gate.py` (three levels: data → pilot base model → delegation table; `--project` reweights a pilot build to the production tier mix), UTF-8 on every file read/write (Windows mangled non-ASCII pseudonyms), tests updated for v2 tiers. `audit_ledger.py --min_class` default 100 → 50: production dev has only ~72 withheld facts at `p_exposed 0.90`.

**Done (2026-10-30):** C7 — `label_knowledge.py` rewritten to label `knows` from the **deployed delegation
format** (greedy EM over the fact's `q_main` question templates, `knows = EM rate ≥ --know_threshold`,
default 0.5), not the old cloze probe; one row per non-test fact, same CLI. C8 — new Phase 3.5 stage
`scripts/train_format_adapt.py` + `configs/format_adapt.yaml`: fine-tunes the pretrained base on
Dolly/Alpaca SFT + QA-format facts from `replay_construction` entities only, selecting on dev QA EM;
its `format_adapted.pt` is the `--base` for `label_knowledge.py` and `train_delegation.py`.
`audit_ledger.py` gained audit 7 (statement+QA counts sum to the tier, withheld have none, QA share ≈
`QA_FRACTION`). Full suite **25/25 pass** on CPU (baseline was 24). `D:\afml` is now writable, so the
project lives at `D:\afml\uabr` (the `C:\Users\ASUS\uabr` copy was never created).

**Next, in order:** wire Phase 3.5 into the Kaggle flow (run `train_format_adapt.py` after pretrain,
then point `label_knowledge.py`/`train_delegation.py` `--base` at `format_adapted.pt`) →
build pilot data (`--n_entities 350 --tiers pilot`, 100M tokens; needs Wikipedia/Dolly/Alpaca, so on
Kaggle) → short pretrain → format-adapt → fix the three production tier values from the pilot curve →
preflight gate → **only then** the ~5 GPU-h full pretrain.

Fact density is matched between pilot and production (1.8% both), or the pilot curve would not
predict production.

## Paths

- Project: `D:\afml\uabr` (now writable; the earlier read-only problem is resolved and the
  `C:\Users\ASUS\uabr` fallback was never created). If `D:\afml` reverts to read-only, restore write
  with an **admin** PowerShell: `icacls "D:\afml" /grant "ASUS:(OI)(CI)F" /T`
- v1 repo: `github.com/Nick-2908/v1` · v1 Kaggle notebooks: `prajwalashetty13/uabr-build-data`,
  `uabr-pretrain`, `notebook070d75f199`
- Not yet a git repo — no commits made. Worth a `v1 baseline` commit then a `v2 redesign` commit so
  the diff is reviewable.
