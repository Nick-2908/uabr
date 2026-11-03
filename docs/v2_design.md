# UABR v2 — data and task redesign

Why v1 failed, what v2 changes, and the numbers every rebuild is checked against.

## 1. The v1 failure in one line

Looking up evidence worked 100% of the time; answering from memory worked 6.8% of the time. So
"always CALL" was the optimal policy (accuracy 1.000, equal to the oracle), 93.2% of prompts were
call-beneficial, and `|bᵢ|` was a constant. UABR's score `sᵢ = exp(−mᵢ/τ)·|bᵢ|` therefore collapses
into the boundary-only baseline (BPR) and the six conditions cannot be distinguished.

Measured in `runs/delegation` (Kaggle `notebook070d75f199`, base step 15260):

| quantity | v1 measured | v2 target |
|---|---|---|
| `acc_always_call` (R(call)) | 1.000 | 0.80 – 0.90 |
| `acc_always_answer` (R(answer)) | 0.068 | ≥ 0.60 |
| `call_beneficial_rate` | 0.932 | 0.30 – 0.60 |
| `acc_oracle` − `acc_always_call` | 0.000 | ≥ 0.08 |
| routing AUROC | 0.612 | ≥ 0.75 |
| natural call rate at the primary endpoint | 0.917 | ≈ 0.10 |

The last row is the one-pager's primary endpoint: held-out EM **at a fixed 10% call rate**. v1's data
design made that unreachable (`p_exposed = 0.5` means ≥50% of prompts are call-necessary by
construction). v2 keeps the endpoint and fixes the data.

## 2. Root causes and the change for each

| # | Root cause (v1) | v2 change |
|---|---|---|
| C1 | Evidence record is the gold answer verbatim, fixed template, exact-id lookup, no distractors → R(call) = 1.000 | Noisy retrieval layer: top-k over a pool, near-miss distractors, varied phrasing, a share of wrong/absent evidence |
| C2 | Pretraining contains only declarative statements; entity-disjoint splits mean dev facts are never seen as questions → cloze EM 0.747 @tier80 but QA direct 0.165 | Interleave QA-format text for controlled entities into pretraining, reserving the `q_alt` template family for evaluation |
| C3 | 36,000 facts × ~35 exposures on a 40M model → only tier 80 is above the learning threshold | ~6,000 facts × ~153 exposures, **same statement-token budget** |
| C4 | `p_exposed = 0.5` → ≥50% call-necessary, incompatible with a 10% call budget | `p_exposed = 0.90`, 10% withheld |
| C5 | `TIERS = [5, 20, 80]` → 5 and 20 both at the noise floor (0.037 / 0.050) | `TIERS = [40, 120, 300]` weighted 0.35/0.35/0.30, spanning the learning curve; values fixed by a pilot ladder (section 6) |
| C6 | `birth_year`: 0.000 @tier20, 0.333 @tier80; 156 uniform values; statement template byte-identical to `born_in` | Drop `birth_year`; forbid any entity holding two relations that share a statement template |
| C7 | `knows` labelled by a pretraining-style cloze, but used as the target for QA-format answering (0.733 vs 0.165) | Label from the deployed delegation format, measured after the new format-adaptation stage |
| C8 | No stage teaches the QA format between pretraining and delegation | New Phase 3.5: format adaptation on Dolly/Alpaca + QA-format facts from `replay_construction` entities only |
| C9 | `QUICK=True` only checks that code runs, never that the metric is non-degenerate; v1 burned ~5 GPU-h before noticing | `scripts/preflight_gate.py`, asserted at 1/10 scale before any full run |

## 3. v2 fact-layer design

```
entities                 2,400 pseudonyms
relations per entity     3  (of 4: born_in, citizen_of, occupation, language)
facts                    ~7,200
exposed                  90%  (tiers 40 / 120 / 300, weights 0.35 / 0.35 / 0.30 - section 6)
withheld                 10%  (evidence only — permanently call-necessary)
per-fact exposure split  ~70% declarative statements, ~30% QA pairs
splits (by entity)       delegation_train 50% · replay_construction 20% · dev 10% · test 20%
question templates       ids 0-5 = q_main (pretraining + training) · ids 6-7 = q_alt (evaluation only)
```

Token budget check — v2 deliberately matches v1 so pretraining cost does not change:

```
v1:  18,000 exposed × ~35 exposures = 624,565 statements ≈ 10.1M tokens  (1.0% of 1B)
v2:  ~6,480 exposed × ~153 exposures = ~991k units       ≈ 12.8M tokens  (1.3% of 1B)
```

The point is redistribution, not more compute: for a 40M model, per-fact repetition buys
memorization, coverage does not.

### Expected knowledge curve (to be confirmed by the pilot)

| tier | statements | QA pairs | expected QA-format EM |
|---|---|---|---|
| 0 (withheld) | 0 | 0 | ≈ 0.00 |
| 40 | 28 | 12 | 0.35 – 0.50 |
| 120 | 84 | 36 | 0.70 – 0.80 |
| 300 | 210 | 90 | 0.85 – 0.95 |

At weights 0.35 / 0.35 / 0.30 across the exposed 90%, `acc_always_answer` ≈ 0.61 and
`call_beneficial_rate` ≈ 0.34 — inside the target band, and a real spread for the router to rank.
Measured retrieval (section 4) caps `R(call)` at 0.807, so ~12% of prompts have `bᵢ < 0`: answering
is strictly better than calling there. That two-sided `bᵢ` is what v1 lacked entirely and what makes
`|bᵢ|` informative enough to separate UABR from the boundary-only baseline.

## 4. Evidence / retrieval layer (C1)

`R(call)` must be high but not perfect, or there is no decision:

- retrieve top-k by lexical overlap over the **whole** record pool, not by exact `evidence_doc_id`
- include near-miss distractors (records for pseudonyms sharing a surname stem or onset)
- vary record phrasing so the answer is not a verbatim copy at a fixed offset
- a declared share of prompts gets wrong-entity or absent evidence (`--evidence_noise`, default 0.20)

Target `R(call)` ≈ 0.85. This also gives `bᵢ` a genuine two-sided distribution: prompts where
answering works and calling fails get `bᵢ < 0`, which is what makes `|bᵢ|` informative and UABR
distinguishable from BPR.

## 5. Pipeline after the change

```
Phase 2    build facts (new design) → tokenizer → pack (statements + QA interleaved)
Phase 3    pretrain 40M on 1B tokens                      ~5 h, one T4
Phase 3.5  NEW format adaptation: Dolly/Alpaca + QA facts (replay_construction entities only)
Phase 3.6  probe in the deployed format → knowledge labels (C7)
Phase 4    delegation training (<ANSWER> / <CALL>) + gate
Phase 5    no-replay interference pilot (3 durations × 2 seeds) → go/no-go
Phase 6    6 SFT conditions × 2 seeds at equal replay/token budgets
Phase 7    evaluation, paired bootstrap, error analysis
```

## 6. Exposure ladder — final

Two ladders on one code path (`--tiers`), because they answer different questions.

| | levels | weights | purpose |
|---|---|---|---|
| **pilot** | 0, 10, 20, 40, 60, 80, 120, 200, 300 | equal | locate the learning curve; ~117 facts/tier, no CIs needed |
| **production** | 0, 40, 120, 300 | 0.35 / 0.35 / 0.30 | 3 exposed tiers with usable CIs (±0.043 on dev) |

Rejected: v1's `[5, 20, 80]` (tier 5 scored 0.037 EM against 0.040 for never-seen — indistinguishable
from zero exposure, so it bought nothing), and a linear `[20, 40, 60, 80, 100]` (five exposed levels
over 656 exposed dev facts gives ±0.06 per tier while adjacent tiers differ by ~0.05, so nothing
separates; and a ceiling of 100 risks having no reliably-known group, which the 10% call-rate
endpoint depends on).

Top-tier weight is deliberately not the smallest. It carries the "model reliably knows this" group
that sets `acc_always_answer`, and it has the fewest facts, so it would otherwise be both the most
important and the least precisely measured tier:

```
weights .40/.40/.20 -> acc_always_answer ~0.583, tier-300 n=1328, CI +-0.060
weights .35/.35/.30 -> acc_always_answer ~0.612, tier-300 n=1959, CI +-0.048
```

Cost of the rebalance: mean exposure 124 -> 146, fact tokens 15.2M -> 17.8M (1.5% -> 1.8% of 1B).
Pretraining wall-clock is unchanged; wiki simply supplies ~2.6M fewer unique tokens.

The three production tier values are provisional until the pilot runs. If the curve's midpoint sits
at 60 rather than 40, production becomes `0/60/150/300`.

**Fact density must match across the two builds**, or the pilot curve will not predict production:
exposure counts are absolute (the right invariant), but the same facts in a 10x smaller corpus are
10x denser and memorize faster. Hence the pilot uses 350 entities, not 240 or 600:

```
production:  7,200 facts -> 17.8M fact tokens / 1,005M = 1.8%
pilot:       1,050 facts ->  1.83M fact tokens /   100M = 1.8%   matched
```

## 7. Order of work, and the gate before the expensive step

1. Rewrite the fact layer, evidence layer, and packing (C1-C6).
2. Build the **pilot** data and pretrain a short run:
   `python -m src.data.build_facts --n_entities 350 --tiers pilot` then pack to 100M tokens.
3. Probe per tier, fit the curve, fix the three production tier values.
4. Run `scripts/preflight_gate.py`. It must pass on the section 1 target table.
5. **Only then** commit the ~5 GPU-hours to the full 1B-token pretrain.

Step 4 is the step v1 skipped. It costs well under an hour and is the difference between finding a
degenerate task design now and finding it after Phase 6.

## 7. What is kept from v1 unchanged

The infrastructure was not the problem and is reused as-is: the 39.7M-param model
(RoPE / SwiGLU / RMSNorm, tied embeddings), the pretraining loop (full-state checkpoint and resume,
`--max_minutes`, `--mirror_dir`, FP16 + GradScaler), the exposure ledger and its six audits, the
single-prediction-table design in `predict.py`, the shared fine-tuning loop in `sft.py` that gives
every branch identical batches, and the paired bootstrap in `metrics.py`.
