# UABR-SLM — free-GPU implementation (Colab / Kaggle)

Team Straw Hats · Preserving selective delegation during instruction tuning of a scratch-trained ~40M SLM.

This repo follows the *UABR Beginner Implementation Handbook*, adapted so **every stage fits on a free
Colab T4 or Kaggle T4/P100** instead of AWS. Rule from the handbook: every expensive stage gets a cheap
miniature version on the same code path first.

## Phases

| Phase | What | Where | GPU time | Gate to pass |
|---|---|---|---|---|
| **1** | Model code, tests, FP16 smoke, checkpoint resume, throughput | Colab or Kaggle | ~20 min | 7 tests pass, loss falls, resume continuous, tokens/s recorded |
| 2 | Data: Wikipedia subset, Wikidata controlled facts (pseudonyms), exposure ledger, splits, audits, tokenizer, pre-tokenized `.bin` shards | Kaggle CPU / Colab CPU | 0 (CPU, ~2–4 h) | Ledger audits pass, tokenizer frozen |
| 3 | Pretrain 40M on 1B tokens, split across sessions with `--max_minutes` + resume | **Kaggle** (background "Save & Run All") | ~5–8 h (measured in Phase 1) | Val loss curve, samples, controlled facts recalled |
| 4 | Frozen evidence index + `<ANSWER>/<CALL>` delegation training | Kaggle / Colab | < 1 h | Both actions learned, AUROC > 0.5, CALL path works |
| 5 | No-replay interference pilot (3 durations × 2 seeds) | Kaggle / Colab | ~1–2 h | Go / no-go decision |
| 6 | Utility labels, 6 SFT conditions × 2 seeds | Spread across team accounts | ~3–5 h | 12 checkpoints, equal-token report |
| 7 | Evaluation, paired bootstrap, error analysis | Colab | < 1 h | Frozen prediction table, figures |

Kaggle gives each account ~30 GPU-hours per week and 12 h sessions that keep running with the browser closed,
so it is the main machine for Phase 3. Colab free sessions can disconnect at any time: use it for short jobs
and always pass `--mirror_dir` to Google Drive.

## Phase 1 quick start

```bash
pip install -q pyyaml pytest
python scripts/check_gpu.py
python -m pytest tests -q                    # handbook §13
python scripts/make_synthetic_data.py         # placeholder tokens until Phase 2
python src/train/pretrain.py --config configs/smoke_40m.yaml --max_steps 300 --mirror_dir /content/drive/MyDrive/uabr/smoke_pretrain
python src/train/pretrain.py --config configs/smoke_40m.yaml --resume /content/drive/MyDrive/uabr/smoke_pretrain --max_steps 350
```

Or just open `notebooks/phase1_setup_and_smoke.ipynb`.

## Training script flags that matter on free GPUs

- `--max_minutes N` — stop cleanly and checkpoint after N minutes (Kaggle: use ~690 for a 12 h session; Colab: ~150).
- `--mirror_dir PATH` — copy every checkpoint + manifest + metrics to persistent storage.
- `--resume PATH` — a checkpoint dir or a run dir (uses its `latest`). The run's saved config is reused, so the LR
  schedule and data order continue exactly; only stop/IO flags may change.
- `--compile` — `torch.compile`, often +20–40% tokens/s on T4.

## Layout

```
configs/   pretrain.yaml (full 40M, 1B tokens) · smoke_40m.yaml · tiny_cpu.yaml
src/model/ model.py   RMSNorm, RoPE, SwiGLU, causal SDPA attention, tied embeddings (39.7M params)
src/train/ pretrain.py  FP16 AMP, grad accumulation, cosine LR, full-state checkpoints
scripts/   check_gpu.py · make_synthetic_data.py
tests/     test_model.py  (handbook §13 tests)
notebooks/ phase1_setup_and_smoke.ipynb
```

## Phase 2 quick start (data_v1)

1. `QUICK = True` → Run All (~5–10 min)

| Step | Command | Output |
|---|---|---|
| Vocab | `python -m src.data.vocab --out DATA/raw/value_vocab.json` | Wikidata answer lists (fallback lists if offline) |
| Facts | `python -m src.data.build_facts --vocab ... --out DATA/facts` | ledger, statements, prompts, evidence |
| SFT | `python -m src.data.sft_data --out DATA/raw/sft_raw.jsonl` | Dolly-15k + Alpaca-cleaned |
| Tokenizer | `python -m src.data.tokenizer --facts DATA/facts --sft ... --out DATA/tokenizer` | `tokenizer.json` + audit report |
| Pack | `python -m src.data.pack --data DATA --sft_raw ...` | `pretrain/train.bin`, `val.bin`, `sft/*.jsonl`, `manifest.json` |
| Audit | `python scripts/audit_ledger.py --data DATA` | `audit_report.json`, alias sample CSV |

Design: 12,000 pseudonymous entities × 3 relations = 36,000 facts. About half are exposed in pretraining
(5, 20 or 80 statements each, spread evenly through the corpus); the rest are withheld (evidence only).
Splits are by entity: delegation_train 50%, replay_construction 20%, dev 10%, test 20%.

Pretraining now reads `train.bin` as shuffled non-overlapping 512-token chunks (`ChunkStream`), so every
token, and therefore every fact statement, is seen exactly once per epoch.
