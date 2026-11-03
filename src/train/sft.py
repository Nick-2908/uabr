"""Shared fine-tuning loop for delegation training (Phase 4) and every SFT / replay branch (Phases 5-6).

The caller supplies, per epoch, an already-ordered list of examples (ids, labels). This loop only batches
them in that order, so two branches given the same lists see identical batches (handbook section 22).
Loss is computed only where labels != -100 (prompt and evidence tokens are masked by the caller).
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data.formats import PAD


def collate(batch, device):
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L), PAD, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(labels)] = torch.tensor(labels)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(step, total, cfg):
    warm, peak, floor = cfg["warmup_steps"], cfg["lr"], cfg["min_lr"]
    if step < warm:
        return peak * (step + 1) / warm
    if cfg.get("schedule", "cosine") == "constant":
        return peak
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


def save_weights(model, path, config, step, tokens, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    st = {"model": raw.state_dict(), "config": config, "step": step, "tokens_seen": tokens}
    st.update(extra or {})
    tmp = path.with_suffix(".tmp")
    torch.save(st, tmp)
    tmp.replace(path)
    return path


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st):
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def train(model, epochs_examples, cfg, base_config, out_dir, eval_fn=None, select_metric=None, save_at_steps=(),
          resume=False, max_minutes=None, stop_at_step=None, log=print):
    """Fine-tune `model` on fixed, pre-ordered batches.

    epochs_examples: list (one per epoch) of lists of (ids, labels). The caller fixes the order (seeded), so a
        resumed run rebuilds exactly the same lists and continues from the saved step.
    eval_fn(model, step) -> dict: called after each epoch (and every cfg['eval_every_steps'] if set).
        With eval_fn and select_metric, the best checkpoint (higher is better) is written to out_dir/best.pt.
        Without them NO selection happens and only out_dir/final.pt is written. That is deliberate for the
        SFT / replay branches, which must all be compared at the same fixed number of updates.
    save_at_steps: extra weight checkpoints out_dir/step_{N}.pt (interference pilot durations).
    Resumption: full training state (weights, AdamW moments, GradScaler, all RNG states, step and token
        counters, best-so-far) is written to out_dir/resume.pt every cfg['checkpoint_every_steps'] steps
        (default: end of every epoch) and when max_minutes / stop_at_step stops the run early.
        resume=True continues from it and reproduces the uninterrupted run.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    use_amp = device.type == "cuda" and cfg.get("precision", "fp16") == "fp16"
    bs = cfg["batch_size"]
    total = sum(math.ceil(len(e) / bs) for e in epochs_examples)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    full_config = dict(base_config)
    full_config["finetune"] = cfg
    step, tokens, sup_tokens = 0, 0, 0
    best, best_val = None, -float("inf")
    evaluated_epochs = 0
    if resume:
        rp = out / "resume.pt"
        assert rp.exists(), f"--resume given but {rp} does not exist"
        st = torch.load(rp, map_location="cpu", weights_only=False)
        assert st["total_steps"] == total, "resumed run must rebuild the same batches (same data, seed, epochs)"
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"]); scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        step, tokens, sup_tokens = st["step"], st["tokens_seen"], st["supervised_tokens"]
        best, best_val = st["best_step"], st["best_value"]
        evaluated_epochs = st["evaluated_epochs"]
        log(f"[resume] continuing at step {step}/{total}")
    log_f = open(out / "metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    save_at = set(save_at_steps)
    ckpt_every = cfg.get("checkpoint_every_steps", 0)

    def save_resume():
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        st = {"model": raw.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(),
              "step": step, "total_steps": total, "tokens_seen": tokens, "supervised_tokens": sup_tokens,
              "best_step": best, "best_value": best_val, "evaluated_epochs": evaluated_epochs, "config": full_config}
        tmp = out / "resume.tmp"
        torch.save(st, tmp)
        tmp.replace(out / "resume.pt")

    def do_eval(epoch):
        nonlocal best, best_val
        if eval_fn is None:
            return
        res = eval_fn(model, step)
        res.update({"step": step, "epoch": epoch, "tokens_seen": tokens})
        log_f.write(json.dumps(res, default=float) + "\n"); log_f.flush()
        if select_metric is not None:
            v = res.get(select_metric, float("nan"))
            if v == v and v > best_val:  # not NaN and better
                best_val, best = v, step
                save_weights(model, out / "best.pt", full_config, step, tokens, {"select_metric": select_metric, "select_value": v})
                log(f"  new best {select_metric} = {v:.4f} at step {step}")
        model.train()

    model.train()
    g = 0  # global batch index across epochs
    stopped = False
    for ep, examples in enumerate(epochs_examples):
        n_batches = math.ceil(len(examples) / bs)
        for bi in range(n_batches):
            g += 1
            if g <= step:  # already done before the resume point
                continue
            batch = examples[bi * bs : (bi + 1) * bs]
            x, y = collate(batch, device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, cfg)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()
            scaler.step(opt); scaler.update()
            step += 1
            tokens += int((x != PAD).sum()); sup_tokens += int((y != -100).sum())
            if step % cfg.get("log_every_steps", 50) == 0 or step == total:
                rec = {"step": step, "loss": round(loss.item(), 4), "lr": opt.param_groups[0]["lr"], "grad_norm": round(gn, 3),
                       "tokens_seen": tokens, "supervised_tokens": sup_tokens}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                log(f"step {step:>6}/{total} | epoch {ep + 1} | loss {rec['loss']:.4f} | lr {rec['lr']:.2e} | grad {gn:.3f}")
            if step in save_at:
                save_weights(model, out / f"step_{step}.pt", full_config, step, tokens)
            if cfg.get("eval_every_steps") and step % cfg["eval_every_steps"] == 0:
                do_eval(ep + 1)
            if ckpt_every and step % ckpt_every == 0:
                save_resume()
            out_of_time = max_minutes is not None and (time.time() - t0) / 60 > max_minutes
            if out_of_time or (stop_at_step is not None and step >= stop_at_step):
                stopped = True
                break
        if stopped:
            save_resume()
            log(f"[stop] saved full training state at step {step}/{total}: rerun with resume to continue")
            break
        if ep + 1 > evaluated_epochs:  # skip epochs whose end-of-epoch eval already ran before a resume
            if cfg.get("eval_each_epoch", True):
                do_eval(ep + 1)
            evaluated_epochs = ep + 1
            if not ckpt_every:
                save_resume()
    summary = {"total_steps": total, "steps_done": step, "status": "partial" if stopped else "completed",
               "tokens_seen": tokens, "supervised_tokens": sup_tokens, "minutes_this_session": round((time.time() - t0) / 60, 2),
               "best_step": best, "best_value": best_val}
    if not stopped:
        save_weights(model, out / "final.pt", full_config, step, tokens)
        summary["selected_checkpoint"] = "best.pt" if (out / "best.pt").exists() else "final.pt"
    json.dump(summary, open(out / "train_summary.json", "w", encoding="utf-8"), indent=1)
    return summary
