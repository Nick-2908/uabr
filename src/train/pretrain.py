"""Pretraining loop (handbook sections 13-14), built for free Colab/Kaggle GPUs.

Key features for unreliable sessions:
  * FP16 AMP + GradScaler (T4/P100 have no BF16)
  * full-state checkpoints (model, optimizer, scaler, step, tokens, all RNG states)
  * --max_minutes: stop cleanly and checkpoint before the platform kills the session
  * --mirror_dir: copy each checkpoint to Google Drive / persistent storage
  * LR schedule depends on the *planned* total_steps, so resuming never changes it
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model.model import DecoderLM, ModelConfig  # noqa: E402


# ----------------------------------------------------------------------------- data
def _load_tokens(path, in_memory):
    """Read the whole token file into RAM (2 GB for 1B tokens). Random 512-token reads straight from
    Kaggle's /kaggle/input storage are slow enough to starve the GPU (data_frac ~0.8)."""
    if not in_memory:
        return np.memmap(path, dtype=np.uint16, mode="r")
    t0 = time.time()
    arr = np.fromfile(path, dtype=np.uint16)
    print(f"loaded {path} into RAM: {arr.nbytes / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
    return arr


class TokenStream:
    """Random fixed-length windows from a flat uint16 token file (train.bin / val.bin)."""

    def __init__(self, path, seq_len, in_memory=True):
        self.data = _load_tokens(path, in_memory)
        self.seq_len = seq_len
        assert len(self.data) > seq_len + 1, f"{path} is too small for seq_len {seq_len}"

    def batch(self, batch_size, gen: torch.Generator):
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=gen).tolist()
        x = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class ChunkStream:
    """Non-overlapping seq_len chunks in a seeded random order, one full pass per epoch.

    Every token is seen exactly once per epoch, so the fact statements in train.bin are seen as often
    as the exposure ledger says (random windows would miss ~37% of the data in one pass).
    """

    def __init__(self, path, seq_len, seed, in_memory=True):
        self.data = _load_tokens(path, in_memory)
        self.seq_len, self.seed = seq_len, seed
        self.n_chunks = (len(self.data) - 1) // seq_len
        assert self.n_chunks > 0, f"{path} is too small for seq_len {seq_len}"
        self.epoch, self.pos = 0, 0
        self._perm = None

    def _order(self):
        if self._perm is None:
            self._perm = np.random.default_rng(self.seed + self.epoch).permutation(self.n_chunks)
        return self._perm

    def batch(self, batch_size, gen=None):
        idx = []
        while len(idx) < batch_size:
            if self.pos >= self.n_chunks:
                self.epoch, self.pos, self._perm = self.epoch + 1, 0, None
            take = min(batch_size - len(idx), self.n_chunks - self.pos)
            idx.extend(self._order()[self.pos : self.pos + take].tolist())
            self.pos += take
        L = self.seq_len
        x = np.stack([self.data[i * L : i * L + L] for i in idx]).astype(np.int64)
        y = np.stack([self.data[i * L + 1 : i * L + L + 1] for i in idx]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def set_state(self, st):
        self.epoch, self.pos, self._perm = st["epoch"], st["pos"], None


# ----------------------------------------------------------------------------- utils
def lr_at(step, cfg):
    warm, total = cfg["warmup_steps"], cfg["total_steps"]
    peak, floor = cfg["lr"], cfg["min_lr"]
    if step < warm:
        return peak * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def build_optimizer(model, tcfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": tcfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=tcfg["lr"], betas=tuple(tcfg["betas"]), eps=1e-8, fused=fused)


def save_checkpoint(out_dir, step, model, opt, scaler, data_gen, tokens_seen, cfg, mirror_dir=None, keep_last=2, sampler=None):
    ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
    tmp = Path(out_dir) / f".tmp-checkpoint-{step}"
    tmp.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": raw.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "data_gen": data_gen.get_state(),
        "sampler": sampler.state() if hasattr(sampler, "state") else None,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": cfg,
    }
    torch.save(state, tmp / "state.pt")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp.rename(ckpt_dir)  # atomic: a half-written checkpoint never looks complete
    (Path(out_dir) / "latest").write_text(ckpt_dir.name)

    # prune old checkpoints locally
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in ckpts[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)

    if mirror_dir:
        m = Path(mirror_dir)
        m.mkdir(parents=True, exist_ok=True)
        dst = m / ckpt_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(ckpt_dir, dst)
        for f in ("config.yaml", "manifest.json", "metrics.jsonl"):
            if (Path(out_dir) / f).exists():
                shutil.copy2(Path(out_dir) / f, m / f)
        (m / "latest").write_text(ckpt_dir.name)
        mck = sorted(m.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for old in mck[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)
    return ckpt_dir


def resolve_resume(path):
    p = Path(path)
    if (p / "latest").exists():  # a run directory was given
        p = p / (p / "latest").read_text().strip()
    return p / "state.pt"


@torch.no_grad()
def evaluate(model, val, tcfg, device, amp_dtype):
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # identical val batches every time
    losses = []
    for _ in range(tcfg["val_batches"]):
        x, y = val.batch(tcfg["micro_batch_size"], gen)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--resume", default=None, help="checkpoint dir or run dir (uses 'latest')")
    ap.add_argument("--max_steps", type=int, default=None, help="stop at this global step")
    ap.add_argument("--max_minutes", type=float, default=None, help="wall-clock stop, then checkpoint")
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--mirror_dir", default=None, help="persistent copy, e.g. Google Drive")
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.resume:
        state = torch.load(resolve_resume(args.resume), map_location="cpu", weights_only=False)
        cfg = state["config"]  # the run's frozen config wins; only stop/IO flags may change
    else:
        state = None
    t = cfg["training"]
    for k in ("output_dir", "data_dir"):
        if getattr(args, k):
            cfg["io"][k] = getattr(args, k)
    if args.save_every:
        t["checkpoint_every_steps"] = args.save_every
    if args.log_every:
        t["log_every_steps"] = args.log_every
    max_steps = args.max_steps if args.max_steps is not None else t["total_steps"]

    out_dir = Path(cfg["io"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # new Colab/Kaggle session: bring the run history back from the persistent copy
        src = resolve_resume(args.resume).parents[1]
        for f in ("manifest.json", "metrics.jsonl"):
            if (src / f).exists() and not (out_dir / f).exists():
                shutil.copy2(src / f, out_dir / f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and t["precision"] == "fp16"
    amp_dtype = torch.float16 if use_amp else None

    seed_everything(t["seed"])
    mcfg = ModelConfig.from_dict(cfg["model"])
    model = DecoderLM(mcfg).to(device)
    opt = build_optimizer(model, t)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_gen = torch.Generator().manual_seed(t["seed"])
    step, tokens_seen = 0, 0
    sampler_state = None

    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        data_gen.set_state(state["data_gen"])
        sampler_state = state.get("sampler")
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        step, tokens_seen = state["step"], state["tokens_seen"]
        print(f"[resume] step={step} tokens_seen={tokens_seen:,}")
        del state

    # every run directory holds its full resolved config + manifest (handbook section 7)
    yaml.safe_dump(cfg, open(out_dir / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    manifest_path = out_dir / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {
        "run_id": out_dir.name,
        "git_commit": git_commit(),
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": "fp16_amp" if use_amp else "fp32",
        "n_params": model.num_params(),
        "sessions": [],
    }
    manifest["sessions"].append({"start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "start_step": step, "hardware": manifest["hardware"]})
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=2)

    if t.get("sampler", "chunks") == "chunks":
        train = ChunkStream(Path(cfg["io"]["data_dir"]) / "train.bin", mcfg.max_seq_len, t["seed"], t.get("data_in_memory", True))
        if sampler_state:
            train.set_state(sampler_state)
        print(f"data: {train.n_chunks:,} chunks of {mcfg.max_seq_len} = {train.n_chunks * mcfg.max_seq_len:,} tokens per epoch "
              f"(epoch {train.epoch}, position {train.pos:,})")
    else:
        train = TokenStream(Path(cfg["io"]["data_dir"]) / "train.bin", mcfg.max_seq_len, t.get("data_in_memory", True))
    val = TokenStream(Path(cfg["io"]["data_dir"]) / "val.bin", mcfg.max_seq_len)

    if args.compile:
        model = torch.compile(model)
    print(f"params={manifest['n_params']:,} device={device} amp={use_amp} start_step={step} stop_step={max_steps}")

    mb, accum = t["micro_batch_size"], t["grad_accum_steps"]
    tokens_per_step = mb * accum * mcfg.max_seq_len
    log_f = open(out_dir / "metrics.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    t_last, data_time = time.time(), 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    stop_reason = "max_steps"

    while step < max_steps:
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(accum):
            td = time.time()
            x, y = train.batch(mb, data_gen)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            data_time += time.time() - td
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}. Stop and inspect (handbook s.14).")
            scaler.scale(loss / accum).backward()
            loss_acc += loss.item() / accum
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step

        if step % t["log_every_steps"] == 0 or step == max_steps:
            dt = time.time() - t_last
            rec = {
                "step": step, "loss": round(loss_acc, 4), "lr": lr, "grad_norm": round(gnorm, 3),
                "tokens_seen": tokens_seen,
                "tok_per_s": round(tokens_per_step * t["log_every_steps"] / max(dt, 1e-9)),
                "data_frac": round(data_time / max(dt, 1e-9), 3),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0,
                "loss_scale": scaler.get_scale() if use_amp else 1.0,
            }
            planned = t["total_steps"] * tokens_per_step
            print(f"step {step:>6}/{t['total_steps']} | loss {rec['loss']:.4f} | lr {lr:.2e} | grad {rec['grad_norm']:.3f} | "
                  f"{rec['tok_per_s']:>7,} tok/s | "
                  f"tokens {tokens_seen / 1e9:.2f}B/{planned / 1e9:.2f}B ({tokens_seen / planned:.0%})", flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()  # data_frac, peak_mem_gb and loss_scale are kept here
            t_last, data_time = time.time(), 0.0

        if step % t["validate_every_steps"] == 0:
            vl = evaluate(model, val, t, device, amp_dtype)
            rec = {"step": step, "val_loss": round(vl, 4), "val_ppl": round(math.exp(min(vl, 20)), 2)}
            print(f"step {step:>6} | VALIDATION loss {rec['val_loss']:.4f} | perplexity {rec['val_ppl']:,.1f}", flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        out_of_time = args.max_minutes is not None and (time.time() - t_start) / 60 > args.max_minutes
        if step % t["checkpoint_every_steps"] == 0 or step == max_steps or out_of_time:
            tc = time.time()
            p = save_checkpoint(out_dir, step, model, opt, scaler, data_gen, tokens_seen, cfg, args.mirror_dir, sampler=train)
            print(f"[ckpt] {p} ({time.time() - tc:.1f}s)", flush=True)
        if out_of_time:
            stop_reason = "max_minutes"
            break

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest["sessions"][-1].update({"end_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "end_step": step, "tokens_seen": tokens_seen, "stop_reason": stop_reason})
    manifest["status"] = "completed" if step >= t["total_steps"] else "partial"
    json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=2)
    if args.mirror_dir:
        shutil.copy2(manifest_path, Path(args.mirror_dir) / "manifest.json")
    print(f"[done] step={step} tokens={tokens_seen:,} reason={stop_reason}")
    return step


if __name__ == "__main__":
    main()
