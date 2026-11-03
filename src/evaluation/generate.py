"""Batched greedy decoding and next-token scoring for many short prompts.

The model has no padding mask, so prompts are grouped by exact token length and each group is run as one
batch. Generation stops per row at a stop token; at most max_new tokens are produced.
"""
from collections import defaultdict

import torch


def _groups(prompts, batch_size):
    by_len = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    for _, idx in sorted(by_len.items()):
        for k in range(0, len(idx), batch_size):
            yield idx[k : k + batch_size]


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def greedy(model, prompts, max_new=12, stop_ids=(), batch_size=256):
    """prompts: list of token-id lists. Returns a list of generated id lists (stop token excluded)."""
    model.eval()
    dev = next(model.parameters()).device
    stop_set = set(stop_ids)
    stop = torch.tensor(sorted(stop_set), device=dev) if stop_set else None
    out = [None] * len(prompts)
    for idx in _groups(prompts, batch_size):
        x = torch.tensor([prompts[i] for i in idx], device=dev)
        done = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        for _ in range(max_new):
            with _autocast(dev):
                logits, _ = model(x[:, -model.cfg.max_seq_len:])
            nxt = logits[:, -1].float().argmax(-1)
            x = torch.cat([x, nxt[:, None]], 1)   # finished rows keep decoding; their extra tokens are cut below
            if stop is not None:
                done |= torch.isin(nxt, stop)
                if bool(done.all()):
                    break
        L = len(prompts[idx[0]])
        for row, i in enumerate(idx):
            ids = x[row, L:].tolist()
            cut = next((k for k, t in enumerate(ids) if t in stop_set), len(ids))
            out[i] = ids[:cut]
    return out


@torch.no_grad()
def next_token_logprobs(model, prompts, token_ids, batch_size=256):
    """Log-probability (full-vocabulary softmax) of each token in token_ids at the next position.

    Returns a list of lists: out[i][j] = log p(token_ids[j] | prompts[i]).
    """
    model.eval()
    dev = next(model.parameters()).device
    out = [None] * len(prompts)
    for idx in _groups(prompts, batch_size):
        x = torch.tensor([prompts[i] for i in idx], device=dev)
        with _autocast(dev):
            logits, _ = model(x)
        lp = torch.log_softmax(logits[:, -1].float(), -1)[:, token_ids]
        for row, i in enumerate(idx):
            out[i] = lp[row].tolist()
    return out


@torch.no_grad()
def sequence_logprob(model, prefixes, continuations, batch_size=256):
    """Mean per-token log-probability of each continuation given its prefix (teacher forced)."""
    model.eval()
    dev = next(model.parameters()).device
    full = [p + c for p, c in zip(prefixes, continuations)]
    out = [None] * len(full)
    by_key = defaultdict(list)
    for i, (p, c) in enumerate(zip(prefixes, continuations)):
        by_key[(len(p), len(c))].append(i)
    for (lp_len, lc), idx in by_key.items():
        for k in range(0, len(idx), batch_size):
            chunk = idx[k : k + batch_size]
            x = torch.tensor([full[i] for i in chunk], device=dev)
            with _autocast(dev):
                logits, _ = model(x)
            lps = torch.log_softmax(logits.float(), -1)
            pos = torch.arange(lp_len - 1, lp_len + lc - 1, device=dev)
            tgt = x[:, lp_len : lp_len + lc]
            vals = lps[:, pos, :].gather(-1, tgt[..., None]).squeeze(-1).mean(-1)
            for row, i in enumerate(chunk):
                out[i] = vals[row].item()
    return out
