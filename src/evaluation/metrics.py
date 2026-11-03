"""Routing and end-to-end metrics from a prediction table (handbook sections 24-26).

"Call-beneficial" (y = 1) means benefit = reward_call - reward_answer > 0: answering directly would be wrong
and the evidence would fix it. The routing score is call_score = log p(CALL) - log p(ANSWER).
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ranking
def auroc(scores, y):
    """Mann-Whitney AUROC with tie handling; NaN if only one class present."""
    s, y = np.asarray(scores, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y, int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    hits = np.cumsum(y)
    prec = hits / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


# ---------------------------------------------------------------- budgets
def route_top_fraction(scores, frac):
    """Boolean mask: CALL for exactly round(frac*n) prompts with the highest call score (ties broken stably)."""
    s = np.asarray(scores, float)
    k = int(round(frac * len(s)))
    mask = np.zeros(len(s), bool)
    if k > 0:
        mask[np.argsort(-s, kind="mergesort")[:k]] = True
    return mask


def outcome(df, call_mask):
    call_mask = np.asarray(call_mask, bool)
    ra, rc = df.reward_answer.to_numpy(), df.reward_call.to_numpy()
    final = np.where(call_mask, rc, ra)
    answered = ~call_mask
    return {
        "accuracy": float(final.mean()),
        "call_rate": float(call_mask.mean()),
        "harmful_answer_rate": float(((ra == 0) & (rc == 1) & answered).mean()),  # share of all prompts
        "wasteful_call_rate": float(((ra == 1) & call_mask).mean()),
    }


def threshold_for_rate(scores, rate):
    """Numeric threshold whose call rate on these scores is `rate` (chosen on dev, applied unchanged to test)."""
    s = np.sort(np.asarray(scores, float))[::-1]
    k = int(round(rate * len(s)))
    if k <= 0:
        return float(s[0] + 1e-6)
    return float(s[k - 1])


# ---------------------------------------------------------------- calibration and selective answering
def p_call(df):
    return 1.0 / (1.0 + np.exp(-df.call_score.to_numpy()))


def brier(df):
    y = (df.benefit > 0).to_numpy().astype(float)
    return float(((p_call(df) - y) ** 2).mean())


def ece(df, bins=10):
    p, y = p_call(df), (df.benefit > 0).to_numpy().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def aurc(df):
    """Area under the risk-coverage curve for direct answers: answer the most ANSWER-confident first."""
    order = np.argsort(df.call_score.to_numpy(), kind="mergesort")
    err = 1 - df.reward_answer.to_numpy()[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


# ---------------------------------------------------------------- summary
def summarize(df, budgets=(0.05, 0.10, 0.20), dev_thresholds=None):
    y = (df.benefit > 0).astype(int)
    out = {
        "n": int(len(df)),
        "call_beneficial_rate": float(y.mean()),
        "default_call_rate": float((df.predicted_action == "CALL").mean()),
        "auroc": auroc(df.call_score, y),
        "average_precision": average_precision(df.call_score, y),
        "acc_always_answer": float(df.reward_answer.mean()),
        "acc_always_call": float(df.reward_call.mean()),
        "acc_oracle": float(np.maximum(df.reward_answer, df.reward_call).mean()),
        "default_decision": outcome(df, df.predicted_action == "CALL"),
        "brier": brier(df), "ece": ece(df), "aurc": aurc(df),
        "both_fail_rate": float(df.both_fail.mean()),
    }
    for b in budgets:
        out[f"budget_{int(round(b * 100))}"] = outcome(df, route_top_fraction(df.call_score, b))
    if dev_thresholds:
        for b, thr in dev_thresholds.items():
            out[f"frozen_threshold_{int(round(b * 100))}"] = outcome(df, df.call_score.to_numpy() >= thr)
    return out


def bootstrap_ci(df, fn, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(df)
    for _ in range(n_boot):
        v = fn(df.iloc[rng.integers(0, n, n)])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(df_a, df_b, fn, n_boot=2000, seed=0, alpha=0.05):
    """CI for fn(a) - fn(b) resampling the SAME prompts in both tables (handbook section 26)."""
    a = df_a.set_index("prompt_id").sort_index()
    b = df_b.set_index("prompt_id").sort_index()
    assert a.index.equals(b.index), "paired bootstrap needs identical prompt sets"
    a, b = a.reset_index(), b.reset_index()
    rng = np.random.default_rng(seed)
    n, diffs = len(a), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(fn(a.iloc[i]) - fn(b.iloc[i]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(fn(a) - fn(b)), float(lo), float(hi)


def flat(summary):
    """One-level dict for printing / CSV."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out
