"""Paired statistics for quantized-vs-full-precision comparisons.

Everything here is numpy/scipy only so it runs on a laptop.

Design: every quantized episode is paired with the full-precision episode that
shares its (suite, task, episode index) and therefore its initial state and
noise seed. All tests operate on the per-pair difference.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

Z_ALPHA_BETA = 1.959964 + 0.841621  # two-sided alpha=0.05, power=0.80


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value. n01/n10 are the discordant pair counts."""
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(n01, n10), n, 0.5).pvalue)


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are passed through)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def paired_continuous(diff, clusters=None, n_boot: int = 5000, seed: int = 0) -> dict:
    """Summarise per-pair differences of a continuous metric.

    diff      : quantized minus full-precision, one value per pair.
    clusters  : optional task id per pair. If given, the bootstrap resamples
                tasks first, then pairs within task (tasks are few, so this CI
                is wider and more honest than the episode-level one).
    """
    d = np.asarray(diff, dtype=float)
    keep = ~np.isnan(d)
    d = d[keep]
    n = len(d)
    out = dict(n=n, mean=np.nan, median=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               dz=np.nan, p_wilcoxon=np.nan, p_t=np.nan, n_for_80pct_power=np.nan)
    if n < 5:
        return out
    rng = np.random.default_rng(seed)
    if clusters is None:
        boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    else:
        c = np.asarray(clusters)[keep]
        groups = [d[c == u] for u in np.unique(c)]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            picked = rng.integers(0, len(groups), size=len(groups))
            vals = [groups[g][rng.integers(0, len(groups[g]), size=len(groups[g]))] for g in picked]
            boots[b] = np.concatenate(vals).mean()
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else 0.0
    nz = d[d != 0]
    out.update(
        mean=float(d.mean()),
        median=float(np.median(d)),
        ci_lo=float(np.percentile(boots, 2.5)),
        ci_hi=float(np.percentile(boots, 97.5)),
        dz=float(dz),
        p_wilcoxon=float(sps.wilcoxon(nz).pvalue) if len(nz) >= 5 else 1.0,
        p_t=float(sps.ttest_1samp(d, 0.0).pvalue) if sd > 0 else 1.0,
        n_for_80pct_power=n_pairs_continuous(dz),
    )
    return out


def paired_binary(a, b) -> dict:
    """a = full-precision flags, b = quantized flags, same pairs. Reports b - a."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = len(a)
    n10 = int(np.sum(a & ~b))  # lost under quantization
    n01 = int(np.sum(~a & b))  # gained under quantization
    return dict(
        n=n,
        rate_fp=float(a.mean()) if n else np.nan,
        rate_q=float(b.mean()) if n else np.nan,
        diff=float(b.mean() - a.mean()) if n else np.nan,
        n_lost=n10,
        n_gained=n01,
        p_mcnemar=mcnemar_exact(n01, n10),
        n_for_80pct_power=n_pairs_mcnemar(n10 / n, n01 / n) if n else np.nan,
    )


def n_pairs_continuous(dz: float) -> float:
    """Pairs needed for 80% power at alpha=0.05 for a paired mean shift of size dz."""
    if dz == 0 or math.isnan(dz):
        return math.inf
    return math.ceil((Z_ALPHA_BETA / abs(dz)) ** 2)


def n_pairs_mcnemar(p10: float, p01: float) -> float:
    """Pairs needed for 80% power, McNemar normal approximation."""
    delta = abs(p10 - p01)
    if delta == 0:
        return math.inf
    return math.ceil(Z_ALPHA_BETA ** 2 * (p10 + p01) / delta ** 2)
