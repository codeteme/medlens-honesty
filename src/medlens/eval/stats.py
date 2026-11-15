from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved for small n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa for two binary raters; None if undefined (both raters constant and equal)."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and the same length")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(values: list, stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of `stat(sample)` resampling `values` (e.g. per-item records)."""
    import random

    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = stat([values[rng.randrange(len(values))] for _ in values])
        if s is not None:
            stats.append(s)
    stats.sort()
    if not stats:
        return (float("nan"), float("nan"))
    return (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1])


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Area under the ROC curve = P(score of a positive > score of a negative), ties count 1/2."""
    if not pos or not neg:
        return None
    import numpy as np

    p, n = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(p)].sum()
    return float((r_pos - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
