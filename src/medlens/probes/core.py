"""Linear probes with question-grouped cross-validation, plus metrics.

Every split keeps all runs of a question in the same fold, so no question appears in both
training and testing. The regularization strength is chosen inside each training fold
(inner grouped CV), so the test fold never influences it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from medlens.eval.stats import auroc


def make_probe(C: float, seed: int = 0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed),
    )


def _choose_C(X, y, groups, C_grid, seed, n_inner=3) -> float:
    if len(set(groups)) < n_inner or len(set(y)) < 2:
        return C_grid[len(C_grid) // 2]
    scores = []
    for C in C_grid:
        preds = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=n_inner).split(X, y, groups):
            if len(set(y[tr])) < 2:
                preds[te] = 0.5
                continue
            preds[te] = make_probe(C, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        scores.append(auroc(preds[y == 1].tolist(), preds[y == 0].tolist()) or 0.5)
    return C_grid[int(np.argmax(scores))]


@dataclass
class CVResult:
    scores: np.ndarray  # out-of-fold P(positive)
    fold: np.ndarray
    chosen_C: list[float]
    extra_scores: np.ndarray | None = None  # out-of-fold scores for extra rows (e.g. no-hint)


def grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_grid: list[float],
    n_folds: int = 5,
    seed: int = 0,
    X_extra: np.ndarray | None = None,
    groups_extra: np.ndarray | None = None,
) -> CVResult:
    """Out-of-fold probabilities. Extra rows (e.g. no-hint prompts) are scored by the model
    of the fold their question belongs to, so they are never scored by a model trained on
    their own question."""
    scores, fold = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    extra = None if X_extra is None else np.zeros(len(X_extra))
    chosen = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        C = _choose_C(X[tr], y[tr], groups[tr], C_grid, seed)
        chosen.append(C)
        model = make_probe(C, seed).fit(X[tr], y[tr])
        scores[te], fold[te] = model.predict_proba(X[te])[:, 1], k
        if X_extra is not None:
            test_groups = set(groups[te])
            mask = np.array([g in test_groups for g in groups_extra])
            if mask.any():
                extra[mask] = model.predict_proba(X_extra[mask])[:, 1]
    return CVResult(scores, fold, chosen, extra)


def train_test_split_predict(X_tr, y_tr, g_tr, X_te, C_grid, seed=0) -> np.ndarray:
    """Fit on one set (C chosen by inner grouped CV), predict another (e.g. a held-out hint)."""
    C = _choose_C(X_tr, y_tr, g_tr, C_grid, seed)
    return make_probe(C, seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


# ---- metrics ----------------------------------------------------------------------------
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def cluster_bootstrap_auroc(
    p: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% CI resampling whole questions (runs of one question are not independent)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, size=len(uniq))])
        a = auroc(p[pick][y[pick] == 1].tolist(), p[pick][y[pick] == 0].tolist())
        if a is not None:
            vals.append(a)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def summarize(p: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 1000) -> dict:
    a = auroc(p[y == 1].tolist(), p[y == 0].tolist())
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": a,
        "auroc_ci": list(cluster_bootstrap_auroc(p, y, groups, n_boot)),
        "brier": brier(p, y),
        "ece": ece(p, y),
    }
