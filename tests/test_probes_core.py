import numpy as np
import pytest

from medlens.probes.core import brier, cluster_bootstrap_auroc, ece, grouped_cv, summarize


def toy(n_groups=40, per=4, signal=2.0, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, 2, size=len(groups))
    X = rng.normal(size=(len(groups), 6))
    X[:, 0] += signal * y
    return X, y, groups


def test_grouped_cv_never_tests_on_training_questions():
    X, y, g = toy()
    res = grouped_cv(X, y, g, [0.1, 1.0], n_folds=5)
    for k in range(5):
        test_groups = set(g[res.fold == k])
        train_groups = set(g[res.fold != k])
        assert not test_groups & train_groups


def test_grouped_cv_learns_signal_and_not_noise():
    X, y, g = toy(signal=2.0)
    res = grouped_cv(X, y, g, [0.1, 1.0])
    assert summarize(res.scores, y, g, n_boot=100)["auroc"] > 0.85
    Xn, yn, gn = toy(signal=0.0, seed=1)
    resn = grouped_cv(Xn, yn, gn, [0.1, 1.0])
    assert 0.3 < summarize(resn.scores, yn, gn, n_boot=100)["auroc"] < 0.7


def test_extra_rows_scored_by_their_own_fold_model():
    X, y, g = toy()
    Xe = np.random.default_rng(2).normal(size=(40, 6))
    ge = np.arange(40)
    res = grouped_cv(X, y, g, [1.0], X_extra=Xe, groups_extra=ge)
    assert res.extra_scores.shape == (40,) and np.all(
        (res.extra_scores > 0) & (res.extra_scores < 1)
    )


def test_calibration_metrics():
    y = np.array([0, 0, 1, 1])
    assert brier(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
    assert brier(np.full(4, 0.5), y) == 0.25
    assert ece(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
    assert ece(np.full(4, 0.9), y) == pytest.approx(0.4)


def test_cluster_bootstrap_ci_contains_point_estimate():
    X, y, g = toy()
    res = grouped_cv(X, y, g, [1.0])
    lo, hi = cluster_bootstrap_auroc(res.scores, y, g, n_boot=200)
    a = summarize(res.scores, y, g, n_boot=10)["auroc"]
    assert lo <= a <= hi
