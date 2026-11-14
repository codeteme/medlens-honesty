import math

import pytest
import torch

from medlens.interp import metrics


def test_fvu_perfect_and_mean_prediction():
    x = torch.randn(50, 8)
    assert metrics.fvu(x, x) == pytest.approx(0.0, abs=1e-7)
    mean_pred = x.mean(dim=0, keepdim=True).expand_as(x)
    assert metrics.fvu(x, mean_pred) == pytest.approx(1.0, rel=1e-5)


def test_cosine_and_relative_error():
    x = torch.randn(10, 4)
    assert metrics.mean_cosine(x, 2 * x) == pytest.approx(1.0, abs=1e-6)
    assert metrics.relative_error(x, 2 * x) == pytest.approx(1.0, rel=1e-5)


def test_mean_l0():
    acts = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]])
    assert metrics.mean_l0(acts) == 1.5


def test_next_token_ce_uniform_logits():
    vocab = 7
    logits = torch.zeros(1, 5, vocab)
    ids = torch.randint(0, vocab, (1, 5))
    ce = metrics.next_token_ce(logits, ids)
    assert ce.shape == (4,)
    assert torch.allclose(ce, torch.full((4,), math.log(vocab)))


def test_kl_zero_for_identical_and_positive_otherwise():
    p = torch.randn(1, 3, 6)
    assert torch.allclose(metrics.kl_divergence(p, p), torch.zeros(3), atol=1e-6)
    assert (metrics.kl_divergence(p, p + torch.randn(1, 3, 6)) > 0).all()


def test_ce_recovered():
    assert metrics.ce_recovered(2.0, 2.0, 6.0) == 1.0
    assert metrics.ce_recovered(2.0, 6.0, 6.0) == 0.0
    assert metrics.ce_recovered(2.0, 3.0, 6.0) == 0.75
    assert metrics.ce_recovered(2.0, 3.0, 2.0) is None
