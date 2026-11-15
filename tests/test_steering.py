import numpy as np
import torch

from medlens.interp.steering import (
    ablate_direction,
    add_direction,
    compose,
    mean_diff_direction,
    random_direction,
)


def test_mean_diff_direction_is_unit_and_points_to_positive_class():
    rng = np.random.default_rng(0)
    neg = rng.normal(size=(50, 8))
    pos = rng.normal(size=(50, 8))
    pos[:, 3] += 5.0
    d, gap = mean_diff_direction(pos, neg)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.argmax(np.abs(d)) == 3 and d[3] > 0
    assert 4.0 < gap < 6.0


def test_ablation_removes_only_the_direction():
    d = np.array([1.0, 0.0, 0.0, 0.0])
    x = torch.tensor([[[3.0, 1.0, -2.0, 0.5], [-1.0, 4.0, 0.0, 0.0]]], dtype=torch.bfloat16)
    y = ablate_direction(d)(x)
    assert y.dtype == x.dtype and y.shape == x.shape
    assert torch.allclose(y[..., 0].float(), torch.zeros(1, 2))
    assert torch.equal(y[..., 1:], x[..., 1:])


def test_add_direction_shifts_projection_by_amount():
    d = random_direction(16, seed=1)
    x = torch.randn(2, 5, 16)
    y = add_direction(d, -3.0)(x)
    dt = torch.as_tensor(d, dtype=torch.float32)
    assert torch.allclose((y - x) @ dt, torch.full((2, 5), -3.0), atol=1e-5)


def test_compose_applies_in_order():
    d = np.array([0.0, 1.0])
    x = torch.tensor([[[1.0, 2.0]]])
    y = compose(ablate_direction(d), add_direction(d, 5.0))(x)
    assert torch.allclose(y, torch.tensor([[[1.0, 5.0]]]))
