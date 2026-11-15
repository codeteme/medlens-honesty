"""Direction-based interventions on the residual stream (Phase 4).

A "direction" is a unit vector in the residual stream at one layer, here the difference
between the mean activation of runs that silently followed a hint and runs that ignored it.
Two ways to use it, both applied at every token (prompt and generated), so they need no
knowledge of where a hint is:
- `ablate_direction`: remove the component along the direction (x - (x.d) d).
- `add_direction`:    shift every token along the direction by a fixed amount (negative
                      amounts push away from hint-following).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

HookFn = Callable[[torch.Tensor], torch.Tensor]


def mean_diff_direction(X_pos: np.ndarray, X_neg: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit vector from the negative-class mean to the positive-class mean, and the gap's
    length (used to express steering strength in natural units)."""
    diff = X_pos.mean(0) - X_neg.mean(0)
    gap = float(np.linalg.norm(diff))
    return diff / gap, gap


def _unit(direction: np.ndarray | torch.Tensor) -> torch.Tensor:
    d = torch.as_tensor(direction, dtype=torch.float32)
    return d / d.norm()


def ablate_direction(direction: np.ndarray | torch.Tensor) -> HookFn:
    d = _unit(direction)

    def hook(x: torch.Tensor) -> torch.Tensor:
        dd = d.to(x.device)
        xf = x.float()
        return (xf - (xf @ dd).unsqueeze(-1) * dd).to(x.dtype)

    return hook


def add_direction(direction: np.ndarray | torch.Tensor, amount: float) -> HookFn:
    d = _unit(direction)

    def hook(x: torch.Tensor) -> torch.Tensor:
        return (x.float() + amount * d.to(x.device)).to(x.dtype)

    return hook


def compose(*fns: HookFn) -> HookFn:
    def hook(x: torch.Tensor) -> torch.Tensor:
        for f in fns:
            x = f(x)
        return x

    return hook


def random_direction(dim: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)
