"""Feature-level interventions on the residual stream using an SAE dictionary.

`remove_features` subtracts the chosen features' contributions (activation x decoder row)
from the residual stream and leaves the SAE's reconstruction error and every other feature
untouched, so only those features change. `scale_features` multiplies them instead
(factor 0 = removal, factor > 1 = amplification).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def scale_features(
    sae: Any, features: list[int], factor: float, start_pos: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:
    idx = torch.tensor(features, dtype=torch.long, device=sae.W_dec.device)
    w_dec = sae.W_dec[idx]  # [k, d]

    def hook(x: torch.Tensor) -> torch.Tensor:
        xs = x[:, start_pos:].float().to(sae.W_dec.device)
        acts = sae.encode(xs)[..., idx]  # [b, t, k]
        delta = (factor - 1.0) * (acts @ w_dec)  # removal when factor == 0
        out = x.clone()
        out[:, start_pos:] = (xs + delta).to(x.dtype).to(x.device)
        return out

    return hook


def remove_features(sae: Any, features: list[int], start_pos: int = 0):
    return scale_features(sae, features, 0.0, start_pos)
