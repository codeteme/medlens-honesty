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
    sae: Any,
    features: list[int],
    factor: float,
    start_pos: int = 0,
    positions: list[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Scale the chosen features from `start_pos` onward, or only at `positions` if given."""
    idx = torch.tensor(features, dtype=torch.long, device=sae.W_dec.device)
    w_dec = sae.W_dec[idx]  # [k, d]

    def hook(x: torch.Tensor) -> torch.Tensor:
        sel = positions if positions is not None else list(range(start_pos, x.shape[1]))
        sel = [p for p in sel if p < x.shape[1]]
        if not sel:
            return x
        xs = x[:, sel].float().to(sae.W_dec.device)
        acts = sae.encode(xs)[..., idx]  # [b, t, k]
        delta = (factor - 1.0) * (acts @ w_dec)  # removal when factor == 0
        out = x.clone()
        out[:, sel] = (xs + delta).to(x.dtype).to(x.device)
        return out

    return hook


def remove_features(sae: Any, features: list[int], start_pos: int = 0):
    return scale_features(sae, features, 0.0, start_pos)


def scale_features_per_row(
    sae: Any, features: list[int], factor: float, row_positions: list[list[int]]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Batched variant for left-padded generation: row r is changed only at row_positions[r]
    (already shifted by that row's padding). Only the prompt pass is affected; later decoding
    steps see one new token at a time and are left untouched."""
    idx = torch.tensor(features, dtype=torch.long, device=sae.W_dec.device)
    w_dec = sae.W_dec[idx]
    need = max((max(p) for p in row_positions if p), default=-1)

    def hook(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= need:  # a decoding step, not the prompt pass
            return x
        out = x.clone()
        for r, pos in enumerate(row_positions):
            if not pos:
                continue
            xs = x[r, pos].float().to(sae.W_dec.device)
            acts = sae.encode(xs)[..., idx]
            out[r, pos] = (xs + (factor - 1.0) * (acts @ w_dec)).to(x.dtype).to(x.device)
        return out

    return hook
