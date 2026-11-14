"""Reconstruction-quality metrics for SAEs and transcoders (pure tensor functions).

Plain-English meaning:
- FVU (fraction of variance unexplained): 0 = perfect reconstruction, 1 = no better than
  predicting the average activation. Explained variance = 1 - FVU.
- L0: how many features are active per token (sparsity).
- Delta CE: how much worse the model predicts the next token (in nats) when we swap the
  real activation for the reconstruction. 0 = no harm.
- CE recovered: 1 = splicing in the reconstruction is as good as the real activation;
  0 = as bad as deleting (zeroing) that activation entirely.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained over tokens. x, x_hat: [n_tokens, d]."""
    resid = (x - x_hat).pow(2).sum()
    total = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    return (resid / total).item()


def mean_cosine(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return F.cosine_similarity(x, x_hat, dim=-1).mean().item()


def relative_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean over tokens of ||x - x_hat|| / ||x||."""
    return ((x - x_hat).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).mean().item()


def mean_l0(feature_acts: torch.Tensor) -> float:
    return (feature_acts > 0).float().sum(dim=-1).mean().item()


def next_token_ce(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position cross-entropy for predicting token t+1. logits: [1, T, V] -> [T-1]."""
    return F.cross_entropy(logits[0, :-1].float(), input_ids[0, 1:], reduction="none")


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Per-position KL(p || q) between next-token distributions. [1, T, V] -> [T]."""
    logp = F.log_softmax(logits_p[0].float(), dim=-1)
    logq = F.log_softmax(logits_q[0].float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(dim=-1)


def ce_recovered(ce_clean: float, ce_spliced: float, ce_zero: float) -> float | None:
    """(ce_zero - ce_spliced) / (ce_zero - ce_clean); None if zero-ablation does no harm."""
    denom = ce_zero - ce_clean
    if abs(denom) < 1e-6:
        return None
    return (ce_zero - ce_spliced) / denom
