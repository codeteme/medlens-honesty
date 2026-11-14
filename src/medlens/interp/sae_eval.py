"""Load Gemma Scope 2 SAEs/transcoders via SAELens and measure how well they fit a model.

SAELens release names (verified in sae_lens/pretrained_saes.yaml, v6.55.5):
  gemma-scope-2-{4b,27b}-{it,pt}-res           residual-stream SAEs (subset layers)
  gemma-scope-2-{4b,27b}-{it,pt}-transcoders   transcoders; sae_id starts with "transcoder/"
Subset layers: 4B -> 9, 17, 22, 29.  27B -> 16, 31, 40, 53.
"""

from __future__ import annotations

from typing import Any

import torch
from sae_lens import SAE
from sae_lens.saes.transcoder import Transcoder

from medlens.interp import metrics
from medlens.models.hf_local import HFLocalModel


def load_sae(release: str, sae_id: str, device: str) -> SAE:
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae.eval()


def is_transcoder(sae: Any) -> bool:
    return isinstance(sae, Transcoder) or getattr(sae, "is_transcoder", False)


@torch.no_grad()
def evaluate_on_ids(
    model: HFLocalModel, sae: Any, layer: int, input_ids: torch.Tensor, skip_first: int = 1
) -> dict[str, Any]:
    """Reconstruction + splice metrics for one tokenized text.

    The first `skip_first` positions (the <bos> token, which has an outsized activation and
    is excluded from SAE training) are left untouched and excluded from all metrics.
    """
    transcoder = is_transcoder(sae)
    in_site, out_site = ("mlp_in", "mlp_out") if transcoder else ("resid_post", "resid_post")
    sae_device = sae.W_enc.device

    def apply(x: torch.Tensor) -> torch.Tensor:
        return sae(x.float().to(sae_device)).to(x.device)

    # 1) Clean pass: capture input and target activations.
    clean_logits, acts = model.forward_logits(
        input_ids, capture=[(layer, in_site), (layer, out_site)]
    )
    x_in = acts[(layer, in_site)][0, skip_first:].to(sae_device)
    target = acts[(layer, out_site)][0, skip_first:].to(sae_device)
    feats = sae.encode(x_in)
    recon = sae(x_in)

    # 2) Spliced pass: replace the output site with the reconstruction (keeping <bos>).
    stash: dict[str, torch.Tensor] = {}

    def keep_input(x: torch.Tensor) -> torch.Tensor:
        stash["x"] = x
        return x

    def splice(x: torch.Tensor) -> torch.Tensor:
        source = stash["x"] if transcoder else x
        out = x.clone()
        out[:, skip_first:] = apply(source[:, skip_first:])
        return out

    def zero(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, skip_first:] = 0
        return out

    replace = {(layer, out_site): splice}
    if transcoder:
        replace[(layer, in_site)] = keep_input
    spliced_logits, _ = model.forward_logits(input_ids, replace=replace)
    zero_logits, _ = model.forward_logits(input_ids, replace={(layer, out_site): zero})

    # Predictions at position p use activations at positions <= p, so predictions from
    # position skip_first onward are the ones affected by the splice.
    def ce(logits: torch.Tensor) -> float:
        return metrics.next_token_ce(logits, input_ids)[skip_first:].mean().item()

    ce_clean, ce_spliced, ce_zero = ce(clean_logits), ce(spliced_logits), ce(zero_logits)
    return {
        "n_tokens": int(x_in.shape[0]),
        "fvu": metrics.fvu(target, recon),
        "cosine": metrics.mean_cosine(target, recon),
        "rel_error": metrics.relative_error(target, recon),
        "l0": metrics.mean_l0(feats),
        "ce_clean": ce_clean,
        "ce_spliced": ce_spliced,
        "ce_zero_ablation": ce_zero,
        "delta_ce": ce_spliced - ce_clean,
        "kl": metrics.kl_divergence(clean_logits, spliced_logits)[skip_first:].mean().item(),
        "ce_recovered": metrics.ce_recovered(ce_clean, ce_spliced, ce_zero),
    }
