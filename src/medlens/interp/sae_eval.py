"""Load Gemma Scope 2 SAEs/transcoders via SAELens and measure how well they fit a model.

SAELens release names (verified in sae_lens/pretrained_saes.yaml, v6.55.5):
  gemma-scope-2-{4b,27b}-{it,pt}-res           residual-stream SAEs (subset layers)
  gemma-scope-2-{4b,27b}-{it,pt}-transcoders   transcoders; sae_id starts with "transcoder/"
Subset layers: 4B -> 9, 17, 22, 29.  27B -> 16, 31, 40, 53.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class SAESpec:
    name: str
    layer: int
    sae: Any

    @property
    def sites(self) -> tuple[str, str]:
        """(input site, output site) in the model."""
        if is_transcoder(self.sae):
            return ("mlp_in", "mlp_out")
        return ("resid_post", "resid_post")


@torch.no_grad()
def evaluate_text(
    model: HFLocalModel, specs: list[SAESpec], input_ids: torch.Tensor, skip_first: int = 1
) -> list[dict[str, Any]]:
    """Reconstruction + splice metrics for one tokenized text, for several SAEs at once.

    One clean pass and one zero-ablation pass per output site are shared across SAEs.
    The first `skip_first` positions (<bos>, which has an outsized activation and is
    excluded from SAE training) are never modified and are excluded from all metrics.
    Predictions at position p depend only on positions <= p, so predictions from
    position `skip_first` onward are exactly the ones a splice can affect.
    """

    def ce(logits: torch.Tensor) -> float:
        return metrics.next_token_ce(logits, input_ids)[skip_first:].mean().item()

    def zero(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, skip_first:] = 0
        return out

    capture = sorted({(s.layer, site) for s in specs for site in s.sites})
    clean_logits, acts = model.forward_logits(input_ids, capture=capture)
    ce_clean = ce(clean_logits)

    ce_zero: dict[tuple[int, str], float] = {}
    for key in sorted({(s.layer, s.sites[1]) for s in specs}):
        zero_logits, _ = model.forward_logits(input_ids, replace={key: zero})
        ce_zero[key] = ce(zero_logits)
        del zero_logits

    results = []
    for spec in specs:
        in_site, out_site = spec.sites
        sae = spec.sae
        sae_device = sae.W_enc.device
        x_in = acts[(spec.layer, in_site)][0, skip_first:].to(sae_device)
        target = acts[(spec.layer, out_site)][0, skip_first:].to(sae_device)
        feats = sae.encode(x_in)
        recon = sae(x_in)

        stash: dict[str, torch.Tensor] = {}

        def keep_input(x: torch.Tensor, stash: dict = stash) -> torch.Tensor:
            stash["x"] = x
            return x

        def splice(
            x: torch.Tensor,
            sae: Any = sae,
            sae_device: Any = sae_device,
            stash: dict = stash,
            transcoder: bool = in_site != out_site,
        ) -> torch.Tensor:
            source = stash["x"] if transcoder else x
            out = x.clone()
            out[:, skip_first:] = sae(source[:, skip_first:].float().to(sae_device)).to(x.device)
            return out

        replace = {(spec.layer, out_site): splice}
        if in_site != out_site:
            replace[(spec.layer, in_site)] = keep_input
        spliced_logits, _ = model.forward_logits(input_ids, replace=replace)
        ce_spliced = ce(spliced_logits)
        zero_ce = ce_zero[(spec.layer, out_site)]
        kl = metrics.kl_divergence(clean_logits, spliced_logits)[skip_first:].mean().item()
        del spliced_logits

        results.append(
            {
                "sae": spec.name,
                "layer": spec.layer,
                "n_tokens": int(x_in.shape[0]),
                "fvu": metrics.fvu(target, recon),
                "cosine": metrics.mean_cosine(target, recon),
                "rel_error": metrics.relative_error(target, recon),
                "l0": metrics.mean_l0(feats),
                "ce_clean": ce_clean,
                "ce_spliced": ce_spliced,
                "ce_zero_ablation": zero_ce,
                "delta_ce": ce_spliced - ce_clean,
                "kl": kl,
                "ce_recovered": metrics.ce_recovered(ce_clean, ce_spliced, zero_ce),
            }
        )
    return results


def evaluate_on_ids(
    model: HFLocalModel, sae: Any, layer: int, input_ids: torch.Tensor, skip_first: int = 1
) -> dict[str, Any]:
    return evaluate_text(model, [SAESpec("sae", layer, sae)], input_ids, skip_first)[0]
