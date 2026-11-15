"""Phase 2: follow-ups to the Phase 0 SAE fit check.

1. Noise control: is splicing in the SAE reconstruction more damaging (KL) than adding random
   noise of exactly the same per-token size as the SAE error? Ratio KL(SAE) / KL(noise) is
   comparable across models, unlike raw KL (which also reflects each model's sensitivity).
2. Wider SAEs: 16k vs 65k features at the later layers.
3. Generated text: fit on MedGemma's own step-by-step answers (teacher-forced), not just prompts.

Runs on MedGemma 4B and on plain Gemma 3 4B IT (the control from Phase 0).

Usage: uv run python scripts/phase2_sae_followups.py --answers cache/tpu/out_stage1.jsonl
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from medlens.interp import metrics
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, read_jsonl, start_run

logger = logging.getLogger("phase2_sae_followups")


@torch.no_grad()
def evaluate(model, sae, layer, ids, gen=None) -> dict:
    skip = 1
    clean, acts = model.forward_logits(ids, capture=[(layer, "resid_post")])
    x = acts[(layer, "resid_post")][0, skip:].to(sae.W_enc.device)
    recon = sae(x)
    err_norm = (x - recon).norm(dim=-1, keepdim=True)

    def splice(h):
        out = h.clone()
        out[:, skip:] = sae(h[:, skip:].float().to(sae.W_enc.device)).to(h.dtype).to(h.device)
        return out

    def noise(h):
        out = h.clone()
        z = torch.randn(h[:, skip:].shape, generator=gen, device="cpu").to(h.device).float()
        z = z / z.norm(dim=-1, keepdim=True) * err_norm.to(h.device)
        out[:, skip:] = (h[:, skip:].float() + z).to(h.dtype)
        return out

    s_logits, _ = model.forward_logits(ids, replace={(layer, "resid_post"): splice})
    n_logits, _ = model.forward_logits(ids, replace={(layer, "resid_post"): noise})
    kl_s = metrics.kl_divergence(clean, s_logits)[skip:].mean().item()
    kl_n = metrics.kl_divergence(clean, n_logits)[skip:].mean().item()
    return {
        "fvu": metrics.fvu(x, recon),
        "kl_sae": kl_s,
        "kl_noise": kl_n,
        "kl_ratio": kl_s / kl_n if kl_n > 0 else None,
        "n_tokens": int(x.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=REPO_ROOT / "cache/tpu/out_stage1.jsonl")
    parser.add_argument("--jobs", type=Path, default=REPO_ROOT / "cache/tpu/jobs_full.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 22, 29])
    parser.add_argument("--widths", nargs="+", default=["16k", "65k"])
    parser.add_argument(
        "--models", nargs="+", default=["google/medgemma-4b-it", "google/gemma-3-4b-it"]
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    run_dir = start_run(
        "phase2_sae_followups", {"args": {k: str(v) for k, v in vars(args).items()}, "seed": 0}
    )

    answers = {r["job_id"]: r["generation"]["text"] for r in read_jsonl(args.answers)}
    jobs = [j for j in read_jsonl(args.jobs) if j["job_id"].endswith("|none")]
    jobs = random.Random(0).sample(jobs, args.n)
    rows = []
    for model_id in args.models:
        model = HFLocalModel(model_id)
        texts = []
        for j in jobs:
            prompt = model.chat_prompt(j["messages"])
            texts.append(
                ("prompt", model.encode(prompt, chat_formatted=True)[:, : args.max_tokens])
            )
            full = prompt + answers[j["job_id"]]
            texts.append(
                ("prompt+answer", model.encode(full, chat_formatted=True)[:, : args.max_tokens])
            )
        for layer in args.layers:
            for width in args.widths:
                sae = load_sae(
                    "gemma-scope-2-4b-pt-res",
                    f"layer_{layer}_width_{width}_l0_medium",
                    model.device,
                )
                gen = torch.Generator().manual_seed(layer)
                for kind, ids in texts:
                    rows.append(
                        {
                            "model": model_id,
                            "layer": layer,
                            "width": width,
                            "text": kind,
                            **evaluate(model, sae, layer, ids, gen),
                        }
                    )
                logger.info("%s L%d %s done", model_id, layer, width)
                del sae
                gc.collect()
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_text.csv", index=False)
    summary = (
        df.groupby(["model", "layer", "width", "text"])[["fvu", "kl_sae", "kl_noise", "kl_ratio"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
