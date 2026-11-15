"""Phase 2: capture readouts and SAE features for the Phase 1 prompts on the Mac (HF).

For each question and condition (none, sham, and the wrong-answer hints), this runs one
forward pass with "Final answer:" prefilled, records the option-letter probabilities, and
stores residual activations and Gemma Scope 2 (pt, 16k) SAE features at three positions.

Outputs (in the run directory):
  readout.parquet                one row per prompt: letter log-probs, Phase 1 label, etc.
  features_L{layer}.safetensors  float16 [n_prompts, 3 positions, 16384] SAE activations
  resid_L{layer}.safetensors     bfloat16 [n_prompts, 3 positions, d_model] residual stream
                                 (older runs: float16; channel 443 at L29 overflows)
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import save_file

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.interp.features import POSITIONS, capture
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run

logger = logging.getLogger("phase2_capture")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    ccfg = cfg["capture"]
    run_dir = start_run("phase2_capture", cfg)

    labels = pd.read_parquet(REPO_ROOT / ccfg["phase1_labels"]).set_index("job_id")
    items = medqa_test_not_in_mpib(ccfg["n_questions"], cfg["seed"])
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    layers = ccfg["layers"]
    saes = {
        layer: load_sae(ccfg["sae_release"], ccfg["sae_id"].format(layer=layer), model.device)
        for layer in layers
    }

    rows = []
    feats = {layer: [] for layer in layers}
    resids = {layer: [] for layer in layers}
    t0 = time.time()
    for qi, item in enumerate(items):
        conds = conditions_for(item, ccfg["framings"], ["wrong"], cfg["seed"], sham=True)
        for cond in conds:
            job_id = f"{item.id}|{cond.name}"
            cap = capture(
                model, build_messages(item, cond), "".join(item.options), layers, cond.cue_text()
            )
            p1 = labels.loc[job_id] if job_id in labels.index else None
            rows.append(
                {
                    "job_id": job_id,
                    "item_id": item.id,
                    "condition": cond.name,
                    "framing": cond.framing,
                    "target": cond.target,
                    "gold": item.answer,
                    **{f"lp_{k}": v for k, v in cap.letter_logprobs.items()},
                    "readout_pred": max(cap.letter_logprobs, key=cap.letter_logprobs.get),
                    "n_tokens": cap.n_tokens,
                    "hint_tokens": cap.hint_tokens,
                    "p1_pred": None if p1 is None else p1["pred"],
                    "p1_eligible": None if p1 is None else bool(p1["eligible"]),
                    "p1_followed": None if p1 is None else p1["followed"],
                    "p1_label": None if p1 is None else p1["label"],
                }
            )
            for layer in layers:
                stack = torch.stack(
                    [
                        cap.resid[layer].get(pos, torch.zeros_like(cap.resid[layer]["final"]))
                        for pos in POSITIONS
                    ]
                )
                resids[layer].append(stack.to(torch.bfloat16))  # float16 overflows at L29
                feats[layer].append(saes[layer].encode(stack.to(model.device)).half().cpu())
        if (qi + 1) % 10 == 0 or qi == len(items) - 1:
            logger.info("questions %d/%d (%.0fs)", qi + 1, len(items), time.time() - t0)

    pd.DataFrame(rows).to_parquet(run_dir / "readout.parquet")
    for layer in layers:
        save_file(
            {"features": torch.stack(feats[layer]).contiguous()},
            str(run_dir / f"features_L{layer}.safetensors"),
        )
        save_file(
            {"resid": torch.stack(resids[layer]).contiguous()},
            str(run_dir / f"resid_L{layer}.safetensors"),
        )
    (run_dir / "capture_info.json").write_text(
        json.dumps(
            {
                "n_prompts": len(rows),
                "positions": list(POSITIONS),
                "layers": layers,
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    logger.info("Saved %d prompts to %s", len(rows), run_dir)


if __name__ == "__main__":
    main()
