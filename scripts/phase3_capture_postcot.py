"""Phase 3: capture MedGemma's internal state AFTER its own step-by-step reasoning.

For each prompt (none, sham, and the wrong-answer hints), the reasoning written in Phase 1
(TPU run) is fed back in (teacher forcing) on the Mac, the closing "Final answer" line is cut,
and "Final answer:" is appended. We store residual activations and SAE features at:
- "post":     the ":" of that final "Final answer:" (the state just before the answer letter)
- "cot_mean": the mean over the reasoning tokens

Outputs mirror phase2_capture: index.parquet, resid_L*.safetensors, features_L*.safetensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time

import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import save_file

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.interp.features import PREFILL
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, read_jsonl, start_run

logger = logging.getLogger("phase3_capture_postcot")
POSITIONS = ("post", "cot_mean")
_FINAL = re.compile(r"final\s+answer", re.I)


def reasoning_only(text: str) -> str:
    """Cut at the last 'Final answer' so the answer letter itself is not fed back in."""
    matches = list(_FINAL.finditer(text or ""))
    return (text[: matches[-1].start()] if matches else text).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--labels", nargs="+", help="override postcot.keep_labels")
    parser.add_argument("--hinted-only", action="store_true", help="skip no-hint prompts")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    ccfg = cfg["postcot"]
    if args.labels:
        ccfg["keep_labels"] = args.labels
    ccfg["hinted_only"] = args.hinted_only
    run_dir = start_run("phase3_capture_postcot", cfg)

    answers = {r["job_id"]: r["generation"] for r in read_jsonl(REPO_ROOT / ccfg["answers"])}
    labels = pd.read_parquet(REPO_ROOT / ccfg["phase1_labels"]).set_index("job_id")
    items = medqa_test_not_in_mpib(ccfg["n_questions"], cfg["seed"])
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    layers = ccfg["layers"]
    saes = {
        L: load_sae(ccfg["sae_release"], ccfg["sae_id"].format(layer=L), model.device)
        for L in layers
    }

    rows, feats, resids = [], {L: [] for L in layers}, {L: [] for L in layers}
    t0 = time.time()
    for qi, item in enumerate(items):
        for cond in conditions_for(item, ccfg["framings"], ["wrong"], cfg["seed"], sham=True):
            job_id = f"{item.id}|{cond.name}"
            lab = labels.loc[job_id]
            # Keep what the probes need: no-hint prompts (false-alarm check) and hinted runs
            # labeled unfaithful or unaffected. Sham and faithful runs are skipped.
            if cond.name == "sham" or (cond.is_hint and lab["label"] not in ccfg["keep_labels"]):
                continue
            if ccfg["hinted_only"] and not cond.is_hint:
                continue
            gen = answers[job_id]
            prompt = model.chat_prompt(build_messages(item, cond))
            reasoning = reasoning_only(gen["text"])
            n_prompt = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            ids = model.encode(prompt + reasoning + "\n" + PREFILL, chat_formatted=True)
            ids = ids[:, : ccfg["max_tokens"]]
            with torch.no_grad():
                _, acts = model.forward_logits(ids, capture=[(L, "resid_post") for L in layers])
            for L in layers:
                a = acts[(L, "resid_post")][0]
                cot = a[n_prompt:-3] if a.shape[0] - 3 > n_prompt else a[-1:]
                stack = torch.stack([a[-1], cot.mean(0)])
                resids[L].append(stack.to(torch.bfloat16))  # float16 overflows at L29
                feats[L].append(saes[L].encode(stack.to(model.device)).half().cpu())
            rows.append(
                {
                    "job_id": job_id,
                    "item_id": item.id,
                    "condition": cond.name,
                    "framing": cond.framing,
                    "target": cond.target,
                    "gold": item.answer,
                    "p1_pred": lab["pred"],
                    "p1_eligible": bool(lab["eligible"]),
                    "p1_followed": lab["followed"],
                    "p1_label": lab["label"],
                    "n_tokens": int(ids.shape[1]),
                    "truncated": ids.shape[1] >= ccfg["max_tokens"],
                    "reasoning": reasoning,
                }
            )
            del acts
        if (qi + 1) % 10 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("questions %d/%d (%.0fs)", qi + 1, len(items), time.time() - t0)

    pd.DataFrame(rows).to_parquet(run_dir / "index.parquet")
    for L in layers:
        save_file(
            {"features": torch.stack(feats[L]).contiguous()},
            str(run_dir / f"features_L{L}.safetensors"),
        )
        save_file(
            {"resid": torch.stack(resids[L]).contiguous()}, str(run_dir / f"resid_L{L}.safetensors")
        )
    (run_dir / "capture_info.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "positions": list(POSITIONS),
                "layers": layers,
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    logger.info("saved %d rows", len(rows))


if __name__ == "__main__":
    main()
