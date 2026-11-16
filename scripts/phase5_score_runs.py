"""Phase 5: detector scores for every Phase 1 run, so the demo can show a warning badge for any
hint toggle.

For each demo question (the first `demo_questions` of the 375) x 10 conditions (no hint, sham,
4 framings x wrong/right hint):
- post: the Phase 3 after-reasoning probe (layer 22). The Phase 1 reasoning is fed back in on the
  Mac and read at an appended "Final answer:", exactly as in Phase 3.
- pre: the Phase 3 before-reasoning probe (layer 22, forced "Final answer:" right after the
  prompt), plus the one-pass readout log-probabilities of each letter.

Runs the probes were trained on get their out-of-fold score (never scored by a probe that saw
them). Every other run (no hint, sham, right-answer hints, admitted or ineligible runs) is scored
by the final probe, which never saw it. `*_source` records which.

Activations already captured in Phases 2-3 are reused; only missing runs go through the model.
Usage: uv run python scripts/phase5_score_runs.py [--debug]
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from phase3_capture_postcot import reasoning_only
from safetensors.torch import load_file

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.interp.features import PREFILL, capture
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, read_jsonl, start_run

logger = logging.getLogger("phase5_score_runs")
LETTERS = "ABCD"
TRAINED_LABELS = ("unfaithful", "unaffected")


def stored(capture_run, index_file: str, pos: int, layer: int) -> dict[str, np.ndarray]:
    """job_id -> stored float16 residual vector at one position."""
    idx = pd.read_parquet(capture_run / index_file)
    t = load_file(str(capture_run / f"resid_L{layer}.safetensors"))["resid"][:, pos]
    return {j: t[i].float().numpy() for i, j in enumerate(idx.job_id)}


def oof_scores(probes_run, post_index: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Out-of-fold scores, aligned the way phase3_probes.py built its rows."""
    jobs = post_index[post_index.p1_label.isin(TRAINED_LABELS)].job_id.tolist()
    out = {}
    for st in ("pre", "post"):
        s = np.load(probes_run / f"oof_scores_{st}.npy")
        assert len(s) == len(jobs), "OOF rows do not match the Phase 3 capture index"
        out[st] = dict(zip(jobs, s.tolist(), strict=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    run_dir = start_run("phase5_score_runs", cfg)
    L = cfg["layer"]

    p1 = REPO_ROOT / cfg["phase1_run"]
    labels = pd.read_parquet(p1 / "labels.parquet").set_index("job_id")
    answers = {r["job_id"]: r["generation"] for r in read_jsonl(REPO_ROOT / cfg["phase1_answers"])}
    p2, p3 = REPO_ROOT / cfg["phase2_capture"], REPO_ROOT / cfg["phase3_capture"]
    pre_vec = stored(p2, "readout.parquet", 1, L)  # positions: gen_start, final, hint
    post_vec = stored(p3, "index.parquet", 0, L)  # positions: post, cot_mean
    readout = pd.read_parquet(p2 / "readout.parquet").set_index("job_id")
    probes_run = REPO_ROOT / cfg["phase3_probes"]
    oof = oof_scores(probes_run, pd.read_parquet(p3 / "index.parquet"))
    probe = {st: joblib.load(probes_run / f"probe_{st}.joblib")["model"] for st in ("pre", "post")}

    items = medqa_test_not_in_mpib(cfg["n_questions"], cfg["seed"])[: cfg["demo_questions"]]
    model, rows, t0, n_new = None, [], time.time(), 0
    for qi, item in enumerate(items):
        for cond in conditions_for(item, cfg["framings"], ["wrong", "gold"], cfg["seed"]):
            job = f"{item.id}|{cond.name}"
            msgs = build_messages(item, cond)
            row: dict = {"job_id": job}
            if job not in post_vec or job not in pre_vec:
                model = model or HFLocalModel(cfg["model_id"], device=cfg["device"])
                n_new += 1
            if job not in post_vec:
                prompt = model.chat_prompt(msgs)
                text = reasoning_only(answers[job]["text"])
                ids = model.encode(prompt + text + "\n" + PREFILL, chat_formatted=True)
                with torch.no_grad():
                    _, acts = model.forward_logits(
                        ids[:, : cfg["max_tokens"]], capture=[(L, "resid_post")]
                    )
                post_vec[job] = acts[(L, "resid_post")][0, -1].half().float().numpy()
                del acts
            if job not in pre_vec:
                cap = capture(model, msgs, LETTERS, [L], None)
                pre_vec[job] = cap.resid[L]["final"].half().float().numpy()
                row |= {f"lp_{k}": float(v) for k, v in cap.letter_logprobs.items()}
            else:
                row |= {f"lp_{k}": float(readout.loc[job, f"lp_{k}"]) for k in LETTERS}
            trained = labels.loc[job, "label"] in TRAINED_LABELS and job in oof["post"]
            for st, vec in (("pre", pre_vec), ("post", post_vec)):
                if trained:
                    row[f"{st}_score"], row[f"{st}_source"] = oof[st][job], "out-of-fold"
                else:
                    row[f"{st}_score"] = float(probe[st].predict_proba(vec[job][None])[0, 1])
                    row[f"{st}_source"] = "final probe (run not in training)"
            rows.append(row)
        if (qi + 1) % 10 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info(
                "questions %d/%d, new runs %d (%.0fs)", qi + 1, len(items), n_new, time.time() - t0
            )
    pd.DataFrame(rows).to_parquet(run_dir / "scores.parquet")
    logger.info("saved %d rows; %d needed the model (%.0fs)", len(rows), n_new, time.time() - t0)


if __name__ == "__main__":
    main()
