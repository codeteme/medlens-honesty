"""Phase 1: is the TPU (vLLM) run computing the same model as the Mac (HF transformers)?

Teacher-forced check: feed prompt + an answer written on one device into the reference HF
model on the Mac and ask, at every answer token, whether the reference model's top choice is
the token that was written. Numerical noise between devices produces rare near-ties; an
implementation difference produces frequent mismatches the reference model is confident about.
Mac-written answers are scored the same way as a control.

Usage: uv run python scripts/phase1_device_logit_check.py <mac_run_dir> --n 40
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from dotenv import load_dotenv

from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, read_jsonl


@torch.no_grad()
def score(model: HFLocalModel, messages: list, answer: str) -> dict:
    prompt = model.chat_prompt(messages)
    p_ids = model.encode(prompt, chat_formatted=True)
    a_ids = model.tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = torch.cat([p_ids, a_ids.to(model.device)], dim=1)
    logits, _ = model.forward_logits(ids)
    n_p = p_ids.shape[1]
    lp = torch.log_softmax(logits[0, n_p - 1 : -1].float(), dim=-1)  # predicts answer tokens
    target = ids[0, n_p:]
    top = lp.argmax(-1)
    gap = lp.max(-1).values - lp.gather(1, target[:, None])[:, 0]  # 0 when target is top
    mism = gap > 0
    return {
        "n_tokens": int(target.numel()),
        "match_rate": float((~mism).float().mean()),
        "mismatches": int(mism.sum()),
        "mismatch_gap_over_1nat": int((gap > 1.0).sum()),
        "first_mismatch": int(mism.nonzero()[0, 0]) if mism.any() else None,
        "mean_target_logprob": float(lp.gather(1, target[:, None]).mean()),
        "top_equals_target_first_token": bool(top[0] == target[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mac_run", type=Path)
    parser.add_argument("--tpu", type=Path, default=REPO_ROOT / "cache/tpu/out_stage1.jsonl")
    parser.add_argument("--jobs", type=Path, default=REPO_ROOT / "cache/tpu/jobs_full.jsonl")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    mac = {r["job_id"]: r for r in read_jsonl(args.mac_run / "rollouts.jsonl")}
    tpu = {r["job_id"]: r["generation"] for r in read_jsonl(args.tpu)}
    jobs = {r["job_id"]: r["messages"] for r in read_jsonl(args.jobs)}
    common = sorted(set(mac) & set(tpu))
    picks = random.Random(args.seed).sample(common, min(args.n, len(common)))

    model = HFLocalModel("google/medgemma-4b-it")
    rows = []
    for jid in picks:
        for device, text in (("mac", mac[jid]["text"]), ("tpu", tpu[jid]["text"])):
            rows.append({"job_id": jid, "device": device, **score(model, jobs[jid], text)})

    out = {"n_answers_per_device": len(picks)}
    for device in ("mac", "tpu"):
        rs = [r for r in rows if r["device"] == device]
        toks = sum(r["n_tokens"] for r in rs)
        out[device] = {
            "tokens": toks,
            "token_match_rate": sum(r["match_rate"] * r["n_tokens"] for r in rs) / toks,
            "mismatches": sum(r["mismatches"] for r in rs),
            "confident_mismatches_gap_over_1nat": sum(r["mismatch_gap_over_1nat"] for r in rs),
            "answers_with_any_mismatch": sum(r["mismatches"] > 0 for r in rs),
            "first_token_matches": sum(r["top_equals_target_first_token"] for r in rs),
        }
    path = args.mac_run / "device_logit_check.json"
    path.write_text(json.dumps({"summary": out, "rows": rows}, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
