"""Phase 2: do the candidate features cause silent hint-following?

readout mode (one forward pass per prompt):
  Remove (or scale) the top-k candidate features at one layer, at every token, and measure
  - on unfaithful hinted prompts: the change in log-prob of the hinted letter after a prefilled
    "Final answer:", and how often the readout still picks the hinted letter;
  - on no-hint prompts: the change in log-prob of the correct letter (side effects).
  Controls: random feature sets matched on mean activation, same size, same layer.

cot mode (full step-by-step regeneration, slower):
  For a subset of unfaithful cases, regenerate on the Mac with and without the intervention
  and count how often the final answer is still the hinted letter. The run without
  intervention doubles as the Mac reproduction check of the Phase 1 (TPU) label.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.eval.answers import extract_answer
from medlens.eval.stats import bootstrap_ci, mcnemar_exact, wilson_interval
from medlens.interp.features import POSITIONS, capture, prompt_positions
from medlens.interp.interventions import scale_features
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run, write_jsonl

logger = logging.getLogger("phase2_causal")


def pick_features(analyze_run: Path, layer: int, position: str, k: int) -> list[int]:
    """Top-k features from differential_features.csv, or from attribution_features.csv when
    `analyze_run` is a phase2_attr_rank run (same columns; `auroc` holds the score)."""
    f = analyze_run / "differential_features.csv"
    d = pd.read_csv(f if f.exists() else analyze_run / "attribution_features.csv")
    d = d[(d.layer == layer) & (d.position == position) & (d.framings_with_positive_diff == 4)]
    return d.sort_values("auroc", ascending=False).feature.head(k).astype(int).tolist()


def matched_random(
    capture_run: Path,
    layer: int,
    position: str,
    chosen: list[int],
    hinted_rows: list[int],
    seed: int,
) -> list[int]:
    """Random features whose mean activation on hinted prompts is close to each chosen one."""
    F = load_file(str(capture_run / f"features_L{layer}.safetensors"))["features"].float()
    mean = F[hinted_rows, POSITIONS.index(position)].mean(0).numpy()
    rng = random.Random(seed)
    order = np.argsort(mean)
    rank = {int(f): i for i, f in enumerate(order)}
    out = []
    for f in chosen:
        lo, hi = max(0, rank[f] - 200), min(len(order), rank[f] + 200)
        pool = [int(x) for x in order[lo:hi] if int(x) not in chosen and int(x) not in out]
        out.append(rng.choice(pool))
    return out


def jobs_for(items, cfg, rows: pd.DataFrame) -> list[dict]:
    by_id = {i.id: i for i in items}
    out = []
    for r in rows.itertuples():
        item = by_id[r.item_id]
        cond = next(
            c
            for c in conditions_for(
                item, cfg["capture"]["framings"], ["wrong"], cfg["seed"], sham=True
            )
            if c.name == r.condition
        )
        out.append({"job_id": r.job_id, "item": item, "cond": cond})
    return out


def readout_mode(args, cfg, run_dir: Path) -> dict:
    r = pd.read_parquet(args.capture_run / "readout.parquet").reset_index(drop=True)
    unf = r[(r.p1_label == "unfaithful") & r.condition.str.endswith("_wrong")]
    unf = unf.sample(n=min(args.n, len(unf)), random_state=cfg["seed"])
    base = r[r.condition == "none"].sample(
        n=min(args.n, (r.condition == "none").sum()), random_state=cfg["seed"]
    )
    items = medqa_test_not_in_mpib(cfg["capture"]["n_questions"], cfg["seed"])
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    sae = load_sae(
        cfg["capture"]["sae_release"],
        cfg["capture"]["sae_id"].format(layer=args.layer),
        model.device,
    )
    hinted_rows = r[r.condition.str.endswith("_wrong")].index.tolist()

    sets = {}
    for k in args.k:
        chosen = pick_features(args.analyze_run, args.layer, args.position, k)
        sets[f"top{k}"] = chosen
        for d in range(args.random_draws):
            sets[f"random{k}_{d}"] = matched_random(
                args.capture_run, args.layer, args.position, chosen, hinted_rows, cfg["seed"] + d
            )
    logger.info("feature sets: %s", sets)

    def run(jobs, feats):
        out = []
        for j in jobs:
            msgs = build_messages(j["item"], j["cond"])
            hook = None
            if feats:
                pos = prompt_positions(model, msgs, j["cond"].cue_text())[args.region]
                hook = {
                    (args.layer, "resid_post"): scale_features(
                        sae, feats, args.factor, positions=pos
                    )
                }
            cap = capture(
                model,
                msgs,
                "".join(j["item"].options),
                [args.layer],
                j["cond"].cue_text(),
                replace=hook,
            )
            out.append(cap.letter_logprobs)
        return out

    hinted_jobs, base_jobs = jobs_for(items, cfg, unf), jobs_for(items, cfg, base)
    t0 = time.time()
    clean_h, clean_b = run(hinted_jobs, None), run(base_jobs, None)
    results, rows = {}, []
    for name, feats in sets.items():
        ab_h, ab_b = run(hinted_jobs, feats), run(base_jobs, feats)
        d_target = [
            a[j["cond"].target] - c[j["cond"].target]
            for a, c, j in zip(ab_h, clean_h, hinted_jobs, strict=True)
        ]
        d_gold = [
            a[j["item"].answer] - c[j["item"].answer]
            for a, c, j in zip(ab_b, clean_b, base_jobs, strict=True)
        ]
        follow_before = [
            max(c, key=c.get) == j["cond"].target for c, j in zip(clean_h, hinted_jobs, strict=True)
        ]
        follow_after = [
            max(a, key=a.get) == j["cond"].target for a, j in zip(ab_h, hinted_jobs, strict=True)
        ]
        acc_before = [
            max(c, key=c.get) == j["item"].answer for c, j in zip(clean_b, base_jobs, strict=True)
        ]
        acc_after = [
            max(a, key=a.get) == j["item"].answer for a, j in zip(ab_b, base_jobs, strict=True)
        ]
        results[name] = {
            "features": feats,
            "mean_delta_logp_target_hinted": float(np.mean(d_target)),
            "delta_target_ci": list(bootstrap_ci(d_target, lambda s: float(np.mean(s)))),
            "readout_follows_hint_before": float(np.mean(follow_before)),
            "readout_follows_hint_after": float(np.mean(follow_after)),
            "mean_delta_logp_gold_nohint": float(np.mean(d_gold)),
            "nohint_readout_accuracy_before": float(np.mean(acc_before)),
            "nohint_readout_accuracy_after": float(np.mean(acc_after)),
        }
        rows += [
            {"set": name, "job_id": j["job_id"], "d_target": d}
            for j, d in zip(hinted_jobs, d_target, strict=True)
        ]
        logger.info(
            "%s: %s (%.0fs)",
            name,
            {k: v for k, v in results[name].items() if k != "features"},
            time.time() - t0,
        )
    write_jsonl(run_dir / "per_prompt.jsonl", rows)
    return {
        "layer": args.layer,
        "position": args.position,
        "factor": args.factor,
        "n_hinted": len(hinted_jobs),
        "n_nohint": len(base_jobs),
        "sets": results,
    }


def cot_mode(args, cfg, run_dir: Path) -> dict:
    r = pd.read_parquet(args.capture_run / "readout.parquet").reset_index(drop=True)
    unf = r[(r.p1_label == "unfaithful") & r.condition.isin([f"{f}_wrong" for f in args.framings])]
    unf = unf.sample(n=min(args.n, len(unf)), random_state=cfg["seed"])
    items = medqa_test_not_in_mpib(cfg["capture"]["n_questions"], cfg["seed"])
    jobs = jobs_for(items, cfg, unf)
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    sae = load_sae(
        cfg["capture"]["sae_release"],
        cfg["capture"]["sae_id"].format(layer=args.layer),
        model.device,
    )
    feats = pick_features(args.analyze_run, args.layer, args.position, args.k[0])
    msgs = [build_messages(j["item"], j["cond"]) for j in jobs]

    def gen(hook):
        outs = []
        for s in range(0, len(msgs), args.batch_size):
            if hook:
                with model.hooks(replace=hook):
                    outs += model.generate_batch(msgs[s : s + args.batch_size], args.max_new_tokens)
            else:
                outs += model.generate_batch(msgs[s : s + args.batch_size], args.max_new_tokens)
        return outs

    t0 = time.time()
    clean = gen(None)
    logger.info("clean regeneration done (%.0fs)", time.time() - t0)
    ablated = gen({(args.layer, "resid_post"): scale_features(sae, feats, args.factor)})
    logger.info("intervened regeneration done (%.0fs)", time.time() - t0)
    rows = []
    for j, c, a in zip(jobs, clean, ablated, strict=True):
        valid = "".join(j["item"].options)
        pc, pa = extract_answer(c.text, valid), extract_answer(a.text, valid)
        rows.append(
            {
                "job_id": j["job_id"],
                "target": j["cond"].target,
                "gold": j["item"].answer,
                "clean_pred": pc,
                "ablated_pred": pa,
                "clean_text": c.text,
                "ablated_text": a.text,
            }
        )
    write_jsonl(run_dir / "cot_rows.jsonl", rows)
    n = len(rows)
    fc = sum(r_["clean_pred"] == r_["target"] for r_ in rows)
    fa = sum(r_["ablated_pred"] == r_["target"] for r_ in rows)
    b = sum(r_["clean_pred"] == r_["target"] and r_["ablated_pred"] != r_["target"] for r_ in rows)
    c2 = sum(r_["clean_pred"] != r_["target"] and r_["ablated_pred"] == r_["target"] for r_ in rows)
    return {
        "layer": args.layer,
        "features": feats,
        "factor": args.factor,
        "n": n,
        "mac_reproduces_following": fc,
        "mac_follow_rate": fc / n,
        "mac_follow_ci": wilson_interval(fc, n),
        "follow_after_intervention": fa,
        "follow_rate_after": fa / n,
        "follow_after_ci": wilson_interval(fa, n),
        "mcnemar_p": mcnemar_exact(b, c2),
        "ablated_correct": sum(r_["ablated_pred"] == r_["gold"] for r_ in rows),
        "ablated_unparsed": sum(r_["ablated_pred"] is None for r_ in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["readout", "cot"])
    parser.add_argument("capture_run", type=Path)
    parser.add_argument("analyze_run", type=Path)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", default="gen_start", choices=["gen_start", "final", "hint"])
    parser.add_argument(
        "--region",
        default="all",
        choices=["all", "hint", "final"],
        help="token positions where features are scaled (readout mode)",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--factor", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--framings", nargs="+", default=["metadata", "explicit"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    run_dir = start_run(
        f"phase2_causal_{args.mode}",
        {**cfg, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}},
    )
    res = (
        readout_mode(args, cfg, run_dir) if args.mode == "readout" else cot_mode(args, cfg, run_dir)
    )
    (run_dir / "causal_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
