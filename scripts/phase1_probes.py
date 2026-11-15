"""Phase 1: position-bias, causal-ablation and format probes on MedGemma 4B.

Generation happens on the TPU with scripts/tpu/vllm_generate.py; this script builds the
prompts and analyses the answers.

  export1   prompts for position bias, ablation baseline, and the three formats
  export2   ablated prompts, built from the ablation-baseline answers (--generations)
  analyze   metrics from all generation files (--generations, repeatable)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from medlens.blackbox.probes import (
    FORMATS,
    POSITION,
    ablation_messages,
    ablation_metrics,
    afolabi_position_ack,
    format_messages,
    mentions_examples,
    option_text,
    parse_ablation,
    parse_json_answer,
    pick_demos,
    position_conditions,
    position_messages,
    redact,
    valid_quotes,
)
from medlens.config import load_config
from medlens.data.loaders import load_medqa, medqa_test_not_in_mpib
from medlens.eval.answers import extract_answer
from medlens.eval.stats import bootstrap_ci, mcnemar_exact, wilson_interval
from medlens.runs import REPO_ROOT, read_jsonl, start_run, write_jsonl

logger = logging.getLogger("phase1_probes")


def items_and_demos(cfg: dict) -> tuple[list, list]:
    pcfg = cfg["probes"]
    items = medqa_test_not_in_mpib(pcfg["n_questions"], cfg["seed"])
    demos = pick_demos(load_medqa("train"), pcfg["n_demos"], cfg["seed"] + 7)
    return items, demos


def export1(cfg: dict, out: Path) -> None:
    pcfg = cfg["probes"]
    items, demos = items_and_demos(cfg)
    jobs = []
    for it in items:
        for cond, test in position_conditions(it, cfg["seed"]).items():
            jobs.append(
                {
                    "job_id": f"pos|{it.id}|{cond}",
                    "max_tokens": pcfg["max_tokens"]["cot"],
                    "messages": position_messages(test, demos, biased=cond != "unbiased"),
                }
            )
        jobs.append(
            {
                "job_id": f"abl0|{it.id}",
                "max_tokens": pcfg["max_tokens"]["ablation"],
                "messages": ablation_messages(it),
            }
        )
        for fmt in FORMATS:
            jobs.append(
                {
                    "job_id": f"fmt|{it.id}|{fmt}",
                    "max_tokens": pcfg["max_tokens"][fmt],
                    "messages": format_messages(it, fmt),
                }
            )
    write_jsonl(out, jobs)
    logger.info("Exported %d stage-1 jobs for %d questions to %s", len(jobs), len(items), out)


def export2(cfg: dict, gens: list[Path], out: Path) -> None:
    pcfg = cfg["probes"]
    items, _ = items_and_demos(cfg)
    by_id = {r["job_id"]: r for g in gens for r in read_jsonl(g)}
    jobs = []
    for it in items:
        base = parse_ablation(by_id[f"abl0|{it.id}"]["generation"]["text"])
        for k, quote in enumerate(valid_quotes(it, base["quotes"])):
            jobs.append(
                {
                    "job_id": f"abl1|{it.id}|{k}",
                    "max_tokens": pcfg["max_tokens"]["ablation"],
                    "messages": ablation_messages(redact(it, quote)),
                }
            )
    write_jsonl(out, jobs)
    logger.info("Exported %d ablated jobs to %s", len(jobs), out)


def ci(k: int, n: int) -> list[float] | None:
    return list(wilson_interval(k, n)) if n else None


def analyze(cfg: dict, gens: list[Path], run_dir: Path) -> dict:
    items, _ = items_and_demos(cfg)
    by_id = {r["job_id"]: r["generation"] for g in gens for r in read_jsonl(g)}
    out: dict = {"n_questions": len(items)}

    # ---- position bias: compare in option-text space, since option order changes ----
    pos_rows = []
    for it in items:
        conds = position_conditions(it, cfg["seed"])
        row = {"item_id": it.id}
        for cond, test in conds.items():
            g = by_id[f"pos|{it.id}|{cond}"]
            pred = extract_answer(g["text"], valid="".join(test.options))
            row[cond] = {
                "pred": pred,
                "pred_text": option_text(test, pred),
                "correct": pred == test.answer,
                "picked_b": pred == POSITION,
                "b_text": test.options[POSITION],
                "unfinished": g["stop_reason"] == "max_tokens",
                "afolabi_ack": afolabi_position_ack(g["text"]),
                "mentions_examples": mentions_examples(g["text"]),
            }
        pos_rows.append(row)
    write_jsonl(run_dir / "position_rows.jsonl", pos_rows)
    pos = {}
    for cond in ("unbiased", "bias_gold", "bias_wrong"):
        rs = [r[cond] for r in pos_rows]
        n = len(rs)
        pos[cond] = {
            "accuracy": sum(r["correct"] for r in rs) / n,
            "accuracy_ci": ci(sum(r["correct"] for r in rs), n),
            "pick_b_rate": sum(r["picked_b"] for r in rs) / n,
            "afolabi_ack_rate": sum(r["afolabi_ack"] for r in rs) / n,
            "mentions_examples_rate": sum(r["mentions_examples"] for r in rs) / n,
            "unfinished": sum(r["unfinished"] for r in rs),
        }
    # Flip toward B under bias_wrong: unbiased answer was NOT the option now placed at B.
    elig = [
        r
        for r in pos_rows
        if r["unbiased"]["pred_text"] is not None
        and r["unbiased"]["pred_text"] != r["bias_wrong"]["b_text"]
        and r["bias_wrong"]["pred"]
    ]
    flips = sum(r["bias_wrong"]["picked_b"] for r in elig)
    pos["bias_wrong_flip_to_b"] = {
        "n_eligible": len(elig),
        "flipped": flips,
        "rate": flips / len(elig) if elig else None,
        "ci": ci(flips, len(elig)),
    }
    b = sum(r["unbiased"]["correct"] and not r["bias_wrong"]["correct"] for r in pos_rows)
    c = sum(not r["unbiased"]["correct"] and r["bias_wrong"]["correct"] for r in pos_rows)
    pos["accuracy_unbiased_vs_bias_wrong_mcnemar_p"] = mcnemar_exact(b, c)
    out["position_bias"] = pos

    # ---- causal ablation ----
    abl_rows = []
    parse_fail = no_quotes = 0
    for it in items:
        base = parse_ablation(by_id[f"abl0|{it.id}"]["text"])
        quotes = valid_quotes(it, base["quotes"])
        if not base["parsed"]:
            parse_fail += 1
        if not quotes:
            no_quotes += 1
        ablated = [
            parse_ablation(by_id[f"abl1|{it.id}|{k}"]["text"])["answer"] for k in range(len(quotes))
        ]
        abl_rows.append(
            {
                "item_id": it.id,
                "gold": it.answer,
                "base": base["answer"],
                "n_steps": base["n_steps"],
                "n_quotes_written": len(base["quotes"]),
                "n_quotes_valid": len(quotes),
                "ablated": [a for a in ablated if a],
            }
        )
    write_jsonl(run_dir / "ablation_rows.jsonl", abl_rows)
    abl = ablation_metrics(abl_rows)
    used = [r for r in abl_rows if r["base"] is not None and r["ablated"]]
    abl["net_flip_ci"] = list(bootstrap_ci(used, lambda s: ablation_metrics(s).get("net_flip")))
    abl["baseline_unparsed"] = parse_fail
    abl["no_valid_quotes"] = no_quotes
    written = sum(r["n_quotes_written"] for r in abl_rows)
    abl["quote_validity"] = (
        sum(r["n_quotes_valid"] for r in abl_rows) / written if written else None
    )
    abl["baseline_accuracy_json"] = sum(r["base"] == r["gold"] for r in abl_rows) / len(abl_rows)
    out["causal_ablation"] = abl

    # ---- formats ----
    fmt_preds = {}
    for fmt in FORMATS:
        preds = []
        for it in items:
            text = by_id[f"fmt|{it.id}|{fmt}"]["text"]
            valid = "".join(it.options)
            p = parse_json_answer(text, valid) if fmt == "json" else extract_answer(text, valid)
            preds.append(p)
        fmt_preds[fmt] = preds
    fmts = {}
    for fmt, preds in fmt_preds.items():
        k = sum(p == it.answer for p, it in zip(preds, items, strict=True))
        fmts[fmt] = {
            "accuracy": k / len(items),
            "accuracy_ci": ci(k, len(items)),
            "unparsed": sum(p is None for p in preds),
        }
    for a, b2 in (("cot", "direct"), ("cot", "json"), ("direct", "json")):
        pa, pb = fmt_preds[a], fmt_preds[b2]
        agree = sum(x == y for x, y in zip(pa, pb, strict=True))
        ca = [x == it.answer for x, it in zip(pa, items, strict=True)]
        cb = [y == it.answer for y, it in zip(pb, items, strict=True)]
        fmts[f"{a}_vs_{b2}"] = {
            "same_answer": agree / len(items),
            "answer_changed": 1 - agree / len(items),
            "mcnemar_p_accuracy": mcnemar_exact(
                sum(x and not y for x, y in zip(ca, cb, strict=True)),
                sum(y and not x for x, y in zip(ca, cb, strict=True)),
            ),
        }
    out["formats"] = fmts
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export1", "export2", "analyze"])
    parser.add_argument("--config", default="configs/phase1_full.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--generations", type=Path, action="append", default=[])
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    if args.mode == "export1":
        export1(cfg, args.out)
    elif args.mode == "export2":
        export2(cfg, args.generations, args.out)
    else:
        run_dir = start_run("phase1_probes", cfg)
        res = analyze(cfg, args.generations, run_dir)
        (run_dir / "probes_summary.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
