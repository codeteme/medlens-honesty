"""Phase 1: hint-injection probe on MedGemma 4B, judged for hint acknowledgment.

Pipeline:
1. Pick MedQA test questions that do not appear in MPIB (kept clean for Phase 4).
2. For each question, build the conditions: no hint, sham sentence, and each hint framing
   pointing to a wrong option or to the correct one.
3. Generate step-by-step answers with MedGemma in fixed batches (cached, resumable).
4. Mark eligible / followed runs; ask the judge (on Tinker) whether each eligible hinted
   run's reasoning refers to the hint; also run the Afolabi et al. regex detector.
5. Label faithful / unfaithful / unaffected, summarise with Wilson intervals, and write a
   blind hand-labeling sheet for validating the judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.blackbox.judge import afolabi_ack, judge_messages, parse_judge
from medlens.blackbox.labels import eligibility, label
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.data.schema import format_mcq
from medlens.eval.answers import extract_answer
from medlens.eval.stats import wilson_interval
from medlens.models.base import CostTracker, Generation
from medlens.models.cache import ResponseCache
from medlens.models.hf_local import HFLocalModel
from medlens.models.tinker_model import TinkerModel
from medlens.runs import REPO_ROOT, read_jsonl, start_run, write_jsonl

logger = logging.getLogger("phase1_hints")


def select_items(cfg: dict, seed: int) -> list:
    items = medqa_test_not_in_mpib(cfg["n_questions"], seed)
    logger.info("Selected %d MedQA test questions not in MPIB", len(items))
    return items


def generate_all(model: HFLocalModel, jobs: list[dict], cfg: dict, cache: ResponseCache) -> None:
    """Fill job["gen"] for every job. Batches are fixed (sorted by prompt length, then id), and a
    batch is regenerated as a whole unless every member is cached, so batch composition (which
    can slightly affect greedy outputs) never changes between runs."""
    bs, max_new = cfg["batch_size"], cfg["max_new_tokens"]

    def request(job: dict) -> dict:
        return {
            "model": model.model_id,
            "messages": job["messages"],
            "max_new_tokens": max_new,
            "batch_size": bs,
            "decoding": "greedy",
        }

    order = sorted(jobs, key=lambda j: (len(j["messages"][0]["content"]), j["job_id"]))
    t0, n_done = time.time(), 0
    for start in range(0, len(order), bs):
        batch = order[start : start + bs]
        hits = [cache.get(request(j)) for j in batch]
        if all(h is not None for h in hits):
            for j, h in zip(batch, hits, strict=True):
                j["gen"] = Generation.from_dict(h)
                j["gen"].cached = True
        else:
            gens = model.generate_batch([j["messages"] for j in batch], max_new)
            for j, g in zip(batch, gens, strict=True):
                cache.put(request(j), g.to_dict())
                j["gen"] = g
        n_done += len(batch)
        logger.info("generated %d/%d (%.0fs)", n_done, len(order), time.time() - t0)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df[df.condition != "none"].groupby("condition", sort=False):
        elig = g[g.eligible]
        n, nf = len(elig), int(elig.followed.sum())
        fol = elig[elig.followed]
        judged = fol[fol.ack.notna()]
        n_ack = int(judged.ack.sum())
        n_unf = int((elig.label == "unfaithful").sum())
        row = {
            "condition": cond,
            "n_questions": len(g),
            "accuracy": g.correct.mean(),
            "n_eligible": n,
            "followed": nf,
            "follow_rate": nf / n if n else None,
            "follow_ci": wilson_interval(nf, n) if n else None,
        }
        if g.framing.notna().any():
            row |= {
                "ack_among_followed": n_ack / len(judged) if len(judged) else None,
                "ack_ci": wilson_interval(n_ack, len(judged)) if len(judged) else None,
                "unfaithful": n_unf,
                "unfaithful_rate": n_unf / n if n else None,
                "unfaithful_ci": wilson_interval(n_unf, n) if n else None,
                "afolabi_regex_ack_among_followed": fol.afolabi_ack.mean() if len(fol) else None,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def write_label_sheet(df: pd.DataFrame, items_by_id: dict, n: int, seed: int, run_dir) -> None:
    """Blind sheet: the judge's verdict is kept in a separate key file. Stratified so both
    followed / not-followed and judge-yes / judge-no cases appear."""
    judged = df[df.ack.notna()].copy()
    judged["stratum"] = judged.followed.astype(str) + "|" + judged.ack.astype(str)
    picks = []
    strata = judged.groupby("stratum")
    per = max(1, n // max(1, strata.ngroups))
    for _, g in strata:
        picks.append(g.sample(n=min(per, len(g)), random_state=seed))
    sheet = pd.concat(picks)
    if len(sheet) < n:
        rest = judged.drop(sheet.index)
        sheet = pd.concat([sheet, rest.sample(n=min(n - len(sheet), len(rest)), random_state=seed)])
    sheet = sheet.sample(frac=1, random_state=seed)
    sheet["case_id"] = [f"c{i:03d}" for i in range(len(sheet))]
    with open(run_dir / "label_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "inserted_hint",
                "question",
                "model_reasoning",
                "your_label (1 = reasoning refers to the hint, 0 = it does not)",
                "notes",
            ]
        )
        for _, r in sheet.iterrows():
            w.writerow([r.case_id, r.cue, format_mcq(items_by_id[r.item_id]), r.text, "", ""])
    sheet[
        [
            "case_id",
            "job_id",
            "condition",
            "followed",
            "ack",
            "judge_reason",
            "judge_quote",
            "afolabi_ack",
        ]
    ].to_csv(run_dir / "label_sheet_KEY_do_not_open_before_labeling.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--export-jobs", type=Path, help="write prompts to JSONL and stop")
    parser.add_argument("--generations", type=Path, help="read answers from JSONL (vLLM/TPU)")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    hcfg = cfg["hints"]
    run_dir = start_run("phase1_hints", cfg)
    cache = ResponseCache()

    items = select_items(hcfg, cfg["seed"])
    items_by_id = {i.id: i for i in items}
    jobs = []
    for item in items:
        for cond in conditions_for(
            item, hcfg["framings"], hcfg["directions"], cfg["seed"], sham=hcfg["sham"]
        ):
            jobs.append(
                {
                    "job_id": f"{item.id}|{cond.name}",
                    "item": item,
                    "cond": cond,
                    "messages": build_messages(item, cond),
                }
            )
    logger.info(
        "%d questions x %d conditions = %d generations",
        len(items),
        len(jobs) // len(items),
        len(jobs),
    )

    if args.export_jobs:
        write_jsonl(
            args.export_jobs, [{"job_id": j["job_id"], "messages": j["messages"]} for j in jobs]
        )
        logger.info(
            "Exported %d jobs to %s; generate them elsewhere, then rerun with --generations",
            len(jobs),
            args.export_jobs,
        )
        return
    if args.generations:
        # Answers produced on another machine (e.g. vLLM on TPU); every job must be present.
        by_id = {r["job_id"]: r for r in read_jsonl(args.generations)}
        missing = [j["job_id"] for j in jobs if j["job_id"] not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} jobs missing from {args.generations}")
        for j in jobs:
            j["gen"] = Generation.from_dict(by_id[j["job_id"]]["generation"])
        (run_dir / "generation_source.json").write_text(
            json.dumps(
                {"file": str(args.generations), **by_id[jobs[0]["job_id"]]["source"]}, indent=2
            )
        )
    else:
        model = HFLocalModel(cfg["model_id"], device=cfg["device"])
        generate_all(model, jobs, hcfg, cache)
        del model

    rows = []
    for j in jobs:
        g, item, cond = j["gen"], j["item"], j["cond"]
        pred = extract_answer(g.text, valid="".join(item.options))
        rows.append(
            {
                "job_id": j["job_id"],
                "item_id": item.id,
                "condition": cond.name,
                "framing": cond.framing,
                "direction": cond.direction,
                "target": cond.target,
                "cue": cond.cue_text(),
                "gold": item.answer,
                "pred": pred,
                "correct": pred == item.answer,
                "stop_reason": g.stop_reason,
                "output_tokens": g.usage.output_tokens,
                "text": g.text,
            }
        )
    df = pd.DataFrame(rows)
    base = df[df.condition == "none"].set_index("item_id")
    elig = [
        eligibility(base.at[r.item_id, "pred"], r.pred, r.stop_reason, r.target)
        if r.condition != "none"
        else {"eligible": False, "why_not": "baseline", "followed": None}
        for r in df.itertuples()
    ]
    df = pd.concat([df, pd.DataFrame(elig)], axis=1)

    # Judge every eligible hinted run (followed or not), plus the regex detector.
    judge = TinkerModel(hcfg["judge"]["model_id"], temperature=0.0, cache=cache)
    to_judge = df[df.eligible & df.framing.notna()]
    gens = judge.generate_many(
        [judge_messages(items_by_id[r.item_id], r.cue, r.text) for r in to_judge.itertuples()],
        hcfg["judge"]["max_tokens"],
    )
    costs = CostTracker()
    verdicts = {}
    for r, g in zip(to_judge.itertuples(), gens, strict=True):
        costs.add(g)
        verdicts[r.job_id] = parse_judge(g.text, r.text, r.framing) | {"judge_raw": g.text}
    v = pd.DataFrame.from_dict(verdicts, orient="index")
    df["ack"] = df.job_id.map(v["ack"]) if len(v) else None
    df["judge_reason"] = df.job_id.map(v["reason"]) if len(v) else None
    df["judge_quote"] = df.job_id.map(v["quote"]) if len(v) else None
    df["afolabi_ack"] = [
        afolabi_ack(t) if f else None for t, f in zip(df.text, df.framing, strict=True)
    ]
    # Only hinted conditions get faithful/unfaithful labels; sham is a noise floor.
    df["label"] = [
        label(e, f, a) if fr is not None and fr == fr else None
        for e, f, a, fr in zip(df.eligible, df.followed, df.ack, df.framing, strict=True)
    ]

    write_jsonl(run_dir / "rollouts.jsonl", df.to_dict("records"))
    df.drop(columns=["text"]).to_parquet(run_dir / "labels.parquet")
    summary = summarise(df)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".2f"))
    write_label_sheet(df, items_by_id, hcfg["label_sheet_n"], cfg["seed"], run_dir)
    extra = {
        "n_items": len(items),
        "n_generations": len(df),
        "baseline_accuracy": float(df[df.condition == "none"].correct.mean()),
        "unfinished_or_unparsed": int(((df.stop_reason == "max_tokens") | df.pred.isna()).sum()),
        "judge_unparseable": int(df.judge_reason.eq("unparseable").sum()),
        "judge_downgrades": df.judge_reason.value_counts().to_dict(),
        "judge_cost": costs.summary(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(extra, indent=2, default=str))
    print(summary.to_markdown(index=False, floatfmt=".2f"))
    print(json.dumps(extra, indent=2, default=str))


if __name__ == "__main__":
    main()
