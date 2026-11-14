"""Phase 0: check that MedQA, MedMCQA and MPIB load, and report their sizes and schemas."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from dotenv import load_dotenv
from huggingface_hub.errors import GatedRepoError

from medlens.config import load_config
from medlens.data.loaders import load_medmcqa, load_medqa, load_mpib_raw, question_overlap
from medlens.data.schema import format_mcq
from medlens.runs import REPO_ROOT, start_run

logger = logging.getLogger("phase0_data_check")


def describe_mpib(rows: list[dict]) -> dict:
    """Schema and label counts only; no payload text is written out."""
    keys = sorted(rows[0].keys())
    low_cardinality = {}
    for k in keys:
        values = [
            json.dumps(r.get(k), sort_keys=True) if not isinstance(r.get(k), str) else r.get(k)
            for r in rows
        ]
        counts = Counter(values)
        if len(counts) <= 15:
            low_cardinality[k] = dict(counts.most_common())
    types = {k: type(rows[0][k]).__name__ for k in keys}
    return {"n": len(rows), "fields": types, "categorical_fields": low_cardinality}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    run_dir = start_run("phase0_data_check", cfg)
    dcfg = cfg["data"]
    summary: dict = {}

    medqa = {split: load_medqa(split) for split in ("train", "validation", "test")}
    summary["medqa"] = {
        "repo": "bigbio/med_qa (med_qa_en_4options_source)",
        "counts": {k: len(v) for k, v in medqa.items()},
        "answer_distribution_test": dict(Counter(i.answer for i in medqa["test"])),
    }
    medmcqa = {split: load_medmcqa(split) for split in ("train", "validation")}
    summary["medmcqa"] = {
        "repo": "openlifescienceai/medmcqa",
        "counts": {k: len(v) for k, v in medmcqa.items()},
        "subjects_validation": dict(
            Counter(i.meta["subject"] for i in medmcqa["validation"]).most_common()
        ),
    }

    try:
        mpib = {split: load_mpib_raw(split) for split in ("train", "validation", "test")}
        summary["mpib"] = {"status": "ok", **{k: describe_mpib(v) for k, v in mpib.items()}}
        text_fields = [k for k, t in summary["mpib"]["test"]["fields"].items() if t == "str"]
        logger.info("MPIB string fields: %s", text_fields)
    except GatedRepoError as e:
        summary["mpib"] = {"status": "no_access", "error": str(e).splitlines()[-1]}
        logger.warning(
            "MPIB not accessible: request access at huggingface.co/datasets/jhlee0619/mpib"
        )

    all_medqa_q = [i.question for split in medqa.values() for i in split]
    summary["overlap_medqa_test_vs_train"] = len(
        question_overlap([i.question for i in medqa["test"]], [i.question for i in medqa["train"]])
    )
    if summary["mpib"]["status"] == "ok":
        summary["mpib_medqa_overlap_note"] = (
            "MPIB question field not yet identified; overlap check added in Phase 1"
        )

    preview = [
        format_mcq(i) + f"\n[answer: {i.answer}]" for i in medqa["test"][: dcfg["n_preview"]]
    ]
    (run_dir / "medqa_preview.txt").write_text("\n\n---\n\n".join(preview))
    summary["n_medqa_questions_total"] = len(all_medqa_q)
    (run_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
