"""Loaders for MedQA, MedMCQA and MPIB, normalized to `MCQItem`.

Dataset IDs (verified on Hugging Face, 2026-11-14):
- MedQA (USMLE, 4 options): `bigbio/med_qa`, config folder `med_qa_en_4options_source`
- MedMCQA: `openlifescienceai/medmcqa` (the test split has no answers: `cop == -1`)
- MPIB: `jhlee0619/mpib` (gated; request access on its Hugging Face page)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from medlens.data.schema import LETTERS, MCQItem

MEDQA_REPO = "bigbio/med_qa"
MEDQA_FOLDER = "med_qa_en_4options_source"
MEDMCQA_REPO = "openlifescienceai/medmcqa"
MPIB_REPO = "jhlee0619/mpib"


def _stable_id(source: str, text: str) -> str:
    return f"{source}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def _isna(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == "nan"


def normalize_medqa_row(row: Mapping[Any, Any], split: str) -> MCQItem:
    """MedQA rows store options as a list of {"key": letter, "value": text}."""
    options = {opt["key"]: opt["value"] for opt in row["options"]}
    options = dict(sorted(options.items()))
    return MCQItem(
        id=_stable_id("medqa", row["question"]),
        source="medqa",
        question=row["question"],
        options=options,
        answer=row["answer_idx"],
        meta={"split": split, "step": row.get("meta_info")},
    )


def normalize_medmcqa_row(row: Mapping[Any, Any], split: str) -> MCQItem:
    """MedMCQA stores options in opa..opd and the answer as index `cop` (0-3)."""
    cop = int(row["cop"])
    if cop < 0:
        raise ValueError(f"MedMCQA row {row['id']} has no answer (cop={cop})")
    texts = [row["opa"], row["opb"], row["opc"], row["opd"]]
    return MCQItem(
        id=f"medmcqa-{row['id']}",
        source="medmcqa",
        question=row["question"],
        options={LETTERS[i]: t for i, t in enumerate(texts)},
        answer=LETTERS[cop],
        meta={
            "split": split,
            "subject": None if _isna(row.get("subject_name")) else row["subject_name"],
            "topic": None if _isna(row.get("topic_name")) else row["topic_name"],
            "choice_type": row.get("choice_type"),
            "explanation": None if _isna(row.get("exp")) else row["exp"],
        },
    )


def _sample(items: list[MCQItem], n: int | None, seed: int) -> list[MCQItem]:
    if n is None or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def load_medqa(split: str = "test", n: int | None = None, seed: int = 0) -> list[MCQItem]:
    path = hf_hub_download(
        MEDQA_REPO, f"{MEDQA_FOLDER}/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)
    items = [normalize_medqa_row(r, split) for r in df.to_dict("records")]
    return _sample(items, n, seed)


def load_medmcqa(split: str = "validation", n: int | None = None, seed: int = 0) -> list[MCQItem]:
    if split == "test":
        raise ValueError("MedMCQA test split has no public answers; use 'validation'.")
    path = hf_hub_download(
        MEDMCQA_REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)
    items = [normalize_medmcqa_row(r, split) for r in df.to_dict("records")]
    return _sample(items, n, seed)


def load_mpib_raw(split: str = "test") -> list[dict[str, Any]]:
    """Return MPIB rows as-is. Schema is documented in reports/phase0.md once inspected.

    Never copy MPIB's restricted payloads into public artifacts (demo, README).
    """
    path = hf_hub_download(MPIB_REPO, f"data/{split}.jsonl", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def question_overlap(a: Iterable[str], b: Iterable[str]) -> set[str]:
    """Questions appearing in both collections, after whitespace/case normalization.

    Used to keep MPIB items (built from MedQA) out of MedQA train/test splits.
    """

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    return {norm(x) for x in a} & {norm(x) for x in b}


WIKITEXT_REPO = "Salesforce/wikitext"


def load_wikitext_paragraphs(
    n: int, seed: int = 0, min_chars: int = 600, config: str = "wikitext-103-raw-v1"
) -> list[str]:
    """General-domain prose (Wikipedia articles) as a non-medical comparison text."""
    path = hf_hub_download(
        WIKITEXT_REPO, f"{config}/test-00000-of-00001.parquet", repo_type="dataset"
    )
    lines = pd.read_parquet(path)["text"].tolist()
    paragraphs = [
        t.strip() for t in lines if len(t.strip()) >= min_chars and not t.strip().startswith("=")
    ]
    return random.Random(seed).sample(paragraphs, min(n, len(paragraphs)))
