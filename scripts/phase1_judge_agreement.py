"""Phase 1: agreement between reference labels and the judge (and the Afolabi regex).

Reference labels come from hand labels in label_sheet.csv and/or blind LLM-annotator files
(columns: case_id,label). Each annotator is compared with the judge, the regex, and the other
annotators. Agreement with an LLM annotator shows consistency between models, not validity
against a person; reports must say which kind of reference was used.

Usage:
  uv run python scripts/phase1_judge_agreement.py outputs/phase1_hints/<run>/ \
      [--annotator llm_A=llm_labels_A.csv --annotator llm_B=llm_labels_B.csv]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from medlens.eval.stats import cohens_kappa, wilson_interval

HAND_COL = "your_label (1 = reasoning refers to the hint, 0 = it does not)"
KEY_FILE = "label_sheet_KEY_do_not_open_before_labeling.csv"


def compare(a: list[int], b: list[int]) -> dict:
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    return {
        "n": len(a),
        "agreement": agree / len(a),
        "agreement_ci": wilson_interval(agree, len(a)),
        "kappa": cohens_kappa(a, b),
        "positives": (sum(a), sum(b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--annotator", action="append", default=[], help="name=file.csv")
    args = parser.parse_args()

    key = pd.read_csv(args.run_dir / KEY_FILE)
    df = key.copy()
    df["judge"] = df["ack"].astype(str).str.lower().eq("true").astype(int)
    df["regex"] = df["afolabi_ack"].astype(str).str.lower().eq("true").astype(int)

    names = []
    sheet = pd.read_csv(args.run_dir / "label_sheet.csv")
    hand = sheet[sheet[HAND_COL].astype(str).str.strip().isin(["0", "1"])]
    if len(hand):
        df = df.merge(
            hand[["case_id", HAND_COL]].rename(columns={HAND_COL: "human"}),
            on="case_id",
            how="left",
        )
        names.append("human")
    for spec in args.annotator:
        name, file = spec.split("=", 1)
        ann = pd.read_csv(args.run_dir / file)[["case_id", "label"]].rename(columns={"label": name})
        df = df.merge(ann, on="case_id", how="left")
        names.append(name)
    if not names:
        raise SystemExit("No reference labels: fill label_sheet.csv or pass --annotator.")

    out: dict = {"references": names, "comparisons": {}}
    for ref in names:
        sub = df[df[ref].notna()]
        r = sub[ref].astype(int).tolist()
        for other in ("judge", "regex"):
            out["comparisons"][f"{ref}_vs_{other}"] = compare(r, sub[other].tolist())
        dis = sub[sub[ref].astype(int) != sub["judge"]]
        out[f"{ref}_vs_judge_disagreements"] = dis[
            ["case_id", "condition", "followed", ref, "judge", "judge_reason"]
        ].to_dict("records")
    for a, b in itertools.combinations(names, 2):
        sub = df[df[a].notna() & df[b].notna()]
        out["comparisons"][f"{a}_vs_{b}"] = compare(
            sub[a].astype(int).tolist(), sub[b].astype(int).tolist()
        )
    (args.run_dir / "judge_agreement.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
