"""Phase 1: agreement between the hand labels and the judge (and the Afolabi regex).

Usage: uv run python scripts/phase1_judge_agreement.py outputs/phase1_hints/<run>/
Reads label_sheet.csv (filled in by hand) and the separate key file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medlens.eval.stats import cohens_kappa, wilson_interval

LABEL_COL = "your_label (1 = reasoning refers to the hint, 0 = it does not)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    sheet = pd.read_csv(args.run_dir / "label_sheet.csv")
    key = pd.read_csv(args.run_dir / "label_sheet_KEY_do_not_open_before_labeling.csv")
    df = sheet.merge(key, on="case_id")
    df = df[df[LABEL_COL].astype(str).str.strip().isin(["0", "1"])].copy()
    if df.empty:
        raise SystemExit("No hand labels found (fill the label column with 0 or 1).")
    human = df[LABEL_COL].astype(int).tolist()
    judge = df["ack"].astype(str).str.lower().eq("true").astype(int).tolist()
    regex = df["afolabi_ack"].astype(str).str.lower().eq("true").astype(int).tolist()
    agree = sum(h == j for h, j in zip(human, judge, strict=True))
    out = {
        "n_labeled": len(df),
        "human_positive": sum(human),
        "judge_positive": sum(judge),
        "judge_agreement": agree / len(df),
        "judge_agreement_ci": wilson_interval(agree, len(df)),
        "judge_kappa": cohens_kappa(human, judge),
        "regex_kappa": cohens_kappa(human, regex),
        "confusion_human_vs_judge": pd.crosstab(
            pd.Series(human, name="human"), pd.Series(judge, name="judge")
        ).to_dict(),
        "disagreements": df.loc[
            [h != j for h, j in zip(human, judge, strict=True)],
            ["case_id", "condition", "judge_reason"],
        ].to_dict("records"),
    }
    (args.run_dir / "judge_agreement.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
