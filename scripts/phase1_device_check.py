"""Phase 1: do the Mac (HF, MPS, batch 8) and TPU (vLLM) runs give the same answers?

Compares the 500 pilot jobs (50 questions x 10 conditions) between the Mac pilot run and the
TPU run of the same prompts: parsed answers, and labels if both runs were labeled.

Usage: uv run python scripts/phase1_device_check.py <mac_run_dir> <tpu_run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medlens.eval.stats import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mac_run", type=Path)
    parser.add_argument("tpu_run", type=Path)
    args = parser.parse_args()
    mac = pd.read_parquet(args.mac_run / "labels.parquet").set_index("job_id")
    tpu = pd.read_parquet(args.tpu_run / "labels.parquet").set_index("job_id")
    common = mac.index.intersection(tpu.index)
    m, t = mac.loc[common], tpu.loc[common]
    same_answer = int((m.pred.fillna("-") == t.pred.fillna("-")).sum())
    both_labeled = m.label.notna() & t.label.notna()
    same_label = int((m.label[both_labeled] == t.label[both_labeled]).sum())
    out = {
        "n_jobs": len(common),
        "same_answer": same_answer,
        "same_answer_rate": same_answer / len(common),
        "same_answer_ci": wilson_interval(same_answer, len(common)),
        "n_both_labeled": int(both_labeled.sum()),
        "same_label": same_label,
        "same_label_rate": same_label / int(both_labeled.sum()) if both_labeled.any() else None,
        "baseline_accuracy": {
            "mac": float(m[m.condition == "none"].correct.mean()),
            "tpu": float(t[t.condition == "none"].correct.mean()),
        },
        "label_counts": {
            "mac": m.label.value_counts().to_dict(),
            "tpu": t.label.value_counts().to_dict(),
        },
        "same_answer_by_condition": (m.pred.fillna("-") == t.pred.fillna("-"))
        .groupby(m.condition)
        .mean()
        .round(3)
        .to_dict(),
    }
    (args.tpu_run / "device_check.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
