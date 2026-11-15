"""Phase 1: final tables and figure for reports/phase1.md.

Reads the labeled hint run and the probe run, re-computes the per-condition rates with and
without the TPU answers whose text starts mid-word (a vLLM-TPU artifact), pools across
framings, and draws the main figure.

Usage: uv run python scripts/phase1_report_tables.py <hints_run_dir> <probes_run_dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from medlens.eval.stats import wilson_interval
from medlens.runs import REPO_ROOT, read_jsonl

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
UNACK, ACK = "#2a78d6", "#eb6834"  # validated pair (dataviz palette checker, light mode)
FRAMING_LABELS = {
    "metadata": "Metadata tag",
    "explicit": 'Explicit "Hint:"',
    "authority": "Attending's opinion",
    "patient": "Patient insists",
}


def broken_start(text: str) -> bool:
    return bool(re.match(r"^['’]s\b|^[a-z]", (text or "").lstrip()))


def rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df[df.condition != "none"].groupby("condition", sort=False):
        e = g[g.eligible.astype(bool)]
        n, f = len(e), int(e.followed.astype(bool).sum())
        unf, fai = int((e.label == "unfaithful").sum()), int((e.label == "faithful").sum())
        rows.append(
            {
                "condition": cond,
                "framing": g.framing.iloc[0],
                "direction": g.direction.iloc[0],
                "accuracy": g.correct.mean(),
                "n_eligible": n,
                "followed": f,
                "follow_rate": f / n if n else None,
                "follow_ci": wilson_interval(f, n) if n else None,
                "faithful": fai,
                "unfaithful": unf,
                "ack_among_followed": fai / (fai + unf) if fai + unf else None,
                "ack_ci": wilson_interval(fai, fai + unf) if fai + unf else None,
                "unfaithful_rate": unf / n if n else None,
                "unfaithful_ci": wilson_interval(unf, n) if n else None,
            }
        )
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, sham_rate: float, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, sharey=True)
    titles = {
        "wrong": "Hint points to a WRONG answer\n(questions the hint could move, n ≈ 322 each)",
        "gold": "Hint points to the CORRECT answer\n(questions missed unhinted, n ≈ 128 each)",
    }
    order = ["metadata", "explicit", "authority", "patient"]
    for ax, direction in zip(axes, ("wrong", "gold"), strict=True):
        ax.set_facecolor(SURFACE)
        sub = t[t.direction == direction].set_index("framing").loc[order]
        x = range(len(order))
        unf = sub.unfaithful / sub.n_eligible
        fai = sub.faithful / sub.n_eligible
        ax.bar(
            x,
            unf,
            color=UNACK,
            width=0.6,
            label="Followed, reasoning never mentions the hint",
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.bar(
            x,
            fai,
            bottom=unf,
            color=ACK,
            width=0.6,
            label="Followed, reasoning mentions it",
            edgecolor=SURFACE,
            linewidth=2,
        )
        lo = [r.follow_rate - r.follow_ci[0] for r in sub.itertuples()]
        hi = [r.follow_ci[1] - r.follow_rate for r in sub.itertuples()]
        ax.errorbar(x, sub.follow_rate, yerr=[lo, hi], fmt="none", ecolor=INK_2, capsize=3, lw=1)
        for i, r in enumerate(sub.itertuples()):
            ax.annotate(
                f"{r.follow_rate:.0%}",
                (i, r.follow_ci[1]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                color=INK_2,
                fontsize=9,
            )
        if direction == "wrong":
            ax.axhline(
                sham_rate,
                color=INK_2,
                ls="--",
                lw=1,
                label=f"Sham sentence (noise floor): {sham_rate:.0%}",
            )
        ax.set_xticks(list(x), [FRAMING_LABELS[f] for f in order], fontsize=9)
        ax.set_title(titles[direction], loc="left", fontsize=10.5)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Answer moved to the hint (share of eligible)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0)
    )
    fig.text(
        0.01,
        0.005,
        "MedGemma 4B, 375 MedQA questions, step-by-step answers (vLLM on TPU). "
        "Whiskers: 95% Wilson interval on the share that followed the hint.",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.01, 0.05, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hints_run", type=Path)
    parser.add_argument("probes_run", type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.hints_run / "labels.parquet")
    texts = {r["job_id"]: r["text"] for r in read_jsonl(args.hints_run / "rollouts.jsonl")}
    df["broken_start"] = df.job_id.map(lambda j: broken_start(texts[j]))
    base_broken = set(df[(df.condition == "none") & df.broken_start].item_id)

    main_t = rates(df)
    clean = df[~df.broken_start & ~df.item_id.isin(base_broken)]
    clean_t = rates(clean)
    hinted = df[df.framing.notna() & df.eligible.astype(bool)]
    n_f = int(hinted.followed.astype(bool).sum())
    n_fai = int((hinted.label == "faithful").sum())
    sham = main_t[main_t.condition == "sham"].iloc[0]

    out = {
        "n_questions": int(df.item_id.nunique()),
        "baseline_accuracy": float(df[df.condition == "none"].correct.mean()),
        "broken_start_answers": int(df.broken_start.sum()),
        "sham_follow_rate": sham.follow_rate,
        "sham_follow_ci": sham.follow_ci,
        "pooled": {
            "hinted_eligible": len(hinted),
            "followed": n_f,
            "faithful": n_fai,
            "unfaithful": int((hinted.label == "unfaithful").sum()),
            "ack_among_followed": n_fai / n_f,
            "ack_ci": wilson_interval(n_fai, n_f),
            "questions_with_unfaithful": int(df[df.label == "unfaithful"].item_id.nunique()),
        },
        "max_abs_change_excluding_broken": {
            col: float(
                (main_t.set_index("condition")[col] - clean_t.set_index("condition")[col])
                .abs()
                .max()
            )
            for col in ("follow_rate", "unfaithful_rate", "ack_among_followed")
        },
        "table": main_t.to_dict("records"),
        "probes": json.loads((args.probes_run / "probes_summary.json").read_text()),
    }
    figure(
        main_t[main_t.framing.notna()],
        sham.follow_rate,
        REPO_ROOT / "reports/figures/phase1_hint_following.png",
    )
    path = args.hints_run / "report_numbers.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    cols = [
        "condition",
        "accuracy",
        "n_eligible",
        "followed",
        "follow_rate",
        "faithful",
        "unfaithful",
        "ack_among_followed",
        "unfaithful_rate",
    ]
    print(main_t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("table", "probes")}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
