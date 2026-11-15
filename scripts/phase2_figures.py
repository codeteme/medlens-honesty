"""Phase 2 report figures: causal tests and readout validity.

Usage: uv run python scripts/phase2_figures.py --causal <run> [<run> ...] --analyze <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from medlens.runs import REPO_ROOT

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MAIN, CTRL = "#2a78d6", "#eb6834"  # validated pair (dataviz palette checker, light mode)
OUT = REPO_ROOT / "reports" / "figures"


def style() -> None:
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


def describe(run: Path) -> str:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    a = cfg.get("args", {})
    sel = (
        "attribution-ranked" if "attr_rank" in str(a.get("analyze_run", "")) else "most correlated"
    )
    where = {"hint": "on hint words", "final": "at answer point", "all": "at every token"}[
        a.get("region", "all")
    ]
    return f"Layer {a['layer']}, {sel},\n{where}"


def causal_figure(runs: list[Path]) -> None:
    rows = []
    for run in runs:
        d = json.loads((run / "causal_summary.json").read_text())
        top = d["sets"]["top10"]
        rand = [v for k, v in d["sets"].items() if k.startswith("random10")]
        rows.append((describe(run), top, rand))
    rows.sort(key=lambda r: r[1]["mean_delta_logp_target_hinted"])
    style()
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_label, top, rand) in enumerate(rows):
        m = top["mean_delta_logp_target_hinted"]
        lo, hi = top["delta_target_ci"]
        ax.barh(i + 0.18, m, height=0.34, color=MAIN, label="Top 10 features" if i == 0 else None)
        ax.errorbar(m, i + 0.18, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK_2, capsize=3)
        for j, rr in enumerate(rand):
            ax.barh(
                i - 0.18 + 0.12 * (j - 0.5),
                rr["mean_delta_logp_target_hinted"],
                height=0.12,
                color=CTRL,
                label="Matched random features" if i == 0 and j == 0 else None,
            )
        ax.annotate(
            f"follows hint {top['readout_follows_hint_before']:.0%} → "
            f"{top['readout_follows_hint_after']:.0%}",
            (min(lo, -0.02), i + 0.18),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_xlabel(
        "Change in log-probability of the hinted (wrong) answer when the features are "
        "switched off\n(more negative = the features were driving hint-following)"
    )
    lim = min(r[1]["delta_target_ci"][0] for r in rows) * 1.6
    ax.set_xlim(lim, 0.15)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title(
        "Which features cause silent hint-following? (MedGemma 4B, 60–80 unfaithful "
        "cases per test)",
        loc="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase2_causal_tests.png", dpi=150, facecolor=SURFACE)


def readout_figure(analyze: Path) -> None:
    d = json.loads((analyze / "readout_check.json").read_text())
    fr = ["metadata", "explicit", "authority", "patient"]
    style()
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    vals = [d[f]["lp_target_auroc_unfaithful_vs_unaffected"] for f in fr]
    ax.bar(range(4), vals, color=MAIN, width=0.6)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:.2f}",
            (i, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_2,
        )
    ax.axhline(0.5, color=INK_2, ls="--", lw=1, label="chance")
    ax.set_xticks(range(4), ["Metadata", "Explicit", "Attending", "Patient"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title(
        "One-pass readout predicts which Phase 1 cases were unfaithful", loc="left", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "phase2_readout_validity.png", dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", type=Path, nargs="+", required=True)
    parser.add_argument("--analyze", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    causal_figure(args.causal)
    readout_figure(args.analyze)
    print("saved", OUT / "phase2_causal_tests.png", OUT / "phase2_readout_validity.png")


if __name__ == "__main__":
    main()
