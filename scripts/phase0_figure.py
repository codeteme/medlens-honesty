"""Phase 0 figure: Gemma Scope 2 fit on MedGemma 4B vs the Gemma 3 4B control, by layer.

Usage: uv run python scripts/phase0_figure.py outputs/phase0_sae_recon/<run>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from medlens.runs import REPO_ROOT

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # validated with the dataviz palette checker (light mode): all checks pass
    "google/medgemma-4b-it": ("MedGemma 4B", "#2a78d6", "o"),
    "google/gemma-3-4b-it": ("Gemma 3 4B (control)", "#eb6834", "s"),
}
PANELS = [
    ("fvu", "res", "Residual SAE: variance left unexplained"),
    ("fvu", "transcoder", "Transcoder: variance left unexplained"),
    ("kl", "res", "Residual SAE: change in predictions (KL, nats)"),
    ("kl", "transcoder", "Transcoder: change in predictions (KL, nats)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--suite", default="pt")
    parser.add_argument("--domain", default="medqa_chat")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "reports/figures/phase0_sae_fit.png"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.run_dir / "summary.csv")
    df = df[(df.suite == args.suite) & (df.domain == args.domain)]

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
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    for ax, (metric, kind, title) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        sub = df[df.kind == kind]
        ends = [sub[sub.model == m].sort_values("layer")[metric].iloc[-1] for m in SERIES]
        close = abs(ends[0] - ends[1]) < 0.06 * max(sub[metric].max(), 1e-9)
        for i, (model, (label, color, marker)) in enumerate(SERIES.items()):
            s = sub[sub.model == model].sort_values("layer")
            ax.plot(
                s.layer,
                s[metric],
                color=color,
                lw=2,
                marker=marker,
                ms=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=label,
            )
            last = s.iloc[-1]
            # Separate end labels vertically when the two series end at nearly the same value.
            dy = 0 if not close else (7 if ends[i] >= ends[1 - i] else -7)
            ax.annotate(
                f"{last[metric]:.2f}",
                (last.layer, last[metric]),
                xytext=(8, dy),
                textcoords="offset points",
                va="center",
                color=INK_2,
                fontsize=9,
            )
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xticks(sorted(df.layer.unique()))
        ax.set_xlim(left=7, right=32)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Layer (of 34)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995)
    )
    fig.suptitle("")
    fig.text(
        0.01,
        0.005,
        f"Gemma Scope 2 '{args.suite}' suite, 16k width, on {args.domain} "
        "(30 texts per point). Lower is better in every panel.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
