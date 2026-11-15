"""Phase 3 report figures: probe AUROC by layer, and calibration of the best probes.

Usage: uv run python scripts/phase3_figures.py <probes_run> <followups_run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase2_figures import CTRL, GRID, INK, INK_2, MAIN, OUT, SURFACE, style

PANELS = [
    ("pre", "gen_start", "Before reasoning: end of prompt"),
    ("pre", "final", 'Before reasoning: forced "Final answer:"'),
    ("post", "post", 'After reasoning: "Final answer:"'),
    ("post", "cot_mean", "After reasoning: average over reasoning"),
]
LAYERS = ["L9", "L17", "L22", "L29"]


def auroc_figure(run: Path, followups: Path) -> None:
    t = pd.read_csv(run / "probe_table.csv")
    base = json.loads((run / "baselines.json").read_text())
    fu = json.loads((followups / "followups.json").read_text())
    refs = [
        ("LLM judge on the text", base["llm_judge_yes_rate"]["auroc"], "--"),
        ("Hint type alone", fu["hint_type_only"]["auroc"], "-."),
        ("TF-IDF on the text", base["text_tfidf_reasoning_only"]["auroc"], ":"),
    ]
    style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharey=True, facecolor=SURFACE)
    for ax, (stage, pos, title) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        for kind, color, label in (
            ("resid", MAIN, "Raw activations"),
            ("sae", CTRL, "SAE features"),
        ):
            d = t[(t.stage == stage) & (t.position == pos) & (t.input == kind)]
            d = d.set_index("layer").loc[LAYERS]
            ci = np.array([json.loads(c) for c in d.auroc_ci])
            x = np.arange(len(LAYERS))
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
            ax.plot(
                x, d.auroc, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label
            )
        for name, v, ls in refs:
            ax.axhline(v, color=INK_2, lw=1.2, ls=ls, label=name)
        ax.axhline(0.5, color=GRID, lw=1)
        ax.set_xticks(range(len(LAYERS)), [f"Layer {s[1:]}" for s in LAYERS])
        ax.set_xlim(-0.3, 3.3)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_ylim(0.45, 1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC (unfaithful vs unaffected)")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    fig.suptitle(
        "Can MedGemma 4B's internals flag a silently followed hint? (hint-blind probes, "
        "question-grouped 5-fold CV, 95% CI)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase3_probe_auroc.png", dpi=160, facecolor=SURFACE)


def calibration_figure(run: Path) -> None:
    t = pd.read_csv(run / "probe_table.csv")
    y = np.load(run / "labels.npy")
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, facecolor=SURFACE)
    for ax, st, color in zip(axes, ("pre", "post"), (MAIN, CTRL), strict=True):
        row = t[t.stage == st].iloc[0]
        p = np.load(run / f"oof_scores_{st}.npy")
        bins = np.clip((p * 10).astype(int), 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = bins == b
            if m.sum() >= 5:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                ns.append(m.sum())
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.plot(xs, ys, color=color, lw=2)
        ax.scatter(
            xs, ys, s=[max(30, n / 2) for n in ns], color=color, edgecolor=SURFACE, lw=2, zorder=3
        )
        where = {"pre": "Before reasoning, forced", "post": "After reasoning,"}[st]
        ax.set_title(
            f'{where} "Final answer:", layer {row.layer[1:]} activations\n'
            f"AUROC {row.auroc:.2f}, ECE {row.ece:.3f}, Brier {row.brier:.3f}",
            fontsize=9,
            loc="left",
            color=INK,
        )
        ax.set_xlabel("Probe's predicted chance of unfaithful")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Actual share unfaithful")
    fig.text(
        0.01,
        0.01,
        "Dot size = number of runs in the bin (bins with < 5 runs hidden).",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "phase3_calibration.png", dpi=160, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes_run", type=Path)
    parser.add_argument("followups_run", type=Path)
    args = parser.parse_args()
    auroc_figure(args.probes_run, args.followups_run)
    calibration_figure(args.probes_run)


if __name__ == "__main__":
    main()
