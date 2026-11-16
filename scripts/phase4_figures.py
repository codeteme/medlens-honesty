"""Phase 4 report figure: each fix's effect on hint-following, quick answer vs step-by-step.

Usage: uv run python scripts/phase4_figures.py <screen_run> <cot_dev_run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from phase2_figures import CTRL, GRID, INK, INK_2, MAIN, OUT, SURFACE, style

SETTINGS = [
    ("prompt_fix", "Prompt instruction only"),
    ("ablate_pre_L22", "Remove direction, layer 22"),
    ("steer_pre_L17_x4", "Steer against direction, layer 17 (4 gaps)"),
    ("steer_pre_L22_x4", "Steer against direction, layer 22 (4 gaps)"),
]


def screen_vs_cot(screen: Path, cot: Path) -> None:
    s = pd.read_csv(screen / "screen_summary.csv").set_index("setting")
    c = pd.read_csv(cot / "cot_summary.csv").set_index("setting")
    style()
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (key, _) in enumerate(SETTINGS):
        a, b = 100 * s.loc[key, "follow_change"], 100 * c.loc[key, "follow_change"]
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(a, i, "o", color=MAIN, ms=9, mec=SURFACE, mew=2, zorder=2)
        ax.plot(b, i, "o", color=CTRL, ms=9, mec=SURFACE, mew=2, zorder=2)
    ax.plot([], [], "o", color=MAIN, ms=9, label="Quick answer (one-pass readout)")
    ax.plot([], [], "o", color=CTRL, ms=9, label="Step-by-step answer")
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(range(len(SETTINGS)), [lab for _, lab in SETTINGS])
    ax.invert_yaxis()
    ax.set_xlabel("Change in hint-following vs no fix (percentage points, dev questions)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(
        "The fixes' effect on quick answers disappears in step-by-step answers",
        fontsize=10,
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "phase4_screen_vs_cot.png", dpi=160, facecolor=SURFACE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("screen_run", type=Path)
    p.add_argument("cot_run", type=Path)
    a = p.parse_args()
    screen_vs_cot(a.screen_run, a.cot_run)


if __name__ == "__main__":
    main()
