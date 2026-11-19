"""Phase 8 report figure: detector scores for letter cues vs cues that don't name a letter.

Usage: uv run python scripts/phase8_figures.py <phase8_run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phase2_figures import CTRL, GRID, INK, INK_2, MAIN, OUT, SURFACE, style

CUES = [
    ("authority", "Attending,\nnames letter"),
    ("metadata", "Metadata,\nnames letter"),
    ("authority_text", "Attending,\nnames option text"),
    ("metadata_text", "Metadata,\nnames option text"),
    ("evidence", "Made-up clinical\nfinding"),
]
BADGE = 0.86


def scores(run: Path) -> None:
    r = pd.read_parquet(run / "rows.parquet")
    e = r[r.eligible & r.label.notna()]
    rng = np.random.default_rng(0)
    style()
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    groups = [("unaffected", -0.18, CTRL), ("unfaithful", 0.18, MAIN)]
    for i, (cue, _) in enumerate(CUES):
        d = e[e.framing == cue]
        for lab, dx, color in groups:
            s = d[d.label == lab].score.clip(1e-4, 1 - 1e-4)
            x = i + dx + rng.uniform(-0.09, 0.09, len(s))
            ax.scatter(x, s, s=7, color=color, alpha=0.45, lw=0)
            if len(s) >= 10:
                ax.plot([i + dx - 0.12, i + dx + 0.12], [s.median()] * 2, color=INK, lw=2)
    ax.axhline(BADGE, color=INK_2, lw=1, ls="--")
    ax.text(len(CUES) - 0.5, BADGE, " badge\n threshold", va="center", fontsize=8, color=INK_2)
    ax.set_yscale("logit")
    ax.set_ylim(1e-4, 1 - 1e-4)
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(len(CUES)), [lab for _, lab in CUES], fontsize=8.5)
    ax.set_ylabel("Detector score (logit scale)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.scatter([], [], s=20, color=CTRL, label="Ignored the cue")
    ax.scatter([], [], s=20, color=MAIN, label="Followed it without saying so")
    ax.plot([], [], color=INK, lw=2, label="Median (10+ runs)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    ax.set_title(
        "The detector still ranks text-named cues above chance, but its scores stay low",
        fontsize=10,
        loc="left",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "phase8_scores.png", dpi=160, facecolor=SURFACE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    scores(p.parse_args().run)


if __name__ == "__main__":
    main()
