"""Phase 2: SAE attribution graphs for the clearest unfaithful cases.

Picks unfaithful cases (Phase 1 label) where the Mac readout also follows the hint, with the
largest pull toward the hinted letter, spread across hint framings. For each, builds a graph
(see medlens.interp.attribution), saves it as JSON, draws it, and summarises recurring
features and where attribution sits (hint tokens vs the rest of the prompt).

Usage: uv run python scripts/phase2_graphs.py <capture_run> <analyze_run> [--n 8]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.interp.attribution import build_graph
from medlens.interp.features import char_span_to_tokens
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run

logger = logging.getLogger("phase2_graphs")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POS_C, NEG_C, HINT_BG = "#2a78d6", "#eb6834", "#eef3fb"


def pick_cases(capture_run: Path, n: int) -> pd.DataFrame:
    r = pd.read_parquet(capture_run / "readout.parquet")
    sham = r[r.condition == "sham"].set_index("item_id")
    h = r[(r.p1_label == "unfaithful") & r.condition.str.endswith("_wrong")].copy()
    h = h[h.readout_pred == h.target]
    h["pull"] = [
        row[f"lp_{row.target}"] - sham.loc[row.item_id, f"lp_{row.target}"]
        for _, row in h.iterrows()
    ]
    per = max(1, n // h.framing.nunique())
    return (
        h.sort_values("pull", ascending=False)
        .groupby("framing")
        .head(per)
        .head(n)
        .reset_index(drop=True)
    )


def hint_positions(model, messages, cue) -> set[int]:
    prompt = model.chat_prompt(messages)
    enc = model.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    start = prompt.find(cue)
    return set(
        char_span_to_tokens([tuple(o) for o in enc["offset_mapping"]], start, start + len(cue))
    )


def draw(g, hint_pos: set[int], title: str, out: Path) -> None:
    used = sorted({n.pos for n in g.nodes})
    xpos = {p: i for i, p in enumerate(used)}
    layers = g.meta["layers"]
    ypos = {L: i for i, L in enumerate(layers)}
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(used) + 2), 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for p in used:
        if p in hint_pos:
            ax.axvspan(xpos[p] - 0.5, xpos[p] + 0.5, color=HINT_BG, zorder=0)
    wmax = max([abs(w) for *_, w in g.edges] or [1.0])
    amax = max(abs(n.attribution) for n in g.nodes)
    # Several features can sit at the same (token, layer); spread them sideways and label only
    # the strongest one, so labels do not overlap.
    spots: dict[tuple[int, int], list[int]] = {}
    for i, n in enumerate(g.nodes):
        spots.setdefault((n.pos, n.layer), []).append(i)
    coords = {}
    for (pos, layer), idxs in spots.items():
        idxs.sort(key=lambda i: -abs(g.nodes[i].attribution))
        for k, i in enumerate(idxs):
            coords[i] = (xpos[pos] + 0.16 * (k - (len(idxs) - 1) / 2), ypos[layer])
    for u, v, w in g.edges:
        (x1, y1), (x2, y2) = coords[u], coords[v]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=POS_C if w > 0 else NEG_C,
            alpha=0.2 + 0.6 * abs(w) / wmax,
            lw=0.6 + 2.4 * abs(w) / wmax,
            zorder=1,
        )
    for i, n in enumerate(g.nodes):
        x, y = coords[i]
        ax.scatter(
            x,
            y,
            s=50 + 450 * abs(n.attribution) / amax,
            color=POS_C if n.attribution > 0 else NEG_C,
            edgecolor=SURFACE,
            lw=1.5,
            zorder=2,
        )
    for (pos, layer), idxs in spots.items():
        top = g.nodes[idxs[0]]
        extra = f" +{len(idxs) - 1}" if len(idxs) > 1 else ""
        ax.annotate(
            f"{top.feature}{extra}",
            (xpos[pos], ypos[layer]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=INK_2,
        )
    ax.set_xticks(
        range(len(used)),
        [g.tokens[p].strip() or "␣" for p in used],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(layers)), [f"layer {L}" for L in layers], fontsize=9)
    ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.text(
        0.01,
        0.01,
        "Node size: |attribution| to logp(hinted) - logp(correct). Blue raises "
        "the hinted answer, orange lowers it. Shaded: hint tokens. Numbers: SAE feature ids.",
        fontsize=7.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_run", type=Path)
    parser.add_argument("analyze_run", type=Path)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--nodes-per-layer", type=int, default=8)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    run_dir = start_run(
        "phase2_graphs", {**cfg, "args": {k: str(v) for k, v in vars(args).items()}}
    )
    cases = pick_cases(args.capture_run, args.n)
    items = {i.id: i for i in medqa_test_not_in_mpib(cfg["capture"]["n_questions"], cfg["seed"])}
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    layers = cfg["capture"]["layers"]
    saes = {
        L: load_sae(
            cfg["capture"]["sae_release"], cfg["capture"]["sae_id"].format(layer=L), model.device
        )
        for L in layers
    }
    diff = pd.read_csv(args.analyze_run / "differential_features.csv")
    diff_set = {(int(r.layer), int(r.feature)) for r in diff.itertuples()}

    summary, recurring = [], Counter()
    for k, case in cases.iterrows():
        item = items[case.item_id]
        cond = next(
            c
            for c in conditions_for(item, cfg["capture"]["framings"], ["wrong"], cfg["seed"])
            if c.name == case.condition
        )
        msgs = build_messages(item, cond)
        g = build_graph(
            model, saes, msgs, "".join(item.options), cond.target, item.answer, args.nodes_per_layer
        )
        hp = hint_positions(model, msgs, cond.cue_text())
        total = sum(abs(n.attribution) for n in g.nodes)
        on_hint = sum(abs(n.attribution) for n in g.nodes if n.pos in hp)
        last = len(g.tokens) - 1
        at_end = sum(abs(n.attribution) for n in g.nodes if n.pos >= last - 4)
        for n in g.nodes:
            recurring[(n.layer, n.feature)] += 1
        summary.append(
            {
                "case": k,
                "job_id": case.job_id,
                "framing": case.framing,
                "metric": g.metric,
                "pull": float(case.pull),
                "n_nodes": len(g.nodes),
                "n_edges": len(g.edges),
                "share_attr_on_hint_tokens": on_hint / total,
                "share_attr_on_last5_tokens": at_end / total,
                "nodes_in_differential_list": sum(
                    (n.layer, n.feature) in diff_set for n in g.nodes
                ),
            }
        )
        (run_dir / f"graph_{k}.json").write_text(
            json.dumps(
                {
                    "job_id": case.job_id,
                    "metric": g.metric,
                    "meta": g.meta,
                    "tokens": g.tokens,
                    "hint_positions": sorted(hp),
                    "nodes": [asdict(n) for n in g.nodes],
                    "edges": g.edges,
                },
                indent=2,
            )
        )
        draw(
            g,
            hp,
            f"Case {k}: {case.framing} hint -> {cond.target} (correct {item.answer}); "
            f"metric {g.metric:.2f}",
            REPO_ROOT / f"reports/figures/phase2_graph_{k}.png",
        )
        logger.info(
            "case %d (%s): metric %.2f, %.0f%% of attribution on hint tokens",
            k,
            case.framing,
            g.metric,
            100 * on_hint / total,
        )
    pd.DataFrame(summary).to_csv(run_dir / "graphs_summary.csv", index=False)
    rec = [{"layer": L, "feature": f, "graphs": c} for (L, f), c in recurring.most_common(20)]
    (run_dir / "recurring_features.json").write_text(json.dumps(rec, indent=2))
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(json.dumps(rec[:10], indent=2))


if __name__ == "__main__":
    main()
