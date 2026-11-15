"""Phase 2: which SAE features separate silent hint-following from hint resistance?

Inputs: a phase2_capture run (readouts + features at 3 positions for none/sham/4 wrong hints).

1. Readout check: does the prefilled-answer pull toward the hinted letter predict the Phase 1
   step-by-step label (unfaithful vs unaffected)? Reported as AUROC per framing.
2. Differential features: per question, delta = feature(hint) - feature(sham), at the
   "final" and "gen_start" positions. Features are ranked by how well their delta separates
   unfaithful from unaffected cases (AUROC) and by the mean difference, pooled over framings,
   with a per-framing consistency count.
3. Cue features: features that fire on the hint's own tokens far more than on the sham
   sentence's tokens, per framing.

Usage: uv run python scripts/phase2_analyze.py outputs/phase2_capture/<run>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from medlens.eval.stats import auroc
from medlens.interp.features import POSITIONS
from medlens.runs import start_run

FRAMINGS = ["metadata", "explicit", "authority", "patient"]


def load(run: Path, layers: list[int]) -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
    r = pd.read_parquet(run / "readout.parquet").reset_index(drop=True)
    feats = {
        layer: load_file(str(run / f"features_L{layer}.safetensors"))["features"].float()
        for layer in layers
    }
    return r, feats


def readout_check(r: pd.DataFrame) -> dict:
    out = {}
    sham = r[r.condition == "sham"].set_index("item_id")
    base = r[r.condition == "none"].set_index("item_id")
    out["none_readout_vs_phase1_cot_same_answer"] = float(
        (base.readout_pred == base.p1_pred).mean()
    )
    for f in FRAMINGS:
        h = r[(r.condition == f"{f}_wrong") & (r.p1_eligible == True)].copy()  # noqa: E712
        h["lp_target"] = [row[f"lp_{row.target}"] for _, row in h.iterrows()]
        h["lp_target_sham"] = [
            sham.loc[i, f"lp_{t}"] for i, t in zip(h.item_id, h.target, strict=True)
        ]
        h["pull"] = h.lp_target - h.lp_target_sham
        unf = h[h.p1_label == "unfaithful"].pull.tolist()
        una = h[h.p1_label == "unaffected"].pull.tolist()
        out[f] = {
            "n_unfaithful": len(unf),
            "n_unaffected": len(una),
            "pull_auroc_unfaithful_vs_unaffected": auroc(unf, una),
            "lp_target_auroc_unfaithful_vs_unaffected": auroc(
                h[h.p1_label == "unfaithful"].lp_target.tolist(),
                h[h.p1_label == "unaffected"].lp_target.tolist(),
            ),
            "readout_follow_auroc": auroc(
                (h[h.p1_label == "unfaithful"].readout_pred == h[h.p1_label == "unfaithful"].target)
                .astype(float)
                .tolist(),
                (h[h.p1_label == "unaffected"].readout_pred == h[h.p1_label == "unaffected"].target)
                .astype(float)
                .tolist(),
            ),
            "readout_follows_hint_rate": float((h.readout_pred == h.target).mean()),
            "phase1_follow_rate": float(h.p1_followed.astype(bool).mean()),
            "readout_and_phase1_agree_on_following": float(
                ((h.readout_pred == h.target) == h.p1_followed.astype(bool)).mean()
            ),
        }
    return out


def differential(r: pd.DataFrame, feats: dict[int, torch.Tensor], top: int) -> pd.DataFrame:
    sham_idx = r[r.condition == "sham"].set_index("item_id").index
    sham_row = {item: i for i, item in zip(r[r.condition == "sham"].index, sham_idx, strict=True)}
    rows = []
    for layer, F in feats.items():
        for pi, pos in enumerate(POSITIONS[:2]):  # final, gen_start (hint is handled below)
            pooled_u, pooled_a, per_framing = [], [], {}
            for f in FRAMINGS:
                h = r[(r.condition == f"{f}_wrong") & (r.p1_eligible == True)]  # noqa: E712
                u = h[h.p1_label == "unfaithful"]
                a = h[h.p1_label == "unaffected"]
                du = F[u.index, pi] - F[[sham_row[i] for i in u.item_id], pi]
                da = F[a.index, pi] - F[[sham_row[i] for i in a.item_id], pi]
                pooled_u.append(du)
                pooled_a.append(da)
                per_framing[f] = (du.mean(0) - da.mean(0)).numpy()
            DU, DA = torch.cat(pooled_u), torch.cat(pooled_a)
            diff = (DU.mean(0) - DA.mean(0)).numpy()
            cand = np.argsort(-diff)[: top * 3]  # AUROC only for the strongest candidates
            for idx in cand:
                consistent = sum(per_framing[f][idx] > 0 for f in FRAMINGS)
                rows.append(
                    {
                        "layer": layer,
                        "position": pos,
                        "feature": int(idx),
                        "mean_delta_unfaithful": float(DU[:, idx].mean()),
                        "mean_delta_unaffected": float(DA[:, idx].mean()),
                        "diff": float(diff[idx]),
                        "auroc": auroc(DU[:, idx].tolist(), DA[:, idx].tolist()),
                        "frac_unfaithful_active": float((DU[:, idx] > 0).float().mean()),
                        "framings_with_positive_diff": int(consistent),
                        **{f"diff_{f}": float(per_framing[f][idx]) for f in FRAMINGS},
                    }
                )
    df = pd.DataFrame(rows)
    return (
        df.sort_values(["layer", "position", "diff"], ascending=[True, True, False])
        .groupby(["layer", "position"])
        .head(top)
        .reset_index(drop=True)
    )


def cue_features(r: pd.DataFrame, feats: dict[int, torch.Tensor], top: int) -> pd.DataFrame:
    hi = POSITIONS.index("hint")
    sham = r[r.condition == "sham"].index
    rows = []
    for layer, F in feats.items():
        s = F[sham, hi].mean(0)
        for f in FRAMINGS:
            h = r[r.condition == f"{f}_wrong"].index
            m = F[h, hi].mean(0)
            frac = (F[h, hi] > 0).float().mean(0)
            d = (m - s).numpy()
            for idx in np.argsort(-d)[:top]:
                rows.append(
                    {
                        "layer": layer,
                        "framing": f,
                        "feature": int(idx),
                        "mean_on_hint": float(m[idx]),
                        "mean_on_sham": float(s[idx]),
                        "frac_hints_active": float(frac[idx]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_run", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 17, 22, 29])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    run_dir = start_run(
        "phase2_analyze",
        {"capture_run": str(args.capture_run), "top": args.top, "layers": args.layers, "seed": 0},
    )
    r, feats = load(args.capture_run, args.layers)
    rc = readout_check(r)
    (run_dir / "readout_check.json").write_text(json.dumps(rc, indent=2))
    diff = differential(r, feats, args.top)
    diff.to_csv(run_dir / "differential_features.csv", index=False)
    cues = cue_features(r, feats, args.top)
    cues.to_csv(run_dir / "cue_features.csv", index=False)
    print(json.dumps(rc, indent=2))
    show = diff.sort_values("auroc", ascending=False).head(25)
    print(
        show[
            [
                "layer",
                "position",
                "feature",
                "diff",
                "auroc",
                "frac_unfaithful_active",
                "framings_with_positive_diff",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )


if __name__ == "__main__":
    main()
