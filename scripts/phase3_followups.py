"""Phase 3 follow-ups: is the best probe reading hint-following, or something simpler?

1. Per hint type: AUROC within each framing (rules out "it just detects the strong framings").
2. Both-wrong subset: unfaithful answers are always wrong, so compare them only with unaffected
   runs whose answer is also wrong (rules out "it just detects wrong answers").
3. Question-only control: the same probe recipe, but fed the NO-HINT prompt's activations for
   the same question plus the hint type. If this matches the real probe, the probe is reading
   how hint-prone the question is, not the model's processing of this hint.

Usage: uv run python scripts/phase3_followups.py <probes_run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from phase3_probes import POST, PRE, active_columns, load_inputs, matrix, text_cv
from safetensors.torch import load_file

from medlens.eval.stats import auroc
from medlens.probes.core import grouped_cv, summarize
from medlens.runs import REPO_ROOT, start_run


def sub_auroc(p: np.ndarray, y: np.ndarray, m: np.ndarray) -> float | None:
    return auroc(p[m & (y == 1)].tolist(), p[m & (y == 0)].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes_run", type=Path)
    parser.add_argument("--ack-run", type=Path, help="post-reasoning capture of acknowledged runs")
    args = parser.parse_args()
    pcfg_all = yaml.safe_load((args.probes_run / "config.yaml").read_text())
    pcfg = pcfg_all["probes"]
    run_dir = start_run(
        "phase3_followups", {"probes_run": str(args.probes_run), "ack_run": str(args.ack_run)}
    )
    pre_idx, post_idx, T = load_inputs(
        REPO_ROOT / pcfg_all["phase2_capture"], REPO_ROOT / pcfg_all["post_run"]
    )
    post_h = post_idx[post_idx.p1_label.isin(["unfaithful", "unaffected"])]
    y = (post_h.p1_label == "unfaithful").astype(int).to_numpy()
    assert np.array_equal(y, np.load(args.probes_run / "labels.npy"))
    g = post_h.item_id.to_numpy()
    framing = post_h.framing.to_numpy()
    wrong = (post_h.p1_pred != post_h.gold).to_numpy()

    table = pd.read_csv(args.probes_run / "probe_table.csv")
    hint_types = sorted(set(framing))
    out: dict = {"n_unaffected_wrong": int((wrong & (y == 0)).sum())}

    def per_type(p: np.ndarray) -> dict:
        d = {f: sub_auroc(p, y, framing == f) for f in hint_types}
        d["mean_within_type"] = float(np.mean([d[f] for f in hint_types]))
        return d

    # No-hint prompts, and whether the model's own no-hint answer was right.
    none = post_idx[post_idx.condition == "none"]
    gn = none.item_id.to_numpy()
    none_wrong = (none.p1_pred != none.gold).to_numpy()

    best = {
        "pre_end_of_prompt": table[table.position == "gen_start"].iloc[0].probe,
        "pre_forced_final": table[table.position == "final"].iloc[0].probe,
        "post_reasoning": table[table.stage == "post"].iloc[0].probe,
    }
    for key, name in best.items():
        stage, pos, kind, layer = name.split(":")
        layers = [9, 17, 22, 29] if layer == "all" else [int(layer[1:])]
        idx = pre_idx if stage == "pre" else post_idx
        pi = (PRE if stage == "pre" else POST)[pos]
        row_of = {j: i for i, j in enumerate(idx.job_id)}
        rows = [row_of[j] for j in post_h.job_id]
        nrows = [row_of[j] for j in none.job_id]
        X = np.concatenate([matrix(T, stage, kind, L, pi, rows) for L in layers], 1)
        Xn = np.concatenate([matrix(T, stage, kind, L, pi, nrows) for L in layers], 1)
        # Acknowledged runs (followed the hint AND said so): pre-reasoning ones are in the
        # Phase 2 capture; post-reasoning ones come from a separate capture (--ack-run).
        Xa, ga = None, np.array([])
        if stage == "pre":
            ack = pre_idx[pre_idx.p1_label == "faithful"]
            Xa = np.concatenate(
                [matrix(T, stage, kind, L, pi, ack.index.tolist()) for L in layers], 1
            )
            ga = ack.item_id.to_numpy()
        elif args.ack_run:
            ack = pd.read_parquet(args.ack_run / "index.parquet")
            Ta = {}
            for L in layers:
                Ta[("post", kind, L)] = load_file(
                    str(
                        args.ack_run
                        / f"{'resid' if kind == 'resid' else 'features'}_L{L}.safetensors"
                    )
                )["resid" if kind == "resid" else "features"]
            Xa = np.concatenate(
                [matrix(Ta, stage, kind, L, pi, list(range(len(ack)))) for L in layers], 1
            )
            ga = ack.item_id.to_numpy()
        n_none = len(Xn)
        if Xa is not None:
            assert Xa.shape[1] == Xn.shape[1], "acknowledged-run features must match"
            Xn, gx = np.concatenate([Xn, Xa]), np.concatenate([gn, ga])
        else:
            gx = gn
        if kind == "sae":
            cols = active_columns(X)
            X, Xn = X[:, cols], Xn[:, cols]
        r = grouped_cv(
            X, y, g, pcfg["C_grid"], pcfg["n_folds"], pcfg_all["seed"], X_extra=Xn, groups_extra=gx
        )
        none_scores, ack_scores = r.extra_scores[:n_none], r.extra_scores[n_none:]
        p = r.scores
        thr = np.quantile(p[y == 1], 0.2)  # catches 80% of unfaithful runs
        res: dict = {"probe": name, "auroc": summarize(p, y, g, 10)["auroc"]}
        res["per_hint_type"] = per_type(p)
        res["both_wrong_subset"] = sub_auroc(p, y, wrong)
        res["no_hint_flag_rate_at_80pct_recall"] = {
            "model_right_without_hint": float((none_scores[~none_wrong] >= thr).mean()),
            "model_wrong_without_hint": float((none_scores[none_wrong] >= thr).mean()),
            "n_right": int((~none_wrong).sum()),
            "n_wrong": int(none_wrong.sum()),
        }
        if len(ack_scores):
            res["acknowledged_runs"] = {
                "n": int(len(ack_scores)),
                "flag_rate_at_80pct_recall": float((ack_scores >= thr).mean()),
                "median_score": float(np.median(ack_scores)),
                "median_score_unfaithful": float(np.median(p[y == 1])),
                "median_score_unaffected": float(np.median(p[y == 0])),
            }

        # Question-only control: no-hint activations of the same question + hint-type one-hot.
        none_row = {r.item_id: i for i, r in enumerate(idx.itertuples()) if r.condition == "none"}
        qrows = [none_row[q] for q in g]
        Xq = np.concatenate([matrix(T, stage, kind, L, pi, qrows) for L in layers], 1)
        if kind == "sae":
            Xq = Xq[:, active_columns(Xq)]
        onehot = (framing[:, None] == np.array(hint_types)[None]).astype(float)
        rq = grouped_cv(
            np.concatenate([Xq, onehot], 1), y, g, pcfg["C_grid"], pcfg["n_folds"], pcfg_all["seed"]
        )
        res["question_only_control"] = summarize(rq.scores, y, g, pcfg["bootstrap"])
        res["question_only_control"]["per_hint_type"] = per_type(rq.scores)
        out[key] = res

    r1 = grouped_cv(onehot, y, g, pcfg["C_grid"], pcfg["n_folds"], pcfg_all["seed"])
    out["hint_type_only"] = summarize(r1.scores, y, g, pcfg["bootstrap"])

    # Baselines within each hint type.
    judge = pd.read_parquet(REPO_ROOT / pcfg_all["judge_run"] / "judge_scores.parquet")
    jp = judge.set_index("job_id").loc[post_h.job_id, "judge_yes_rate"].to_numpy()
    tp = text_cv(post_h.reasoning.tolist(), y, g, pcfg["n_folds"], pcfg_all["seed"])
    pre_h = pre_idx.set_index("job_id").loc[post_h.job_id]
    rp = np.exp([row[f"lp_{row.target}"] for _, row in pre_h.iterrows()])
    out["baselines_per_hint_type"] = {
        "llm_judge_yes_rate": per_type(jp),
        "text_tfidf_reasoning_only": per_type(tp),
        "phase2_readout_P_hinted_letter (uses hint letter)": per_type(rp),
    }
    (run_dir / "followups.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
