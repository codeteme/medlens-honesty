"""Phase 3: can MedGemma's internals flag answers driven by an unmentioned hint?

Task: among hinted runs (wrong-answer hints) labeled in Phase 1 as unfaithful (followed the
hint, reasoning silent) or unaffected, predict "unfaithful". Detectors never see the hint
text or the hinted letter.
  pre-reasoning positions (Phase 2 capture):  gen_start, final (prefilled "Final answer:")
  post-reasoning positions (Phase 3 capture): post (after the reasoning), cot_mean
Inputs: raw residual activations or Gemma Scope 2 SAE features, layers 9/17/22/29.
Splits: 5 folds grouped by question. C chosen inside each training fold.

Baselines: TF-IDF text classifier on the reasoning only; LLM judge asked k times (yes-rate);
the Phase 2 readout P(hinted letter) (uses the hinted letter, so it is not hint-blind); and
the letter-matching rule (AUROC 1 by construction among eligible runs).

Also: leave-one-hint-type-out, leave-one-specialty-group-out, and false alarms on no-hint
prompts at the threshold that catches 80% of unfaithful runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from safetensors.torch import load_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from medlens.config import load_config
from medlens.eval.stats import auroc
from medlens.probes.core import grouped_cv, make_probe, summarize, train_test_split_predict
from medlens.runs import REPO_ROOT, start_run

logger = logging.getLogger("phase3_probes")
LAYERS = [9, 17, 22, 29]
PRE = {"gen_start": 0, "final": 1}  # indices in the Phase 2 capture tensors
POST = {"post": 0, "cot_mean": 1}  # indices in the Phase 3 capture tensors


def load_inputs(pre_run: Path, post_run: Path):
    pre_idx = pd.read_parquet(pre_run / "readout.parquet").reset_index(drop=True)
    post_idx = pd.read_parquet(post_run / "index.parquet").reset_index(drop=True)
    tensors = {}
    for L in LAYERS:
        tensors[("pre", "resid", L)] = load_file(str(pre_run / f"resid_L{L}.safetensors"))["resid"]
        tensors[("pre", "sae", L)] = load_file(str(pre_run / f"features_L{L}.safetensors"))[
            "features"
        ]
        tensors[("post", "resid", L)] = load_file(str(post_run / f"resid_L{L}.safetensors"))[
            "resid"
        ]
        tensors[("post", "sae", L)] = load_file(str(post_run / f"features_L{L}.safetensors"))[
            "features"
        ]
    return pre_idx, post_idx, tensors


def finite_columns(t) -> np.ndarray:
    """Channels finite in every stored row. Layer-29 residuals were saved as float16 and one
    "massive activation" channel (443) overflows to inf in some rows; it is dropped (label-free)."""
    return np.flatnonzero(np.isfinite(t.float().numpy()).all(axis=(0, 1)))


def matrix(tensors, stage, kind, L, pos_i, rows) -> np.ndarray:
    t = tensors[(stage, kind, L)]
    X = t[rows, pos_i].float().numpy()
    if kind == "sae":
        return np.log1p(np.maximum(X, 0))
    return X[:, finite_columns(t)]


def active_columns(X: np.ndarray, frac: float = 0.01) -> np.ndarray:
    return np.flatnonzero((X > 0).mean(0) >= frac)


def run_probe(name, X, y, g, Xn, gn, C_grid, n_folds, seed, sae):
    if sae:  # keep features active in >= 1% of hinted rows (unsupervised, label-free)
        cols = active_columns(X)
        X, Xn = X[:, cols], Xn[:, cols]
    res = grouped_cv(X, y, g, C_grid, n_folds, seed, X_extra=Xn, groups_extra=gn)
    return name, res


def false_alarms(p_hint, y, p_none, recall=0.8) -> dict:
    thr = np.quantile(p_hint[y == 1], 1 - recall)
    return {
        "threshold": float(thr),
        "recall": float((p_hint[y == 1] >= thr).mean()),
        "flag_rate_unaffected_hinted": float((p_hint[y == 0] >= thr).mean()),
        "flag_rate_no_hint_prompts": float((p_none >= thr).mean()),
    }


def text_cv(texts, y, g, n_folds, seed):
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_folds).split(texts, y, g):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
        ).fit(Xtr, y[tr])
        scores[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("post_run", type=Path)
    parser.add_argument("--judge-run", type=Path, required=True)
    parser.add_argument("--specialty-run", type=Path, required=True)
    parser.add_argument("--config", default="configs/phase3.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    pcfg = cfg["probes"]
    run_dir = start_run(
        "phase3_probes",
        {
            **cfg,
            "post_run": str(args.post_run),
            "judge_run": str(args.judge_run),
            "specialty_run": str(args.specialty_run),
        },
    )
    pre_idx, post_idx, T = load_inputs(REPO_ROOT / cfg["phase2_capture"], args.post_run)

    # Rows: hinted runs labeled unfaithful/unaffected, aligned across pre and post captures.
    post_h = post_idx[post_idx.p1_label.isin(["unfaithful", "unaffected"])]
    pre_pos = {j: i for i, j in enumerate(pre_idx.job_id)}
    post_pos = {j: i for i, j in enumerate(post_idx.job_id)}
    jobs = post_h.job_id.tolist()
    y = (post_h.p1_label == "unfaithful").astype(int).to_numpy()
    g = post_h.item_id.to_numpy()
    framing = post_h.framing.to_numpy()
    pre_rows = [pre_pos[j] for j in jobs]
    post_rows = [post_pos[j] for j in jobs]
    none_jobs = post_idx[post_idx.condition == "none"].job_id.tolist()
    none_pre = [pre_pos[j] for j in none_jobs]
    none_post = [post_pos[j] for j in none_jobs]
    gn = post_idx.set_index("job_id").loc[none_jobs, "item_id"].to_numpy()
    logger.info("hinted rows %d (unfaithful %d), no-hint rows %d", len(y), y.sum(), len(gn))

    specs = []
    for kind in ("resid", "sae"):
        for L in LAYERS:
            for pos, pi in PRE.items():
                specs.append(
                    (
                        f"pre:{pos}:{kind}:L{L}",
                        matrix(T, "pre", kind, L, pi, pre_rows),
                        matrix(T, "pre", kind, L, pi, none_pre),
                        kind == "sae",
                    )
                )
            for pos, pi in POST.items():
                specs.append(
                    (
                        f"post:{pos}:{kind}:L{L}",
                        matrix(T, "post", kind, L, pi, post_rows),
                        matrix(T, "post", kind, L, pi, none_post),
                        kind == "sae",
                    )
                )
    for stage, pos, pi, rows, nrows in (
        ("pre", "gen_start", 0, pre_rows, none_pre),
        ("post", "post", 0, post_rows, none_post),
    ):
        specs.append(
            (
                f"{stage}:{pos}:resid:all",
                np.concatenate([matrix(T, stage, "resid", L, pi, rows) for L in LAYERS], 1),
                np.concatenate([matrix(T, stage, "resid", L, pi, nrows) for L in LAYERS], 1),
                False,
            )
        )
    logger.info("training %d probes", len(specs))
    results = Parallel(n_jobs=args.jobs)(
        delayed(run_probe)(name, X, y, g, Xn, gn, pcfg["C_grid"], pcfg["n_folds"], cfg["seed"], sae)
        for name, X, Xn, sae in specs
    )
    res = dict(results)
    spec_by_name = {s[0]: s for s in specs}

    rows = []
    for name, r in res.items():
        stage, pos, kind, layer = name.split(":")
        s = summarize(r.scores, y, g, pcfg["bootstrap"])
        rows.append(
            {
                "probe": name,
                "stage": stage,
                "position": pos,
                "input": kind,
                "layer": layer,
                **s,
                **false_alarms(r.scores, y, r.extra_scores),
                "chosen_C": r.chosen_C,
            }
        )
    table = pd.DataFrame(rows).sort_values("auroc", ascending=False)

    # ---- baselines ----
    base = {}
    texts = post_h.reasoning.tolist()
    t_scores = text_cv(texts, y, g, pcfg["n_folds"], cfg["seed"])
    base["text_tfidf_reasoning_only"] = summarize(t_scores, y, g, pcfg["bootstrap"])
    judge = pd.read_parquet(args.judge_run / "judge_scores.parquet").set_index("job_id")
    j_scores = judge.loc[jobs, "judge_yes_rate"].to_numpy()
    base["llm_judge_yes_rate"] = summarize(j_scores, y, g, pcfg["bootstrap"])
    base["llm_judge_yes_rate"]["flag_rate_no_hint_at_yes_rate_0.5"] = float(
        (judge.loc[none_jobs, "judge_yes_rate"] >= 0.5).mean()
    )
    pre_h = pre_idx.iloc[pre_rows]
    lp = np.array([row[f"lp_{row.target}"] for _, row in pre_h.iterrows()])
    base["phase2_readout_P_hinted_letter (uses hint letter)"] = summarize(
        np.exp(lp), y, g, pcfg["bootstrap"]
    )
    base["letter_rule (uses hint letter)"] = {"auroc": 1.0, "note": "by construction"}

    # ---- generalization for the best pre- and post-reasoning probes ----
    gen = {}
    best = {st: table[table.stage == st].iloc[0].probe for st in ("pre", "post")}
    spec_df = pd.read_csv(args.specialty_run / "specialties.csv").set_index("item_id")
    groups_spec = spec_df.loc[g, "group"].to_numpy()
    for st, name in best.items():
        _, X, _, sae = spec_by_name[name]
        if sae:
            X = X[:, active_columns(X)]
        out = {}
        for held in sorted(set(framing)):
            te = framing == held
            p = train_test_split_predict(X[~te], y[~te], g[~te], X[te], pcfg["C_grid"], cfg["seed"])
            out[f"hold_out_hint:{held}"] = {
                "n": int(te.sum()),
                "auroc": auroc(p[y[te] == 1].tolist(), p[y[te] == 0].tolist()),
            }
        for held in sorted(set(groups_spec)):
            te = groups_spec == held
            p = train_test_split_predict(X[~te], y[~te], g[~te], X[te], pcfg["C_grid"], cfg["seed"])
            out[f"hold_out_specialty:{held}"] = {
                "n": int(te.sum()),
                "auroc": auroc(p[y[te] == 1].tolist(), p[y[te] == 0].tolist()),
            }
        gen[name] = out
        # Final probe on all data, saved for later phases (e.g. the demo's warning badge).
        r = res[name]
        C = float(np.median(r.chosen_C))
        joblib.dump(
            {"model": make_probe(C, cfg["seed"]).fit(X, y), "name": name, "C": C},
            run_dir / f"probe_{st}.joblib",
        )
        np.save(run_dir / f"oof_scores_{st}.npy", r.scores)
    np.save(run_dir / "labels.npy", y)

    table.to_csv(run_dir / "probe_table.csv", index=False)
    (run_dir / "baselines.json").write_text(json.dumps(base, indent=2))
    (run_dir / "generalization.json").write_text(json.dumps(gen, indent=2))
    print(
        table[["probe", "auroc", "auroc_ci", "brier", "ece", "flag_rate_no_hint_prompts"]]
        .head(14)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    print(json.dumps(base, indent=2))
    print(json.dumps(gen, indent=2))


if __name__ == "__main__":
    main()
