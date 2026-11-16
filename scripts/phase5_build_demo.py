"""Phase 5: assemble the static demo's data files from saved Phase 1-5 outputs (no model needed).

Writes demo/data/index.json (summary numbers, question list, gallery, features) and one
demo/data/q/<item_id>.json per question (the prompt, the model's reasoning and answer, the
judge's verdict and the detector scores for each of the 10 conditions).

Usage: PYTHONPATH=scripts uv run python scripts/phase5_build_demo.py <score_run> [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from phase5_score_runs import oof_scores, stored

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.demo import GALLERY, changed_by_hint, gallery_category, pick_gallery, recall_threshold
from medlens.eval.stats import auroc
from medlens.runs import REPO_ROOT, read_jsonl, start_run

logger = logging.getLogger("phase5_build_demo")
LETTERS = "ABCD"
FEATURES = [(17, 1894), (17, 12508), (22, 2710), (22, 146)]  # Phase 2 causal features


def clean(v):
    """JSON-safe value: NaN and pandas NA become None, numpy scalars become Python ones."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def probs(row: pd.Series) -> dict[str, float]:
    lp = np.array([row[f"lp_{k}"] for k in LETTERS])
    p = np.exp(lp - lp.max())
    return {k: round(float(v), 4) for k, v in zip(LETTERS, p / p.sum(), strict=True)}


def full_scores(cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Scores over ALL 375 questions (not just the demo's), for the threshold and headline
    numbers: out-of-fold scores of the 1,272 training runs, and the final probe on the 375
    no-hint runs (never trained on). Uses only saved activations."""
    p3, probes_run = REPO_ROOT / cfg["phase3_capture"], REPO_ROOT / cfg["phase3_probes"]
    index = pd.read_parquet(p3 / "index.parquet")
    oof = oof_scores(probes_run, index)
    y = np.load(probes_run / "labels.npy")
    none_jobs = index[index.condition == "none"].job_id.tolist()
    vecs = {
        "post": stored(p3, "index.parquet", 0, cfg["layer"]),
        "pre": stored(REPO_ROOT / cfg["phase2_capture"], "readout.parquet", 1, cfg["layer"]),
    }
    out = {}
    for st in ("post", "pre"):
        probe = joblib.load(probes_run / f"probe_{st}.joblib")["model"]
        X = np.stack([vecs[st][j] for j in none_jobs])
        out[st] = {
            "oof": np.array(list(oof[st].values())),
            "y": y,
            "none": probe.predict_proba(X)[:, 1],
        }
    return out


def summary_numbers(full: dict, recall: float) -> dict:
    """Warning threshold and headline numbers, from the same probes the badge uses."""
    out = {}
    for st, d in full.items():
        s, y = d["oof"], d["y"]
        thr = recall_threshold(s, y, recall)
        out[st] = {
            "auroc_out_of_fold": auroc(s[y == 1].tolist(), s[y == 0].tolist()),
            "threshold": thr,
            "recall": float((s[y == 1] >= thr).mean()),
            "flag_rate_ignored_hint": float((s[y == 0] >= thr).mean()),
            "flag_rate_no_hint": float((d["none"] >= thr).mean()),
            "n_runs": int(len(y)),
            "n_no_hint": int(len(d["none"])),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_run", type=Path)
    parser.add_argument("--config", default="configs/phase5.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    run_dir = start_run("phase5_build_demo", {**cfg, "scores_run": str(args.scores_run)})

    p1 = REPO_ROOT / cfg["phase1_run"]
    labels = pd.read_parquet(p1 / "labels.parquet").set_index("job_id")
    texts = {r["job_id"]: r["text"] for r in read_jsonl(p1 / "rollouts.jsonl")}
    scores = pd.read_parquet(args.scores_run / "scores.parquet").set_index("job_id")
    spec = pd.read_csv(REPO_ROOT / cfg["specialties"]).set_index("item_id")
    feats = pd.read_csv(REPO_ROOT / cfg["feature_labels"])

    numbers = summary_numbers(full_scores(cfg), cfg["badge"]["recall"])
    thr = {st: numbers[st]["threshold"] for st in ("post", "pre")}
    hinted = labels[labels.label.isin(["unfaithful", "faithful"])]
    numbers["admitted"] = {
        "followed": len(hinted),
        "admitted": int((hinted.label == "faithful").sum()),
        "rate": float((hinted.label == "faithful").mean()),
    }
    mpib = json.loads((REPO_ROOT / cfg["phase4_mpib"] / "results.json").read_text())
    numbers["injection_auroc"] = mpib["H1"]["post"]["auroc_attacked_run"]
    logger.info("summary numbers: %s", json.dumps(numbers))

    items = medqa_test_not_in_mpib(cfg["n_questions"], cfg["seed"])[: cfg["demo_questions"]]
    out_dir = REPO_ROOT / cfg["demo_dir"] / "data"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "q").mkdir(parents=True)
    questions, gallery_rows = [], []
    for item in items:
        none_pred = clean(labels.loc[f"{item.id}|none", "pred"])
        runs = {}
        for cond in conditions_for(item, cfg["framings"], ["wrong", "gold"], cfg["seed"]):
            job = f"{item.id}|{cond.name}"
            lab, sc = labels.loc[job], scores.loc[job]
            flagged = bool(sc.post_score >= thr["post"])
            pred = clean(lab["pred"])
            runs[cond.name] = {
                "framing": cond.framing,
                "direction": cond.direction,
                "target": cond.target,
                "cue": cond.cue_text(),
                "prompt": build_messages(item, cond)[0]["content"],
                "text": texts[job],
                "pred": pred,
                "stop_reason": clean(lab["stop_reason"]),
                "effect": changed_by_hint(none_pred, pred, cond.target if cond.framing else None),
                "eligible": clean(lab["eligible"]),
                "why_not": clean(lab["why_not"]),
                "followed": clean(lab["followed"]),
                "ack": clean(lab["ack"]),
                "judge_quote": clean(lab["judge_quote"]),
                "label": clean(lab["label"]),
                "post_score": round(float(sc.post_score), 4),
                "post_source": sc.post_source,
                "pre_score": round(float(sc.pre_score), 4),
                "pre_source": sc.pre_source,
                "flagged": flagged,
                "flagged_pre": bool(sc.pre_score >= thr["pre"]),
                "probs": probs(sc),
            }
            if cond.framing:
                cat = gallery_category(runs[cond.name]["label"], cond.direction, flagged)
                if cat:
                    gallery_rows.append(
                        {"item_id": item.id, "condition": cond.name, "category": cat}
                    )
        sp = spec.loc[item.id] if item.id in spec.index else None
        q = {
            "id": item.id,
            "specialty": clean(sp["specialty"]) if sp is not None else None,
            "question": item.question,
            "options": item.options,
            "gold": item.answer,
            "runs": runs,
        }
        (out_dir / "q" / f"{item.id}.json").write_text(json.dumps(q, ensure_ascii=False))
        questions.append(
            {
                "id": item.id,
                "specialty": q["specialty"],
                "preview": item.question[:110].rsplit(" ", 1)[0] + "…",
                "n_flagged": sum(r["flagged"] for r in runs.values()),
                "n_followed": sum(r["label"] in ("unfaithful", "faithful") for r in runs.values()),
            }
        )

    g = pick_gallery(pd.DataFrame(gallery_rows), cfg["gallery"]["per_category"], cfg["seed"])
    shelf_sizes = pd.DataFrame(gallery_rows).category.value_counts().to_dict()
    fl = feats.drop_duplicates(["layer", "feature"]).set_index(["layer", "feature"])
    features = [
        {
            "layer": L,
            "feature": f,
            "label": clean(fl.loc[(L, f), "label"]) if (L, f) in fl.index else None,
            "url": f"https://www.neuronpedia.org/gemma-3-4b/{L}-gemmascope-2-res-16k/{f}",
        }
        for L, f in FEATURES
    ]
    index = {
        "numbers": numbers,
        "badge_recall": cfg["badge"]["recall"],
        "n_questions": len(items),
        "questions": questions,
        "gallery": {
            c: {"title": GALLERY[c], "picks": g[c], "n_total": int(shelf_sizes.get(c, 0))}
            for c in GALLERY
        },
        "features": features,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    (run_dir / "index_summary.json").write_text(
        json.dumps({k: index[k] for k in ("numbers", "gallery", "features")}, indent=2)
    )
    # How the badge behaves on each kind of run in the demo (most were never seen in training).
    flat = pd.DataFrame(
        [
            {
                "condition": c,
                "label": r["label"] or "not scored",
                "effect": r["effect"],
                "flagged": r["flagged"],
                "source": r["post_source"],
            }
            for qf in (out_dir / "q").glob("*.json")
            for c, r in json.loads(qf.read_text())["runs"].items()
        ]
    )
    flat["kind"] = flat.condition.str.replace(r"^\w+_(wrong|gold)$", r"hint_\1", regex=True)
    by = flat.groupby(["kind", "label"]).flagged.agg(["size", "mean"]).reset_index()
    by.to_csv(run_dir / "badge_by_kind.csv", index=False)
    logger.info("badge by kind of run:\n%s", by.to_string(index=False))
    index["badge_by_kind"] = by.to_dict("records")
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    size = sum(p.stat().st_size for p in out_dir.rglob("*.json"))
    logger.info("wrote %d questions to %s (%.1f MB)", len(items), out_dir, size / 1e6)


if __name__ == "__main__":
    main()
