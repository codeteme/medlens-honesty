"""Phase 4: can we stop MedGemma 4B from silently following hints?

Modes:
  directions  Learn "hint-following" directions from dev questions: the mean activation of
              unfaithful runs minus that of unaffected runs, before reasoning (Phase 2 capture,
              forced "Final answer:") and after it (Phase 3 capture), at layers 17 and 22.
  screen      One-pass readout (P of each letter at a forced "Final answer:") for every setting
              in the config, on dev questions. Cheap; used to shortlist fixes.
  cot         Real step-by-step answers for chosen settings, on dev or test questions. Then the
              Phase 1 judge (Qwen 27B on Tinker) checks whether answers that followed the hint
              admit it.

Every internal fix acts at every token, so it needs no knowledge of where the hint is.
Questions are split in half (dev/test) by a seeded hash; test is used only for confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from safetensors.torch import load_file

from medlens.blackbox.hints import Condition, conditions_for
from medlens.blackbox.judge import judge_messages, parse_judge
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.data.schema import MCQItem, format_mcq
from medlens.eval.answers import COT_INSTRUCTION, extract_answer
from medlens.eval.stats import auroc, mcnemar_exact, wilson_interval
from medlens.interp.features import capture
from medlens.interp.interventions import scale_features
from medlens.interp.sae_eval import load_sae
from medlens.interp.steering import (
    ablate_direction,
    add_direction,
    compose,
    mean_diff_direction,
    random_direction,
)
from medlens.models.base import CostTracker, Message
from medlens.models.cache import ResponseCache
from medlens.models.hf_local import HFLocalModel
from medlens.models.tinker_model import TinkerModel
from medlens.runs import REPO_ROOT, read_jsonl, start_run, write_jsonl

logger = logging.getLogger("phase4_fix")
DIR_LAYERS = [17, 22]
KEEP = ["unfaithful", "unaffected"]


# ---- shared helpers ----------------------------------------------------------------------
def split_of(item_id: str, seed: int, dev_fraction: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest(), 16) / 2**256
    return "dev" if h < dev_fraction else "test"


def messages_for(item: MCQItem, cond: Condition, prompt_fix: str | None) -> list[Message]:
    """Phase 1 prompt (question, options, optional cue, instruction), optionally with the
    prompt-only fix inserted before the instruction."""
    parts = [format_mcq(item)]
    if cond.cue_text():
        parts.append(cond.cue_text())
    if prompt_fix:
        parts.append(prompt_fix)
    parts.append(COT_INSTRUCTION)
    return [{"role": "user", "content": "\n\n".join(parts)}]


def load_directions(run: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(run / "directions.npz")
    return {k: z[k] for k in z.files}, json.loads((run / "directions.json").read_text())


def build_hooks(spec: dict, dirs: dict, info: dict, cfg: dict, sae_cache: dict, model) -> dict:
    """Turn a setting from the config into {(layer, "resid_post"): hook}."""
    per_layer: dict[int, list] = {}
    for op in spec.get("ops", []):
        L = op["layer"]
        if op["kind"] == "sae":
            if L not in sae_cache:
                sae_cache[L] = load_sae(
                    cfg["sae_release"], cfg["sae_id"].format(layer=L), model.device
                )
            feats = pick_attr_features(REPO_ROOT / cfg["attr_run"], L, op["k"])
            per_layer.setdefault(L, []).append(scale_features(sae_cache[L], feats, 0.0))
            continue
        name = op["dir"]
        if name.startswith("random"):
            d = random_direction(len(dirs[f"pre_L{L}"]), seed=1000 * cfg["seed"] + int(name[6:]))
            gap = info["gap"][f"pre_L{L}"]
        else:
            d, gap = dirs[f"{name}_L{L}"], info["gap"][f"{name}_L{L}"]
        if op["kind"] == "ablate":
            per_layer.setdefault(L, []).append(ablate_direction(d))
        else:
            per_layer.setdefault(L, []).append(add_direction(d, op["gaps"] * gap))
    return {(L, "resid_post"): compose(*fns) for L, fns in per_layer.items()}


def pick_attr_features(attr_run: Path, layer: int, k: int) -> list[int]:
    """Top-k attribution-ranked hint-token features (Phase 2), consistent in all 4 framings."""
    d = pd.read_csv(attr_run / "attribution_features.csv")
    d = d[(d.layer == layer) & (d.position == "hint") & (d.framings_with_positive_diff == 4)]
    return d.sort_values("auroc", ascending=False).feature.head(k).astype(int).tolist()


def items_and_split(cfg: dict) -> tuple[list[MCQItem], dict[str, str]]:
    if cfg.get("items_file"):  # Phase 8: the same items, exported so MPIB stays on the Mac
        rows = json.loads((REPO_ROOT / cfg["items_file"]).read_text())
        items = [MCQItem(**d) for d in rows][: cfg["n_questions"]]
    else:
        items = medqa_test_not_in_mpib(cfg["n_questions"], cfg["seed"])
    return items, {i.id: split_of(i.id, cfg["seed"], cfg["dev_fraction"]) for i in items}


# ---- mode: directions --------------------------------------------------------------------
def directions_mode(cfg: dict, run_dir: Path) -> None:
    _, split = items_and_split(cfg)
    dirs, info = {}, {"gap": {}, "n": {}, "dev_auroc_of_projection": {}}
    sources = {
        "pre": (REPO_ROOT / cfg["phase2_capture"], "readout.parquet", 1),  # forced "Final answer:"
        "post": (REPO_ROOT / cfg["phase3_capture"], "index.parquet", 0),  # after reasoning
    }
    for name, (run, index_file, pos) in sources.items():
        idx = pd.read_parquet(run / index_file).reset_index(drop=True)
        keep = idx[idx.p1_label.isin(KEEP) & (idx.item_id.map(split) == "dev")]
        y = (keep.p1_label == "unfaithful").to_numpy()
        for L in DIR_LAYERS:
            R = load_file(str(run / f"resid_L{L}.safetensors"))["resid"]
            X = R[keep.index.to_numpy(), pos].float().numpy()
            d, gap = mean_diff_direction(X[y], X[~y])
            proj = X @ d
            dirs[f"{name}_L{L}"] = d.astype(np.float32)
            info["gap"][f"{name}_L{L}"] = gap
            info["n"][f"{name}_L{L}"] = {"unfaithful": int(y.sum()), "unaffected": int((~y).sum())}
            info["dev_auroc_of_projection"][f"{name}_L{L}"] = auroc(
                proj[y].tolist(), proj[~y].tolist()
            )
            info.setdefault("mean_resid_norm", {})[f"{name}_L{L}"] = float(
                np.linalg.norm(X, axis=1).mean()
            )
    info["n_dev_questions"] = sum(v == "dev" for v in split.values())
    info["cosine_pre_post"] = {
        f"L{L}": float(dirs[f"pre_L{L}"] @ dirs[f"post_L{L}"]) for L in DIR_LAYERS
    }
    np.savez(run_dir / "directions.npz", **dirs)
    (run_dir / "directions.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ---- mode: screen ------------------------------------------------------------------------
def screen_mode(cfg: dict, run_dir: Path, dir_run: Path, settings: list[str]) -> None:
    scfg = cfg["screen"]
    items, split = items_and_split(cfg)
    by_id = {i.id: i for i in items}
    labels = pd.read_parquet(REPO_ROOT / cfg["phase1_labels"])
    labels = labels[labels.item_id.map(split) == "dev"]
    hinted = pd.concat(
        [
            g.sample(n=min(scfg["n_hinted_per_framing"], len(g)), random_state=cfg["seed"])
            for _, g in labels[
                labels.label.isin(KEEP) & labels.condition.str.endswith("_wrong")
            ].groupby("framing")
        ]
    )
    none = labels[labels.condition == "none"].sample(
        n=min(scfg["n_none"], (labels.condition == "none").sum()), random_state=cfg["seed"]
    )
    jobs = []
    for r in pd.concat([hinted, none]).itertuples():
        item = by_id[r.item_id]
        cond = next(
            c
            for c in conditions_for(item, cfg["framings"], ["wrong"], cfg["seed"], sham=False)
            if c.name == r.condition
        )
        jobs.append({"job_id": r.job_id, "item": item, "cond": cond, "p1_label": r.label})
    logger.info("screen: %d hinted + %d no-hint prompts", len(hinted), len(none))

    dirs, info = load_directions(dir_run)
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    sae_cache: dict = {}
    rows, t0 = [], time.time()
    for name in settings:
        spec = cfg["settings"][name]
        hooks = build_hooks(spec, dirs, info, cfg, sae_cache, model) or None
        fix = cfg["prompt_fix_text"] if spec.get("prompt") else None
        for j in jobs:
            msgs = messages_for(j["item"], j["cond"], fix)
            cap = capture(model, msgs, "".join(j["item"].options), [], None, replace=hooks)
            lp = cap.letter_logprobs
            rows.append(
                {
                    "setting": name,
                    "job_id": j["job_id"],
                    "item_id": j["item"].id,
                    "hinted": j["cond"].is_hint,
                    "framing": j["cond"].framing,
                    "target": j["cond"].target,
                    "gold": j["item"].answer,
                    "p1_label": j["p1_label"],
                    "pred": max(lp, key=lp.get),
                    "lp_target": lp[j["cond"].target] if j["cond"].target else None,
                    "lp_gold": lp[j["item"].answer],
                }
            )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("setting %s done (%.0fs)", name, time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "screen_rows.parquet")
    summary = summarize_screen(df, scfg["max_accuracy_drop"])
    summary.to_csv(run_dir / "screen_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def summarize_screen(df: pd.DataFrame, max_drop: float) -> pd.DataFrame:
    base = df[df.setting == "baseline"].set_index("job_id")
    out = []
    for name, g in df.groupby("setting", sort=False):
        g = g.set_index("job_id")
        h, n = g[g.hinted], g[~g.hinted]
        bh, bn = base.loc[h.index], base.loc[n.index]
        follow, follow0 = (h.pred == h.target), (bh.pred == bh.target)
        acc, acc0 = (n.pred == n.gold), (bn.pred == bn.gold)
        out.append(
            {
                "setting": name,
                "follow_rate": follow.mean(),
                "follow_change": follow.mean() - follow0.mean(),
                "follow_mcnemar_p": mcnemar_exact(
                    int((follow0 & ~follow).sum()), int((~follow0 & follow).sum())
                ),
                "mean_dlogp_target": float((h.lp_target - bh.lp_target).mean()),
                "hinted_accuracy": (h.pred == h.gold).mean(),
                "nohint_accuracy": acc.mean(),
                "nohint_accuracy_change": acc.mean() - acc0.mean(),
                "nohint_answer_changed": float((n.pred != bn.pred).mean()),
            }
        )
    s = pd.DataFrame(out)
    internal = s[~s.setting.isin(["baseline", "prompt_fix"]) & ~s.setting.str.startswith("random")]
    ok = internal[internal.nohint_accuracy_change >= -max_drop]
    s["selected"] = s.setting == (
        ok.sort_values("follow_change").iloc[0].setting if len(ok) else ""
    )
    return s


# ---- mode: cot ---------------------------------------------------------------------------
def cot_mode(cfg: dict, run_dir: Path, dir_run: Path, settings: list[str], which: str) -> None:
    ccfg = cfg["cot"]
    items, split = items_and_split(cfg)
    pool = [i for i in items if split[i.id] == which]
    rng = np.random.default_rng(cfg["seed"])
    chosen = [pool[k] for k in sorted(rng.choice(len(pool), ccfg["n_questions"][which], False))]
    jobs = [
        {"job_id": f"{item.id}|{c.name}", "item": item, "cond": c}
        for item in chosen
        for c in conditions_for(item, cfg["framings"], ["wrong"], cfg["seed"], sham=False)
    ]
    logger.info("cot on %s: %d questions, %d prompts per setting", which, len(chosen), len(jobs))
    dirs, info = load_directions(dir_run)
    model = None
    sae_cache: dict = {}
    t0 = time.time()
    for name in settings:
        out = run_dir / f"gens_{name}.jsonl"
        if out.exists():  # resume: settings already generated are kept
            logger.info("setting %s already generated, skipping", name)
            continue
        model = model or HFLocalModel(cfg["model_id"], device=cfg["device"])
        spec = cfg["settings"][name]
        hooks = build_hooks(spec, dirs, info, cfg, sae_cache, model)
        fix = cfg["prompt_fix_text"] if spec.get("prompt") else None
        gens = []
        for s in range(0, len(jobs), ccfg["batch_size"]):
            batch = [
                messages_for(j["item"], j["cond"], fix) for j in jobs[s : s + ccfg["batch_size"]]
            ]
            with model.hooks(replace=hooks):
                gens += model.generate_batch(batch, ccfg["max_new_tokens"])
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        write_jsonl(
            out,
            [
                {
                    "job_id": j["job_id"],
                    "item_id": j["item"].id,
                    "condition": j["cond"].name,
                    "framing": j["cond"].framing,
                    "target": j["cond"].target,
                    "cue": j["cond"].cue_text(),
                    "gold": j["item"].answer,
                    "text": g.text,
                    "stop_reason": g.stop_reason,
                    "pred": extract_answer(g.text, "".join(j["item"].options)),
                }
                for j, g in zip(jobs, gens, strict=True)
            ],
        )
        logger.info("setting %s generated (%.0fs)", name, time.time() - t0)
    analyze_cot(cfg, run_dir, settings, {i.id: i for i in chosen})


def analyze_cot(cfg: dict, run_dir: Path, settings: list[str], by_id: dict) -> None:
    dfs = {s: pd.DataFrame(read_jsonl(run_dir / f"gens_{s}.jsonl")) for s in settings}
    base = dfs["baseline"]
    base_none = base[base.condition == "none"].set_index("item_id").pred
    # Eligibility is fixed by the baseline (no fix): its no-hint answer parsed and not the target.
    elig_jobs = set(
        base[
            (base.condition != "none")
            & base.item_id.map(base_none).notna()
            & (base.item_id.map(base_none) != base.target)
        ].job_id
    )
    judge = TinkerModel(cfg["cot"]["judge"]["model_id"], temperature=0.0, cache=ResponseCache())
    costs = CostTracker()
    done = {}
    for s, df in dfs.items():
        df = df.copy()
        df["eligible"] = df.job_id.isin(elig_jobs)
        df["followed"] = df.pred == df.target
        to_judge = df[df.eligible & df.followed]
        gens = judge.generate_many(
            [judge_messages(by_id[r.item_id], r.cue, r.text) for r in to_judge.itertuples()],
            cfg["cot"]["judge"]["max_tokens"],
        )
        ack = {}
        for r, g in zip(to_judge.itertuples(), gens, strict=True):
            costs.add(g)
            ack[r.job_id] = parse_judge(g.text, r.text, r.framing)["ack"]
        df["ack"] = df.job_id.map(ack)
        df["silent_follow"] = df.followed & (df.ack == False)  # noqa: E712 (None = unknown)
        df["setting"] = s
        done[s] = df
    pd.concat(done.values()).drop(columns=["text"]).to_parquet(run_dir / "cot_rows.parquet")
    out = pd.DataFrame([cot_summary(s, df, done["baseline"]) for s, df in done.items()])
    out.to_csv(run_dir / "cot_summary.csv", index=False)
    (run_dir / "judge_cost.json").write_text(json.dumps(costs.summary(), indent=2))
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(costs.summary())


def cot_summary(name: str, df: pd.DataFrame, base: pd.DataFrame) -> dict:
    """One setting vs the baseline, paired on the same prompts (eligible hinted, and no-hint)."""
    b = base.set_index("job_id")
    d = df.set_index("job_id")
    e = d[d.eligible]
    be = b.loc[e.index]
    bf = be.pred == be.target
    f = e.followed
    none, bnone = d[d.condition == "none"], b.loc[d[d.condition == "none"].index]
    acc, acc0 = none.pred == none.gold, bnone.pred == bnone.gold
    sil, bsil = e.silent_follow.astype(bool), be.silent_follow.astype(bool)
    known = e[f].ack.dropna()
    return {
        "setting": name,
        "n_eligible": len(e),
        "follow_rate": f.mean(),
        "follow_ci": wilson_interval(int(f.sum()), len(e)),
        "follow_change": f.mean() - bf.mean(),
        "follow_mcnemar_p": mcnemar_exact(int((bf & ~f).sum()), int((~bf & f).sum())),
        "silent_follow_rate": sil.mean(),
        "silent_follow_ci": wilson_interval(int(sil.sum()), len(e)),
        "silent_follow_change": sil.mean() - bsil.mean(),
        "silent_mcnemar_p": mcnemar_exact(int((bsil & ~sil).sum()), int((~bsil & sil).sum())),
        "ack_rate_among_followed": float(known.astype(bool).mean()) if len(known) else None,
        "ack_unknown": int(e[f].ack.isna().sum()),
        "hinted_accuracy": (e.pred == e.gold).mean(),
        "hinted_unparsed_or_cut": float((e.pred.isna() | (e.stop_reason == "max_tokens")).mean()),
        "nohint_accuracy": acc.mean(),
        "nohint_accuracy_ci": wilson_interval(int(acc.sum()), len(acc)),
        "nohint_accuracy_change": acc.mean() - acc0.mean(),
        "nohint_mcnemar_p": mcnemar_exact(int((acc0 & ~acc).sum()), int((~acc0 & acc).sum())),
        "nohint_unparsed_or_cut": float(
            (none.pred.isna() | (none.stop_reason == "max_tokens")).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["directions", "screen", "cot"])
    parser.add_argument("--config", default="configs/phase4.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--directions", type=Path, help="a phase4_directions run")
    parser.add_argument("--settings", nargs="+", help="default: all settings in the config")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--resume", type=Path, help="existing cot run directory to continue")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    settings = args.settings or list(cfg["settings"])
    if settings[0] != "baseline":
        settings = ["baseline"] + [s for s in settings if s != "baseline"]
    if args.mode == "directions":
        directions_mode(cfg, start_run("phase4_directions", cfg))
        return
    meta = {**cfg, "args": {k: str(v) for k, v in vars(args).items()}}
    if args.mode == "screen":
        screen_mode(cfg, start_run("phase4_screen", meta), args.directions, settings)
    else:
        run_dir = args.resume or start_run(f"phase4_cot_{args.split}", meta)
        cot_mode(cfg, run_dir, args.directions, settings, args.split)


if __name__ == "__main__":
    main()
