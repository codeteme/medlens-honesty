"""Phase 2: plain-language labels for candidate features, with a held-out check.

For each feature (layer, index) of the Gemma Scope 2 pt 16k residual SAEs:
1. Examples: Neuronpedia's top activations (model `gemma-3-4b`, source
   `{layer}-gemmascope-2-res-16k`, general text on the base model) plus MedGemma's own
   top-activating tokens on a corpus of our prompts (hinted/unhinted MedQA, MedMCQA prose,
   WikiText).
2. Label: Qwen3.8-27B on Tinker sees 12 examples with the peak token marked [[like this]] and
   writes a short label.
3. Check (detection scoring): a fresh call sees only the label and 12 held-out contexts
   (6 where the feature fires strongly in MedGemma, 6 where it is silent) and picks the ones
   the label describes. Balanced accuracy is reported; 0.5 is chance.

Usage: uv run python scripts/phase2_interpret.py <capture_run> <analyze_run> [--n-diff 12 --n-cue 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
import torch
from dotenv import load_dotenv

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import load_medmcqa, load_wikitext_paragraphs, medqa_test_not_in_mpib
from medlens.interp.features import PREFILL
from medlens.interp.sae_eval import load_sae
from medlens.models.base import CostTracker
from medlens.models.cache import ResponseCache
from medlens.models.hf_local import HFLocalModel
from medlens.models.tinker_model import TinkerModel
from medlens.runs import REPO_ROOT, start_run, write_jsonl

logger = logging.getLogger("phase2_interpret")
NP_URL = "https://www.neuronpedia.org/api/feature/gemma-3-4b/{layer}-gemmascope-2-res-16k/{idx}"
NP_CACHE = REPO_ROOT / "cache" / "neuronpedia"
WINDOW = 10

LABEL_PROMPT = """You are interpreting one feature of a sparse autoencoder trained on a language model's internal activations.
Below are text snippets where the feature activates strongly. The token where it peaks is marked [[like this]].
Write a short, specific label (at most 10 words) for what the feature responds to, and one sentence of description.
Return JSON only: {{"label": "...", "description": "..."}}

{examples}"""

SCORE_PROMPT = """A feature of a language model has this description:
"{label}: {description}"
Below are numbered text snippets; in each, one token is marked [[like this]]. For which snippets does the description fit the marked token in its context?
Return JSON only: {{"fits": [snippet numbers]}}

{snippets}"""


def neuronpedia(layer: int, idx: int) -> dict:
    NP_CACHE.mkdir(parents=True, exist_ok=True)
    path = NP_CACHE / f"L{layer}_{idx}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = requests.get(NP_URL.format(layer=layer, idx=idx), timeout=60)
    r.raise_for_status()
    d = r.json()
    keep = {
        "explanations": [e.get("description") for e in d.get("explanations") or []],
        "examples": [
            {
                "tokens": a["tokens"],
                "max_index": a.get("maxValueTokenIndex"),
                "max_value": a.get("maxValue"),
            }
            for a in (d.get("activations") or [])
        ],
    }
    path.write_text(json.dumps(keep))
    time.sleep(0.3)
    return keep


def snippet(tokens: list[str], i: int) -> str:
    lo, hi = max(0, i - WINDOW), min(len(tokens), i + WINDOW // 2 + 1)
    parts = [(f"[[{t}]]" if j == i else t) for j, t in enumerate(tokens[lo:hi], start=lo)]
    return "".join(parts).replace("\n", " ").strip()


def corpus(cfg: dict, n_hint: int, n_other: int, seed: int) -> list[str]:
    items = medqa_test_not_in_mpib(cfg["capture"]["n_questions"], seed)
    rng = random.Random(seed)
    texts = []
    for item in rng.sample(items, n_hint):
        for cond in conditions_for(item, cfg["capture"]["framings"], ["wrong"], seed):
            if rng.random() < 0.35:
                texts.append(("chat", build_messages(item, cond)))
    mm = [i for i in load_medmcqa("validation") if i.meta.get("explanation")]
    for i in rng.sample(mm, n_other):
        texts.append(("raw", f"{i.question}\n\n{i.meta['explanation']}"))
    for t in load_wikitext_paragraphs(n_other, seed):
        texts.append(("raw", t))
    return texts


@torch.no_grad()
def collect(model, saes, feats_by_layer, texts):
    """Per feature: list of (activation, tokens, position) over every token of the corpus."""
    out = {(L, f): [] for L, fs in feats_by_layer.items() for f in fs}
    for kind, content in texts:
        if kind == "chat":
            text, chat = model.chat_prompt(content) + PREFILL, True
        else:
            text, chat = content, False
        ids = model.encode(text, chat_formatted=chat)[:, :384]
        toks = [model.tokenizer.decode([t]) for t in ids[0].tolist()]
        _, acts = model.forward_logits(ids, capture=[(L, "resid_post") for L in feats_by_layer])
        for L, fs in feats_by_layer.items():
            a = saes[L].encode(acts[(L, "resid_post")][0].to(model.device))[:, fs].cpu()
            a[0] = 0  # skip <bos>
            for j, f in enumerate(fs):
                col = a[:, j]
                p = int(col.argmax())
                out[(L, f)].append(
                    (
                        float(col[p]),
                        toks,
                        p,
                        [int(x) for x in (col == 0).nonzero().flatten().tolist()],
                    )
                )
    return out


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_run", type=Path)
    parser.add_argument("analyze_run", type=Path)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--n-diff", type=int, default=12)
    parser.add_argument("--n-cue", type=int, default=3)
    parser.add_argument("--n-hint-questions", type=int, default=60)
    parser.add_argument("--n-other", type=int, default=40)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    run_dir = start_run(
        "phase2_interpret", {**cfg, "args": {k: str(v) for k, v in vars(args).items()}}
    )

    diff = pd.read_csv(args.analyze_run / "differential_features.csv")
    diff = diff[diff.framings_with_positive_diff == 4].sort_values("auroc", ascending=False)
    chosen = diff.drop_duplicates(["layer", "feature"]).head(args.n_diff)
    cues = pd.read_csv(args.analyze_run / "cue_features.csv")
    cue_pick = cues.sort_values("mean_on_hint", ascending=False).groupby("framing").head(args.n_cue)
    feats = [("differential", int(r.layer), int(r.feature)) for r in chosen.itertuples()]
    feats += [
        (f"cue:{r.framing}", int(r.layer), int(r.feature))
        for r in cue_pick.itertuples()
        if (int(r.layer), int(r.feature)) not in {(L, f) for _, L, f in feats}
    ]
    by_layer: dict[int, list[int]] = {}
    for _, L, f in feats:
        by_layer.setdefault(L, []).append(f)

    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    saes = {
        L: load_sae(
            cfg["capture"]["sae_release"], cfg["capture"]["sae_id"].format(layer=L), model.device
        )
        for L in by_layer
    }
    texts = corpus(cfg, args.n_hint_questions, args.n_other, cfg["seed"] + 11)
    logger.info("corpus: %d texts, %d features", len(texts), len(feats))
    acts = collect(model, saes, by_layer, texts)
    del model

    judge = TinkerModel(cfg.get("interp_model", "Qwen/Qwen3.8-27B"), cache=ResponseCache())
    costs = CostTracker()
    rng = random.Random(cfg["seed"])
    rows = []
    for kind, L, f in feats:
        np_ = neuronpedia(L, f)
        ours = sorted(acts[(L, f)], key=lambda x: -x[0])
        active = [x for x in ours if x[0] > 0]
        shown_ours, held = active[:6], active[6:12]
        silent = [x for x in ours if x[0] == 0]
        examples = [
            snippet(e["tokens"], e["max_index"])
            for e in np_["examples"][:6]
            if e.get("max_index") is not None
        ]
        examples += [snippet(t, p) for _, t, p, _ in shown_ours]
        label_gen = judge.generate(
            [
                {
                    "role": "user",
                    "content": LABEL_PROMPT.format(examples="\n".join(f"- {e}" for e in examples)),
                }
            ],
            200,
        )
        costs.add(label_gen)
        lab = parse_json(label_gen.text)
        negs = []
        for _, t, _, zeros in rng.sample(silent, min(6, len(silent))):
            cand = [z for z in zeros if z > 0]
            if cand:
                negs.append(snippet(t, rng.choice(cand)))
        pos = [snippet(t, p) for _, t, p, _ in held]
        snippets = [(s, 1) for s in pos] + [(s, 0) for s in negs]
        rng.shuffle(snippets)
        score_gen = judge.generate(
            [
                {
                    "role": "user",
                    "content": SCORE_PROMPT.format(
                        label=lab.get("label", ""),
                        description=lab.get("description", ""),
                        snippets="\n".join(f"{i + 1}. {s}" for i, (s, _) in enumerate(snippets)),
                    ),
                }
            ],
            100,
        )
        costs.add(score_gen)
        fits = set(parse_json(score_gen.text).get("fits") or [])
        tp = sum(1 for i, (_, y) in enumerate(snippets) if y == 1 and (i + 1) in fits)
        tn = sum(1 for i, (_, y) in enumerate(snippets) if y == 0 and (i + 1) not in fits)
        n_pos, n_neg = len(pos), len(negs)
        bal = (tp / n_pos + tn / n_neg) / 2 if n_pos and n_neg else None
        rows.append(
            {
                "kind": kind,
                "layer": L,
                "feature": f,
                "neuronpedia_url": f"https://www.neuronpedia.org/gemma-3-4b/{L}-gemmascope-2-res-16k/{f}",
                "neuronpedia_explanations": np_["explanations"],
                "label": lab.get("label"),
                "description": lab.get("description"),
                "detection_balanced_accuracy": bal,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "medgemma_top_examples": [snippet(t, p) for _, t, p, _ in active[:3]],
                "max_activation_medgemma": active[0][0] if active else 0.0,
                "corpus_active_fraction": len(active) / len(ours),
            }
        )
        logger.info(
            "L%d/%d %s -> %s (detection %.2f)",
            L,
            f,
            kind,
            lab.get("label"),
            bal if bal is not None else float("nan"),
        )
    write_jsonl(run_dir / "feature_labels.jsonl", rows)
    pd.DataFrame(rows).drop(columns=["medgemma_top_examples"]).to_csv(
        run_dir / "feature_labels.csv", index=False
    )
    (run_dir / "cost.json").write_text(json.dumps(costs.summary(), indent=2))
    print(
        pd.DataFrame(rows)[
            ["kind", "layer", "feature", "label", "detection_balanced_accuracy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
