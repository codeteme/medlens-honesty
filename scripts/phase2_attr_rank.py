"""Phase 2: rank SAE features by attribution to the hinted answer, over many unfaithful cases.

For each case: metric = logp(hinted letter) - logp(correct letter) after "Final answer:";
node attribution = activation x gradient (see medlens.interp.attribution). Attributions are
summed per feature within two regions, the hint's tokens and the last 5 tokens (the
"Final answer:" readout), and averaged over cases. Output columns match
differential_features.csv so phase2_causal.py can select from either.

Usage: uv run python scripts/phase2_attr_rank.py <capture_run> [--n 40]
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from medlens.blackbox.hints import build_messages, conditions_for
from medlens.config import load_config
from medlens.data.loaders import medqa_test_not_in_mpib
from medlens.interp.attribution import _splice_hooks
from medlens.interp.features import PREFILL, char_span_to_tokens, letter_token_ids
from medlens.interp.sae_eval import load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run

logger = logging.getLogger("phase2_attr_rank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_run", type=Path)
    parser.add_argument("--config", default="configs/phase2.yaml")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    run_dir = start_run(
        "phase2_attr_rank", {**cfg, "args": {k: str(v) for k, v in vars(args).items()}}
    )
    r = pd.read_parquet(args.capture_run / "readout.parquet")
    cases = r[(r.p1_label == "unfaithful") & r.condition.str.endswith("_wrong")]
    cases = pd.concat(
        [
            g.sample(n=min(len(g), args.n // 4), random_state=cfg["seed"])
            for _, g in cases.groupby("framing")
        ]
    )
    items = {i.id: i for i in medqa_test_not_in_mpib(cfg["capture"]["n_questions"], cfg["seed"])}
    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    layers = cfg["capture"]["layers"]
    saes = {
        L: load_sae(
            cfg["capture"]["sae_release"], cfg["capture"]["sae_id"].format(layer=L), model.device
        )
        for L in layers
    }
    sums = {
        (L, reg): torch.zeros(saes[L].W_dec.shape[0]) for L in layers for reg in ("hint", "final")
    }
    counts = {
        (L, reg): torch.zeros(saes[L].W_dec.shape[0]) for L in layers for reg in ("hint", "final")
    }

    for k, case in enumerate(cases.itertuples()):
        item = items[case.item_id]
        cond = next(
            c
            for c in conditions_for(item, cfg["capture"]["framings"], ["wrong"], cfg["seed"])
            if c.name == case.condition
        )
        prompt = model.chat_prompt(build_messages(item, cond))
        enc = model.tokenizer(
            prompt + PREFILL, return_offsets_mapping=True, add_special_tokens=False
        )
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        start = prompt.find(cond.cue_text())
        hint = char_span_to_tokens(
            [tuple(o) for o in enc["offset_mapping"]], start, start + len(cond.cue_text())
        )
        final = list(range(ids.shape[1] - 5, ids.shape[1]))
        lt = dict(zip(item.options, letter_token_ids(model, "".join(item.options)), strict=True))
        store: dict = {}
        with torch.enable_grad(), model.hooks(replace=_splice_hooks(saes, store)):
            lp = torch.log_softmax(model.model(input_ids=ids).logits[0, -1].float(), dim=-1)
            metric = lp[lt[cond.target]] - lp[lt[item.answer]]
            grads = torch.autograd.grad(metric, [store[L] for L in layers])
        for L, g in zip(layers, grads, strict=True):
            attr = (store[L].detach() * g)[0].cpu()  # [T, F]
            for reg, pos in (("hint", hint), ("final", final)):
                sums[(L, reg)] += attr[pos].sum(0)
                counts[(L, reg)] += (attr[pos] != 0).any(0).float()
        m = float(metric)
        del store, grads, lp, metric, attr
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("case %d/%d (%s) metric %.2f", k + 1, len(cases), case.framing, m)

    rows = []
    n = len(cases)
    for (L, reg), s in sums.items():
        mean = s / n
        for idx in torch.topk(mean, args.top).indices.tolist():
            rows.append(
                {
                    "layer": L,
                    "position": reg,
                    "feature": idx,
                    "mean_attribution": float(mean[idx]),
                    "cases_active": int(counts[(L, reg)][idx]),
                    "auroc": float(mean[idx]),  # score column used for ranking
                    "framings_with_positive_diff": 4,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "attribution_features.csv", index=False)
    print(
        out.groupby(["layer", "position"])
        .head(5)
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )


if __name__ == "__main__":
    main()
