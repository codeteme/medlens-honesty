"""Phase 0: how well do Gemma Scope 2 SAEs/transcoders fit MedGemma 4B's activations?

For each model (MedGemma 4B, plus plain Gemma 3 4B IT as a control), each Gemma Scope 2
suite (trained on Gemma 3 4B IT or PT), each kind (residual SAE, transcoder) and each
layer, we measure reconstruction error and the damage done by splicing the reconstruction
back into the model, on medical and general text. Comparing MedGemma to the control shows
how much of the error comes from MedGemma's medical fine-tuning.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time

import pandas as pd
import torch
from dotenv import load_dotenv

from medlens.config import load_config
from medlens.data.loaders import load_medmcqa, load_medqa, load_wikitext_paragraphs
from medlens.eval.answers import cot_messages
from medlens.interp.sae_eval import SAESpec, evaluate_text, load_sae
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run, write_jsonl

logger = logging.getLogger("phase0_sae_recon")
METRICS = ["fvu", "cosine", "l0", "delta_ce", "kl", "ce_recovered"]


def build_texts(cfg: dict, seed: int) -> dict[str, list[tuple[str, bool]]]:
    """domain -> list of (text, is_chat_formatted_prompt)."""
    n = cfg["domains"]
    texts: dict[str, list[tuple[str, bool]]] = {}
    if n.get("medqa_chat"):
        items = load_medqa("test", n=n["medqa_chat"], seed=seed + 1)
        texts["medqa_chat"] = [(item, True) for item in items]  # formatted per model below
    if n.get("medmcqa_explanation"):
        pool = [
            i
            for i in load_medmcqa("validation")
            if i.meta.get("explanation") and len(i.meta["explanation"]) > 300
        ]
        items = pd.Series(pool).sample(n=n["medmcqa_explanation"], random_state=seed).tolist()
        texts["medmcqa_explanation"] = [
            (f"{i.question}\n\n{i.meta['explanation']}", False) for i in items
        ]
    if n.get("wikitext"):
        texts["wikitext"] = [(t, False) for t in load_wikitext_paragraphs(n["wikitext"], seed)]
    return texts


def tokenize(model: HFLocalModel, texts: dict, max_tokens: int) -> dict[str, list[torch.Tensor]]:
    out = {}
    for domain, entries in texts.items():
        ids_list = []
        for content, is_chat in entries:
            text = model.chat_prompt(cot_messages(content)) if is_chat else content
            ids_list.append(model.encode(text, chat_formatted=is_chat)[:, :max_tokens])
        out[domain] = ids_list
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "suite", "kind", "layer", "domain"]
    agg = df.groupby(keys)[METRICS].mean()
    agg["n_texts"] = df.groupby(keys).size()
    agg["n_tokens"] = df.groupby(keys)["n_tokens"].sum()
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    scfg = cfg["sae_recon"]
    run_dir = start_run("phase0_sae_recon", cfg)
    texts = build_texts(scfg, cfg["seed"])
    logger.info("Texts per domain: %s", {k: len(v) for k, v in texts.items()})

    rows = []
    t_start = time.time()
    for model_id in scfg["models"]:
        model = HFLocalModel(model_id, device=scfg["device"])
        ids_by_domain = tokenize(model, texts, scfg["max_tokens"])
        for layer in scfg["layers"]:
            specs = []
            for suite in scfg["suites"]:
                for kind, release_kind, template in [
                    ("res", "res", scfg["res_id"]),
                    ("transcoder", "transcoders", scfg["transcoder_id"]),
                ]:
                    release = f"gemma-scope-2-4b-{suite}-{release_kind}"
                    sae_id = template.format(layer=layer)
                    logger.info("Loading %s / %s", release, sae_id)
                    sae = load_sae(release, sae_id, device=model.device)
                    specs.append(SAESpec(f"{suite}|{kind}|{sae_id}", layer, sae))
            for domain, ids_list in ids_by_domain.items():
                for t_idx, ids in enumerate(ids_list):
                    for res in evaluate_text(model, specs, ids):
                        suite, kind, sae_id = res.pop("sae").split("|")
                        rows.append(
                            {
                                "model": model_id,
                                "suite": suite,
                                "kind": kind,
                                "sae_id": sae_id,
                                "domain": domain,
                                "text_idx": t_idx,
                                **res,
                            }
                        )
            logger.info(
                "Layer %d done for %s (%.0fs elapsed)", layer, model_id, time.time() - t_start
            )
            del specs
            gc.collect()
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    write_jsonl(run_dir / "per_text.jsonl", rows)
    summary = summarize(pd.DataFrame(rows))
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".3f"))
    (run_dir / "timing.json").write_text(json.dumps({"seconds": round(time.time() - t_start)}))
    print(summary.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
