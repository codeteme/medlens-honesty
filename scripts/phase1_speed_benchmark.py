"""Phase 1: how much faster is batched generation on this machine?

Generates step-by-step answers for the same MedQA prompts at several batch sizes and reports
tokens/second, plus whether the parsed answers agree with batch size 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from dotenv import load_dotenv

from medlens.config import load_config
from medlens.data.loaders import load_medqa
from medlens.eval.answers import cot_messages, extract_answer
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run, write_jsonl

logger = logging.getLogger("phase1_speed_benchmark")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    bcfg = cfg["speed_benchmark"]
    run_dir = start_run("phase1_speed_benchmark", cfg)

    model = HFLocalModel(cfg["model_id"], device=cfg["device"])
    items = load_medqa("test", n=bcfg["n_prompts"], seed=cfg["seed"] + 100)
    chats = [cot_messages(i) for i in items]

    results, rows = [], []
    answers_by_bs: dict[int, list] = {}
    for bs in bcfg["batch_sizes"]:
        t0 = time.time()
        gens = []
        for start in range(0, len(chats), bs):
            gens += model.generate_batch(chats[start : start + bs], bcfg["max_new_tokens"])
        secs = time.time() - t0
        n_tok = sum(g.usage.output_tokens for g in gens)
        answers_by_bs[bs] = [extract_answer(g.text) for g in gens]
        res = {
            "batch_size": bs,
            "seconds": round(secs, 1),
            "output_tokens": n_tok,
            "tokens_per_second": round(n_tok / secs, 1),
            "seconds_per_answer": round(secs / len(gens), 1),
            "hit_max_tokens": sum(g.stop_reason == "max_tokens" for g in gens),
        }
        results.append(res)
        logger.info("%s", res)
        rows += [
            {"batch_size": bs, "id": i.id, **g.to_dict()} for i, g in zip(items, gens, strict=True)
        ]

    base = answers_by_bs[bcfg["batch_sizes"][0]]
    for res in results:
        same = sum(a == b for a, b in zip(base, answers_by_bs[res["batch_size"]], strict=True))
        res["answers_same_as_first_batch_size"] = f"{same}/{len(base)}"
    write_jsonl(run_dir / "generations.jsonl", rows)
    (run_dir / "speed.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
