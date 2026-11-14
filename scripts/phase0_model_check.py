"""Phase 0: load MedGemma, check it answers MedQA well enough, and cache activations.

Three checks:
1. Direct accuracy: compare the model's next-token probability for "A"/"B"/"C"/"D"
   after an "answer with only the letter" prompt (one forward pass per question).
2. Step-by-step answers: generate reasoning + "Final answer: X" for a few questions.
3. Activations: capture residual-stream activations at the Gemma Scope 2 layers and save
   them to disk (outputs/, not committed).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from dotenv import load_dotenv
from safetensors.torch import save_file

from medlens.config import load_config
from medlens.data.loaders import load_medqa
from medlens.eval.answers import cot_messages, direct_messages, extract_answer
from medlens.eval.stats import wilson_interval
from medlens.models.hf_local import HFLocalModel
from medlens.runs import REPO_ROOT, start_run, write_jsonl

logger = logging.getLogger("phase0_model_check")


def memory_gb(device: str) -> float | None:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return None


def letter_token_ids(model: HFLocalModel, letters: str = "ABCD") -> list[int]:
    ids = []
    for letter in letters:
        toks = model.tokenizer.encode(letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Letter {letter!r} is not a single token: {toks}")
        ids.append(toks[0])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    mcfg = cfg["model_check"]
    run_dir = start_run("phase0_model_check", cfg)
    summary: dict = {"model_id": mcfg["model_id"]}

    t0 = time.time()
    model = HFLocalModel(mcfg["model_id"], device=mcfg["device"])
    summary["load_seconds"] = round(time.time() - t0, 1)
    summary["device"] = model.device
    summary["n_layers"] = model.n_layers
    summary["memory_after_load_gib"] = memory_gb(model.device)
    logger.info(
        "Loaded in %.0fs on %s; %d layers", summary["load_seconds"], model.device, model.n_layers
    )

    n_needed = max(mcfg["n_direct"], mcfg["n_cot"], mcfg["n_activation_cache"])
    items = load_medqa("test", n=n_needed, seed=cfg["seed"])

    # 1) Direct-answer accuracy from next-token letter probabilities.
    letters = letter_token_ids(model)
    direct_rows = []
    t0 = time.time()
    for item in items[: mcfg["n_direct"]]:
        ids = model.encode(model.chat_prompt(direct_messages(item)), chat_formatted=True)
        logits, _ = model.forward_logits(ids)
        probs = torch.softmax(logits[0, -1, letters], dim=-1)
        pred = "ABCD"[int(probs.argmax())]
        mass = torch.softmax(logits[0, -1], dim=-1)[letters].sum().item()
        direct_rows.append(
            {
                "id": item.id,
                "answer": item.answer,
                "pred": pred,
                "correct": pred == item.answer,
                "letter_prob_mass": mass,
                "probs": dict(zip("ABCD", probs.tolist(), strict=True)),
            }
        )
    n_correct = sum(r["correct"] for r in direct_rows)
    summary["direct"] = {
        "n": len(direct_rows),
        "accuracy": n_correct / len(direct_rows),
        "ci95": wilson_interval(n_correct, len(direct_rows)),
        "mean_letter_prob_mass": sum(r["letter_prob_mass"] for r in direct_rows) / len(direct_rows),
        "seconds_per_question": round((time.time() - t0) / len(direct_rows), 2),
    }
    write_jsonl(run_dir / "direct_answers.jsonl", direct_rows)
    logger.info("Direct accuracy: %s", summary["direct"])

    # 2) Step-by-step answers.
    cot_rows = []
    t0 = time.time()
    for item in items[: mcfg["n_cot"]]:
        gen = model.generate(cot_messages(item), max_new_tokens=mcfg["max_new_tokens"])
        pred = extract_answer(gen.text)
        cot_rows.append(
            {
                "id": item.id,
                "answer": item.answer,
                "pred": pred,
                "correct": pred == item.answer,
                "output_tokens": gen.usage.output_tokens,
                "stop_reason": gen.stop_reason,
                "text": gen.text,
            }
        )
    gen_seconds = time.time() - t0
    n_correct = sum(r["correct"] for r in cot_rows)
    total_tokens = sum(r["output_tokens"] for r in cot_rows)
    summary["cot"] = {
        "n": len(cot_rows),
        "accuracy": n_correct / len(cot_rows),
        "ci95": wilson_interval(n_correct, len(cot_rows)),
        "unparseable": sum(r["pred"] is None for r in cot_rows),
        "hit_max_tokens": sum(r["stop_reason"] == "max_tokens" for r in cot_rows),
        "mean_output_tokens": total_tokens / len(cot_rows),
        "tokens_per_second": round(total_tokens / gen_seconds, 1),
    }
    write_jsonl(run_dir / "cot_answers.jsonl", cot_rows)
    logger.info("CoT: %s", summary["cot"])

    # 3) Activation capture at the Gemma Scope 2 layers.
    capture = [(layer, "resid_post") for layer in mcfg["layers"]]
    tensors, norms = {}, {layer: [] for layer in mcfg["layers"]}
    for i, item in enumerate(items[: mcfg["n_activation_cache"]]):
        ids = model.encode(model.chat_prompt(cot_messages(item)), chat_formatted=True)
        _, acts = model.forward_logits(ids, capture=capture)
        for (layer, site), act in acts.items():
            tensors[f"{i}.layer{layer}.{site}"] = act[0].to(torch.bfloat16).contiguous()
            norms[layer].append(act[0, 1:].norm(dim=-1).mean().item())  # skip <bos>
        tensors[f"{i}.input_ids"] = ids[0].cpu()
    save_file(tensors, str(run_dir / "activations.safetensors"))
    summary["activations"] = {
        "n_prompts": mcfg["n_activation_cache"],
        "layers": mcfg["layers"],
        "mean_token_norm_by_layer": {str(k): sum(v) / len(v) for k, v in norms.items()},
        "file_mb": round((run_dir / "activations.safetensors").stat().st_size / 1e6, 1),
    }
    summary["peak_memory_gib"] = memory_gb(model.device)

    (run_dir / "model_check_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
