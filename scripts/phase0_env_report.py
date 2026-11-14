"""Phase 0: report hardware and estimate memory needs for MedGemma 4B and 27B.

All sizes come from Hugging Face metadata (parameter counts, SAE file sizes and tensor
shapes), not from hard-coded numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess

import requests
import torch
from dotenv import load_dotenv
from sae_lens.loading.pretrained_sae_loaders import get_safetensors_tensor_shapes

from medlens.config import load_config
from medlens.runs import REPO_ROOT, start_run

HF_API = "https://huggingface.co/api/models"
MODELS = {"4b": "google/medgemma-4b-it", "27b": "google/medgemma-27b-text-it"}
SCOPE = {"4b": "google/gemma-scope-2-4b-it", "27b": "google/gemma-scope-2-27b-it"}
EXAMPLE_LAYER = {"4b": 17, "27b": 31}
GB = 1e9
GIB = 2**30


def hf_json(url: str) -> dict:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def file_sizes(repo: str, folder: str) -> dict[str, int]:
    tree = hf_json(f"{HF_API}/{repo}/tree/main/{folder}")
    return {item["path"].split("/")[-1]: item.get("size", 0) for item in tree}


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "disk_free_gb": round(shutil.disk_usage(REPO_ROOT).free / GB, 1),
    }
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
            capture_output=True,
            text=True,
        ).stdout.split("\n")
        info["chip"] = out[0]
        info["unified_memory_gib"] = round(int(out[1]) / GIB, 1)
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def estimates(size: str) -> dict:
    meta = hf_json(f"{HF_API}/{MODELS[size]}")
    n_params = meta["safetensors"]["total"]
    layer = EXAMPLE_LAYER[size]
    res = file_sizes(SCOPE[size], f"resid_post/layer_{layer}_width_65k_l0_medium")
    tc = file_sizes(SCOPE[size], f"transcoder/layer_{layer}_width_262k_l0_medium_affine")
    shapes = get_safetensors_tensor_shapes(
        SCOPE[size], f"resid_post/layer_{layer}_width_65k_l0_medium/params.safetensors"
    )
    d_model = shapes["w_enc"][0]
    layers_meta = hf_json(f"{HF_API}/{SCOPE[size]}/tree/main/resid_post_all")
    n_layers = len({p["path"].split("/")[1].split("_")[1] for p in layers_meta})
    act_per_token_layer = d_model * 2  # bytes, bf16
    return {
        "model_id": MODELS[size],
        "n_params": n_params,  # 4B count includes the image encoder, unused here
        "weights_bf16_gb": round(n_params * 2 / GB, 1),
        "d_model": d_model,
        "n_layers": n_layers,
        "resid_activation_kb_per_token_per_layer_bf16": round(act_per_token_layer / 1e3, 1),
        "all_layers_mb_per_1k_tokens_bf16": round(act_per_token_layer * n_layers * 1000 / 1e6),
        "four_layers_mb_per_1k_tokens_bf16": round(act_per_token_layer * 4 * 1000 / 1e6),
        "sae_65k_params_gb": round(res["params.safetensors"] / GB, 2),
        "transcoder_262k_affine_params_gb": round(tc["params.safetensors"] / GB, 2),
        "unneeded_examples_file_gb": round(tc.get("examples.safetensors", 0) / GB, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config, args.debug)
    run_dir = start_run("phase0_env_report", cfg)

    report = {"hardware": hardware(), "estimates": {s: estimates(s) for s in MODELS}}
    for e in report["estimates"].values():
        # Inference footprint: weights + one wide transcoder (fp32) + working memory for
        # activations/KV cache, which we bound loosely at 10% of weights.
        e["rough_total_gb_weights_plus_one_transcoder"] = round(
            e["weights_bf16_gb"] * 1.1 + e["transcoder_262k_affine_params_gb"], 1
        )
    (run_dir / "env_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
