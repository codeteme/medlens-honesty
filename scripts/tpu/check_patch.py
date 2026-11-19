"""Phase 8, Part B: check that the vLLM fix patch (vllm_steer.py) is active and matches the Mac.

Compares the TPU validation outputs (run_validate.sh: 10 prompts, no fix and the layer-22
removal, plus a huge push that should break the text) with the same prompts generated on the
Mac with HF hooks (a `phase4_fix.py cot --debug` run).

Usage: uv run python scripts/tpu/check_patch.py <tpu_results_dir> <mac_phase4_cot_debug_run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medlens.eval.answers import extract_answer
from medlens.runs import read_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tpu", type=Path)
    p.add_argument("mac", type=Path)
    a = p.parse_args()

    def tpu(name: str) -> dict[str, str]:
        return {r["job_id"]: r["generation"]["text"] for r in read_jsonl(a.tpu / name)}

    def mac(name: str) -> dict[str, str]:
        return {r["job_id"]: r["pred"] for r in read_jsonl(a.mac / name)}

    tb, ta = tpu("out_V_baseline.jsonl"), tpu("out_V_ablate_pre_L22.jsonl")
    mb, ma = mac("gens_baseline.jsonl"), mac("gens_ablate_pre_L22.jsonl")
    huge = [r["generation"]["text"] for r in read_jsonl(a.tpu / "out_V_huge.jsonl")]
    out = {
        "n": len(mb),
        "same_answer_as_mac_no_fix": sum(mb[j] == extract_answer(tb[j], "ABCD") for j in mb),
        "same_answer_as_mac_ablate": sum(ma[j] == extract_answer(ta[j], "ABCD") for j in mb),
        "tpu_ablate_text_changed": sum(ta[j] != tb[j] for j in tb),
        "huge_push_samples": [t[:120] for t in huge],
    }
    (a.tpu / "patch_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
