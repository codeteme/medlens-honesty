"""Generate with vLLM while applying a Phase 4 fix to the residual stream (run on the TPU).

Phase 8, Part B. The Phase 4 fixes act on the residual stream after decoder layer L, at every
token (`medlens.interp.steering`):
  ablate: x - (x.d) d          add: x + amount * d      (d a unit vector, amount = gaps * gap)
vLLM's Gemma 3 decoder layer returns (hidden, residual) with the true residual stream equal to
hidden + residual (the next layer's fused add-norm sums them). The patch computes that sum,
applies the op, and returns (op(sum) - residual, residual), so the next layer sees op(sum).

One setting per process: the patch reads its settings once, before the model is built and
compiled, so nothing from one setting can leak into another.

Input: JSONL of {"job_id", "messages"}. Output: the same format as vllm_generate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def make_op(op: dict, dirs, gaps: dict):
    import torch

    d = torch.as_tensor(dirs[f"{op['dir']}_L{op['layer']}"], dtype=torch.float32)
    d = d / d.norm()
    if op["kind"] == "ablate":

        def f(x):
            dd = d.to(x.device)
            xf = x.float()
            return (xf - (xf @ dd).unsqueeze(-1) * dd).to(x.dtype)

    elif op["kind"] == "add":
        amount = op["gaps"] * gaps[f"{op['dir']}_L{op['layer']}"]

        def f(x):
            return (x.float() + amount * d.to(x.device)).to(x.dtype)

    else:
        raise ValueError(op["kind"])
    return f


def patch_gemma3(ops_by_layer: dict) -> None:
    from vllm.model_executor.models import gemma3

    cls = gemma3.Gemma3DecoderLayer
    orig_init, orig_forward = cls.__init__, cls.forward

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix", "")
        self._medlens_idx = int(prefix.split(".")[-1]) if prefix.split(".")[-1].isdigit() else None

    def forward(self, *args, **kwargs):
        hidden, residual = orig_forward(self, *args, **kwargs)
        fns = ops_by_layer.get(self._medlens_idx)
        if not fns:
            return hidden, residual
        r = hidden + residual
        for f in fns:
            r = f(r)
        return r - residual, residual

    cls.__init__, cls.forward = init, forward
    print(f"patched Gemma3DecoderLayer for layers {sorted(ops_by_layer)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs")
    parser.add_argument("out")
    parser.add_argument(
        "--ops", default="[]", help='JSON list, e.g. [{"kind":"ablate","layer":22,"dir":"pre"}]'
    )
    parser.add_argument("--directions", help="directions.npz (with directions.json beside it)")
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # patch must reach the model
    ops = json.loads(args.ops)
    ops_by_layer: dict[int, list] = {}
    if ops:
        import numpy as np

        z = np.load(args.directions)
        dirs = {k: z[k] for k in z.files}
        with open(args.directions.replace(".npz", ".json")) as f:
            gaps = json.load(f)["gap"]
        for op in ops:
            ops_by_layer.setdefault(op["layer"], []).append(make_op(op, dirs, gaps))
    patch_gemma3(ops_by_layer)  # also patches with no ops, so every setting runs the same code

    import vllm
    from vllm import LLM, SamplingParams

    with open(args.jobs) as f:
        jobs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        jobs = jobs[: args.limit]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 0},
    )
    t0 = time.time()
    outs = llm.chat(
        [j["messages"] for j in jobs],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=True,
    )
    secs = time.time() - t0
    source = {
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "device": "tpu-v6e-1",
        "model": args.model,
        "decoding": "greedy",
        "ops": ops,
    }
    n_out = 0
    with open(args.out, "w") as f:
        for job, out in zip(jobs, outs, strict=True):
            c = out.outputs[0]
            n_out += len(c.token_ids)
            gen = {
                "text": c.text,
                "stop_reason": "max_tokens" if c.finish_reason == "length" else "end_turn",
                "usage": {
                    "input_tokens": len(out.prompt_token_ids),
                    "output_tokens": len(c.token_ids),
                },
            }
            f.write(
                json.dumps({"job_id": job["job_id"], "generation": gen, "source": source}) + "\n"
            )
    print(f"{len(jobs)} jobs, {n_out} tokens, {secs:.0f}s ({n_out / max(secs, 1e-9):.0f} tok/s)")


if __name__ == "__main__":
    main()
