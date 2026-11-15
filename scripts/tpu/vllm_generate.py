"""Generate answers for exported jobs with vLLM (run on the TPU machine).

Input: JSONL of {"job_id", "messages"} written by `phase1_hints.py --export-jobs`.
Output: JSONL of {"job_id", "generation", "source"} read by `phase1_hints.py --generations`.
Greedy decoding (temperature 0). vLLM batches requests dynamically, so exact outputs can
depend on what else is in flight; the saved file is the reproducible record.

Only depends on vllm, so it runs in a plain environment with `pip install vllm-tpu`.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs")
    parser.add_argument("out")
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

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
        limit_mm_per_prompt={"image": 0},  # text only: skip the image encoder's inputs
    )
    # Jobs may set their own "max_tokens"; run one vLLM call per distinct limit.
    limits = sorted({j.get("max_tokens", args.max_tokens) for j in jobs})
    outputs: list = [None] * len(jobs)
    t0 = time.time()
    for limit in limits:
        idx = [i for i, j in enumerate(jobs) if j.get("max_tokens", args.max_tokens) == limit]
        params = SamplingParams(temperature=0.0, max_tokens=limit)
        outs = llm.chat([jobs[i]["messages"] for i in idx], params, use_tqdm=True)
        for i, o in zip(idx, outs, strict=True):
            outputs[i] = o
    secs = time.time() - t0
    source = {
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "device": "tpu-v6e-1",
        "model": args.model,
        "decoding": "greedy",
        "max_tokens_default": args.max_tokens,
    }
    n_out = 0
    with open(args.out, "w") as f:
        for job, out in zip(jobs, outputs, strict=True):
            c = out.outputs[0]
            n_out += len(c.token_ids)
            gen = {
                "text": c.text,
                "model": f"vllm/{args.model}",
                "usage": {
                    "input_tokens": len(out.prompt_token_ids),
                    "output_tokens": len(c.token_ids),
                    "cost_usd": None,
                },
                "stop_reason": "max_tokens" if c.finish_reason == "length" else "end_turn",
                "cached": False,
                "meta": {"finish_reason": c.finish_reason},
            }
            f.write(
                json.dumps({"job_id": job["job_id"], "generation": gen, "source": source}) + "\n"
            )
    print(
        json.dumps(
            {
                "n": len(jobs),
                "seconds": round(secs, 1),
                "output_tokens": n_out,
                "tokens_per_second": round(n_out / secs, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
