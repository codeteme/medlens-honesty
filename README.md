# medlens-honesty

**Does a medical AI's written reasoning match what actually drove its answer?**

Medical AI models often explain their reasoning before answering. Sometimes that explanation
leaves out what really mattered. For example, a model may switch answers because the prompt
said "an attending thinks it's B," without ever mentioning that. This project tries to
(1) catch these cases by looking inside the model, (2) fix them, and (3) test whether
showing clinicians a warning helps them decide when to trust the model.

- **Model:** MedGemma 4B for every phase, analysed with Google DeepMind's Gemma Scope 2 tools.
- **Optional outside check:** Qwen 27B (via Tinker), inputs and outputs only, at the end.

This is a Duke Explainable AI final project. It is built in phases; see `reports/` for each
phase's results. **Status: Phase 3 complete (see `reports/phase1.md` to `reports/phase3.md`).**

**Phase 1 headline:** across 375 MedQA questions, MedGemma 4B moved to a hinted answer in
912 eligible runs, and its written reasoning mentioned the hint in only 30 of them (3.3%).

**Phase 2 headline:** inside the model, the hint travels mainly as an *answer letter*
("multiple-choice answer letter" features on the hint's own tokens at layers 17/22), not as
an authority signal. Removing 10 of these features weakens hint-following when the model
answers immediately (87% to 75%) but not when it reasons step by step first.

**Phase 3 headline:** a simple probe on the model's activations, never told what the hint was,
flags silently followed hints with AUROC 0.98 once the model has reasoned (0.87 before it
reasons). It rarely fires on prompts with no hint (under 1%), and it works on hint types and
specialties it wasn't trained on. Text-only detectors score 0.56–0.72. Caveat: the probe
detects *following* the hint, not *hiding* it; it fires on most runs that admitted the hint too.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your Hugging Face token
```

Several resources are gated: accept the terms on Hugging Face for
`google/medgemma-4b-it`, `google/medgemma-27b-text-it`, `google/gemma-3-4b-it`, and the
MPIB dataset `jhlee0619/mpib`.

## Reproducing Phase 0

Every script has a `--debug` mode that runs in minutes on a laptop.

```bash
uv run python scripts/phase0_env_report.py           # hardware + memory estimates
uv run python scripts/phase0_data_check.py           # MedQA, MedMCQA, MPIB load and sizes
uv run python scripts/phase0_model_check.py --debug  # load MedGemma 4B, accuracy, activations
uv run python scripts/phase0_sae_recon.py --debug    # how well Gemma Scope 2 fits MedGemma
uv run python scripts/phase0_figure.py outputs/phase0_sae_recon/<run>/  # report figure
uv run pytest                                        # tests
```

## Reproducing Phase 1

Generation can run on the Mac (slow) or with vLLM on a TPU (fast). The judge runs on Tinker,
so set `TINKER_API_KEY` in `.env`.

```bash
# Pilot on the Mac: 50 questions x 10 conditions, judged on Tinker
uv run python scripts/phase1_hints.py
# Full run: export prompts, generate on the TPU, then judge and label on the Mac
uv run python scripts/phase1_hints.py --config configs/phase1_full.yaml --export-jobs cache/tpu/jobs_full.jsonl
uv run python scripts/phase1_probes.py export1 --out cache/tpu/probes1_full.jsonl
#   on the TPU (pip install vllm-tpu==0.33.0):
#   python scripts/tpu/vllm_generate.py <jobs.jsonl> <out.jsonl>
uv run python scripts/phase1_probes.py export2 --generations cache/tpu/out_stage1.jsonl --out cache/tpu/probes2_full.jsonl
uv run python scripts/phase1_hints.py --config configs/phase1_full.yaml --generations cache/tpu/out_stage1.jsonl
uv run python scripts/phase1_probes.py analyze --generations cache/tpu/out_stage1.jsonl --generations cache/tpu/out_stage2.jsonl
uv run python scripts/phase1_judge_agreement.py outputs/phase1_hints/<run>/ --annotator llm_A=llm_labels_A.csv --annotator llm_B=llm_labels_B.csv
uv run python scripts/phase1_report_tables.py outputs/phase1_hints/<run>/ outputs/phase1_probes/<run>/
```

Phase 2 and Phase 3 reproduction steps are at the end of `reports/phase2.md` and
`reports/phase3.md`.

Each run writes to `outputs/<script>/<timestamp>/` with the exact config, the git commit,
package versions, logs and results. `outputs/` and `cache/` are not committed.

## Repository layout

```
configs/        YAML configs (each has a `debug:` block) and verified API prices
src/medlens/    data loaders, model wrappers, interpretability and evaluation code
scripts/        one entry point per experiment
reports/        one plain-English report per phase
tests/          pytest tests for data processing and metrics
```
