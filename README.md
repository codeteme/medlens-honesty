# medlens-honesty

**Does a medical AI's written reasoning match what actually drove its answer?**

Medical AI models often explain their reasoning before answering. Sometimes that explanation
leaves out what really mattered. For example, a model may switch answers because the prompt
said "an attending thinks it's B," without ever mentioning that. This project tries to
(1) catch these cases by looking inside the model, (2) fix them, and (3) show the difference
in a demo that puts the model's reasoning next to an internal warning.

- **Model:** MedGemma 4B for every phase, analysed with Google DeepMind's Gemma Scope 2 tools.
- **Status:** complete. Duke AIPI 590 (Explainable AI) final project, built in phases.

**Read this first:**
- Paper draft: `reports/paper.md`
- Poster: `poster/medlens-honesty-poster-24x36.pdf`
- Demo: `demo/` (see `demo/README.md`). A private online copy is at
  [claude.ai](https://claude.ai/artifact/Voko8MSzjQnDq1jkEMYypa); ask the author for access.
- Reproducibility checklist: `reports/reproducibility.md`
- One report per phase: `reports/phase0.md` to `reports/phase5.md`

## What we found

1. **The reasoning hides the hint.** Across 375 MedQA questions, a hint changed MedGemma 4B's
   answer in 912 eligible runs. Its reasoning mentioned the hint in only 30 (3.3%). The
   strongest hint, a fake metadata tag, moved 64% of answers and was never mentioned.
2. **Inside, the hint is an answer letter.** The causal features sit on the hint's own letter
   ("multiple-choice answer letter", layers 17 and 22), not on who gave the hint. Removing 10
   of them weakens hint-following in quick answers (87% to 75%) but not in step-by-step answers.
3. **The activations reveal it.** A probe on layer-22 activations, never told the hint, flags
   silently followed hints with AUROC 0.98 after reasoning (0.87 before). It works on unseen
   hint types and specialties. Text-only detectors score 0.56–0.72.
4. **The probe flags following, not hiding, and leans on the letter.** It fires on 76% of runs
   that admitted the hint, and on 54% of right-answer hints the model would have chosen anyway.
   When the hint names the option text instead of a letter, its AUROC falls to 0.82–0.83 and
   the warning badge catches under 2% of silent runs (Phase 8). A made-up clinical finding moved
   20% of answers, but the reasoning almost always mentioned it.
5. **Fixing it failed once the model reasons.** Removing the hint-following direction
   cut quick-answer hint-following from 45% to 28%. Silent following in step-by-step answers
   stayed at 31% or got worse (pre-registered test). On 100 untouched test questions, no fix
   passed either (best: 37.5% to 33.4%, Holm p = 0.29).
6. **No detectable link to prompt injection.** On 150 MPIB injection cases, the probe could not tell
   which answers the injection changed (AUROC 0.48). The pre-registered verdict on a shared
   mechanism is inconclusive, leaning against.

**Main limits:**
- One 4B model, English multiple choice, greedy decoding.
- Most hints name an answer letter; the Phase 8 hints that don't were tested only with the
  frozen probe.
- The judge was checked against model annotators, not people.
- The optional Qwen 27B outside check and the human study were not run.

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

Phase 2 to Phase 5 reproduction steps are at the end of `reports/phase2.md` to
`reports/phase5.md`.

Each run writes to `outputs/<script>/<timestamp>/` with the exact config, the git commit,
package versions, logs and results. `outputs/` and `cache/` are not committed.

## Repository layout

```
configs/        YAML configs (each has a `debug:` block) and verified API prices
src/medlens/    data loaders, model wrappers, interpretability and evaluation code
scripts/        one entry point per experiment
reports/        one plain-English report per phase, the paper draft, reproducibility checklist
demo/           static demo page and its precomputed data
poster/         poster build script, fonts, logo and the PDF
tests/          pytest tests for data processing, metrics and hooks
```
