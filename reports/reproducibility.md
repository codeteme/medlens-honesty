# Reproducibility checklist

Every number in the reports, paper, poster and demo comes from a script in `scripts/` that ran
successfully, with its config saved alongside the outputs. This page lists what you need to
repeat the work and what to expect.

## Environment

- [x] **Code:** this repository. Each run folder records the git commit it ran from.
- [x] **Python 3.12** with `uv` and a lockfile (`uv.lock`). Install with `uv sync`.
- [x] **Key packages** (from the lockfile): torch 2.15.0, transformers 5.23.0, sae-lens 6.55.5,
  scikit-learn 1.9.1.
- [x] **Hardware used:**
  - Apple M4 laptop, 24 GB unified memory, PyTorch MPS backend, bf16. All internals work
    (Phases 0 and 2–5).
  - Course TPU v6e-1 with `vllm-tpu==0.33.0`. Bulk generation only (Phase 1).
- [x] **Secrets** go in `.env` (template: `.env.example`): a Hugging Face token and a Tinker API
  key. Never committed.
- [x] **Gated resources:** accept the terms on Hugging Face for MedGemma
  (`google/medgemma-4b-it`) and MPIB (`jhlee0619/mpib`). Gemma Scope 2 (`gemma-scope-2-4b-pt-res`)
  loads through SAELens.

## Data and splits

- [x] **Questions:** all 375 MedQA (`bigbio/med_qa`, 4 options) test questions that do not
  appear in MPIB (`medqa_test_not_in_mpib`; seed 0 only sets their order).
- [x] **Conditions:** 10 per question (`conditions_for`, seed 0), so the wrong-answer target is
  the same across framings.
- [x] **Detector folds:** 5 folds grouped by question, so no question is in both training and
  testing (tested in `tests/`).
- [x] **Phase 4 dev/test split:** a seeded hash of the question ID (`dev_fraction: 0.5`), giving
  164 dev and 211 test questions. Fixed before any Phase 4 run.
- [x] **MPIB:** 150 V1 cases matched to 4-option MedQA by their anchor question, seed 0.
- [x] **Demo:** the first 40 of the 375 questions. Not hand-picked.

## Determinism

- [x] **Greedy decoding everywhere;** fixed seeds in every config.
- [ ] **Bit-exact repeats are not guaranteed across devices.**
  - The TPU (vLLM) and Mac (HF) agree on 88.8% of answers over 500 prompts (Phase 1). So we
    never mix devices within one comparison.
  - Activations reused from Phases 2–3 were recomputed on the Mac for 10 runs and matched with
    cosine similarity 1.000 (Phase 5).
- [x] **Judge calls are cached** by a hash of the rubric and input. Reruns reuse them unless the
  rubric changes.

## Pre-registration and deviations

- [x] **Phase 4 selection rules and verdict criteria** were committed before any Phase 4 run, in
  `reports/phase4_preregistration.md` plus one dated clarification.
- [x] **Deviations are reported** in `reports/phase4.md`:
  - the test-half confirmation was skipped because no fix survived dev;
  - the MPIB fix arm (H2) was not run;
  - two exploratory numbers were added after the results and are labelled as such.
- [x] **Other scope changes** are recorded in `CLAUDE.md`:
  - judge validation by model annotators instead of hand labels;
  - the demo limited to 40 questions;
  - Phase 6 (human study) dropped.

## Runs behind the reported numbers

All run folders are under `outputs/`, which is not committed. Each one holds `config.yaml`,
`meta.json` (commit, package versions) and `run.log`.

| Phase | Run folders | Reproduce |
|---|---|---|
| 0 | `phase0_model_check/20261114-142730`, `phase0_sae_recon/20261114-143911` | `reports/phase0.md` |
| 1 | `phase1_hints/20261114-193610`, `phase1_probes/20261114-193340` | `README.md` (Phase 1 commands) |
| 2 | `phase2_capture/20261114-194828`, `phase2_attr_rank/20261114-211250`, `phase2_causal_readout/20261114-212634`, `phase2_causal_cot/20261114-222220`, `phase2_interpret/20261114-223420` | `reports/phase2.md` |
| 3 | `phase3_capture_postcot/20261115-142843`, `phase3_probes/20261115-153757`, `phase3_followups/20261115-154212` | `reports/phase3.md` |
| 4 | `phase4_directions/20261115-163308`, `phase4_screen/20261115-164028`, `phase4_cot_dev/20261115-174147`, `phase4_mpib/20261116-101344` | `reports/phase4.md` |
| 5 | `phase5_score_runs/20261116-171012`, `phase5_build_demo/20261116-173236` | `reports/phase5.md`, `demo/README.md` |
| Poster | reads the runs above | `poster/README.md` |

Every experiment script has a `--debug` mode, roughly 3–20 examples, that runs in minutes.
Run it first.

## Compute and cost

| Phase | Where | Main time | Cost |
|---|---|---|---|
| 0 | Mac | SAE fit check 90 min, model check 11 min | — |
| 1 | TPU + Tinker judge | 0.4 TPU-hours of generation | $1.91 ($1.33 Tinker + $0.58 TPU) |
| 2 | Mac + Tinker labels | capture 46 min, attribution and causal tests about 2 h in all | $0.02 |
| 3 | Mac + Tinker judge | post-reasoning capture 67 min | $3.24 |
| 4 | Mac + Tinker judge | screen 61 min, dev step-by-step 5.5 h (memory swapping), MPIB 75 min | $0.09 |
| 5 | Mac | scoring 17 min (plus an 18-min attempt that was stopped) | — |
| **Total** | | | **about $5.26** |

API costs are logged per run in `cost.json` or `judge_cost.json`.

## Tests and code quality

- [x] 87 `pytest` tests: data processing, labels, metrics, probe folds, steering hooks, demo
  helpers. Run with `uv run pytest`.
- [x] `ruff` lint and format: `uv run ruff check .`.

## Known limitations that affect reproduction

- **Mac memory.** With 24 GB, long runs can swap. Phase 4 step-by-step generation ran about 4×
  slower than expected for this reason. Close other apps, and keep `torch.mps.empty_cache()`
  between batches.
- **Old float16 captures.** Captures before 2026-11-15 stored residuals as float16. Layer-29
  channel 443 overflows in some rows, and the probes drop non-finite channels without looking at
  labels. Newer captures store bfloat16.
- **Restricted MPIB text.** MPIB's restricted payload text is never written to reports, logs
  meant for sharing, the demo or the poster.
