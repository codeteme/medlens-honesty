# medlens-honesty

**Does a medical AI's written reasoning match what actually drove its answer?**

Medical AI models often explain their reasoning before answering. Sometimes that explanation
leaves out what really mattered. For example, a model may switch answers because the prompt
said "an attending thinks it's B," without ever mentioning that. This project tries to
(1) catch these cases by looking inside the model, (2) fix them, and (3) test whether
showing clinicians a warning helps them decide when to trust the model.

- **Inside view:** MedGemma 4B, analysed with Google DeepMind's Gemma Scope 2 tools.
- **Outside view:** MedGemma 27B and other models, where we only see inputs and outputs.

This is a Duke Explainable AI final project. It is built in phases; see `reports/` for each
phase's results. **Status: Phase 0 complete (see `reports/phase0.md`).**

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
