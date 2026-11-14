# medlens-honesty

Research project for the Duke Explainable AI final project, built across several sessions, one phase at a time.

## Project context

**The question.** Medical AI models write out their reasoning before answering, but sometimes that reasoning hides what actually drove the answer. Example: the model switches its answer because the prompt says "an attending thinks it's B," but its reasoning never mentions that. We ask three things:
1. Can we catch this by looking inside the model?
2. Can we fix it?
3. Does showing clinicians a warning help them judge when to trust the model?

**The models, split by what we can see:**
- **Inside view: MedGemma 4B** (`google/medgemma-4b-it`). We can read its internal activity, because it runs on a laptop and Google's Gemma Scope 2 tools are built for its size. All the "looking inside" work happens here. It also handles images, but we use it with text only.
- **Outside view: MedGemma 27B** (`google/medgemma-27b-text-it`), **Qwen 27B** (via Tinker), and **2–3 frontier API models**. For these we only see inputs and outputs. They tell us whether 4B's behavior is typical or a quirk of a small model.
- **Control:** `google/gemma-3-4b-it` (no medical fine-tuning), used to measure how well Gemma Scope 2 fits MedGemma.

**Hardware decision (2026-11-14):** development runs on an Apple M4 with 24 GB of memory, no CUDA GPU, and limited free disk. 27B can't run there at full precision. Tinker (Thinking Machines, $500 credit) has no Gemma models and exposes no activations, so it's used only for black-box sampling of open models. MedGemma 27B will be used for black-box tests only, through a host we haven't picked yet. The first option to test is a 4-bit version running locally.

**Tools and sources:**
- Gemma Scope 2 (Google DeepMind, Dec 2025): SAEs and transcoders for Gemma 3. The SAELens release names use `-res`, `-transcoders`, `-mlp`, `-att` and their `-all` variants (not `-resid_post`). The 4B subset layers are 9, 17, 22 and 29. Download only `params.safetensors`, not `examples.safetensors`.
- Tan et al., AAAI 2026, "Efficient Transcoder Adaptation for Fine-Tuned Models." No public code was found. Their fine-tuning step (with attention frozen) likely can't be reused because Google did MedGemma's fine-tuning. Only their transcoder-adaptation step may apply.
- Afolabi et al., ML4H 2026 (arXiv 2603.13988): black-box probes for causal ablation, positional bias and hint injection. No public code was found, so we reimplement from the paper.
- Prompt-format perturbations, "inspired by Modi et al., medRxiv 2026." That paper is still unverified; we're waiting on a link from the user.
- MPIB (Lee et al., Findings of EMNLP 2026): data at `jhlee0619/mpib` on Hugging Face (gated), code at `jhlee0619/mpib-eval` (no license file). It is built from MedQA, so deduplicate questions against our MedQA data. Never put its restricted attack text in the public demo.

**Main hypothesis (Phase 4):** the same internal features that make 4B follow hints without admitting it also make it vulnerable to prompt injection.

## Phases

0. **Setup.** Build the repo, load MedGemma 4B, and check how well Gemma Scope 2 works on it (base vs. instruction-tuned versions, with Gemma 3 4B as the control). Check that 4B answers enough MedQA questions correctly to be worth studying.
1. **Catch it from the outside (all models).** Add right or wrong hints framed as an authority figure, an insistent patient, or metadata, plus small rewordings of the question. Record whether the answer changes and whether the reasoning admits the hint. An LLM judge does the labeling, checked against about 100 hand labels. Label each case *honest*, *dishonest*, or *unaffected*, with rates and confidence intervals per model. **Compare 4B with 27B** to show the 4B findings are relevant to bigger models.
2. **Look inside (4B only).** Find the features behind hidden hint-following. Interpret them, then turn them off or patch them to confirm they cause the behavior. Adapt the SAEs following Tan et al. if Phase 0 shows they fit poorly.
3. **Build a detector (4B only).** Train a simple classifier on internal features to flag dishonest reasoning. Compare it with detectors that read only the written text. Split by question so no question appears in both training and test.
4. **Fix it, and test the injection link.**
   - **Fix (4B):** turn down the hint-following features, or boost features that make the model admit the hint. Measure the accuracy gained and the side effects on normal questions.
   - **Injection test:** run MPIB on all models (black-box) and on 4B with and without the fix.
   - **Verdict:** test the main hypothesis and report supported, not supported, or inconclusive.
5. **Demo.** A web app showing what the model *says* next to what *actually drove* the answer, with a warning badge, live hint toggles and a gallery of cases. Results are precomputed, so the public demo needs no GPU.
6. **Human study materials.** Protocol, consent form draft, survey and IRB exempt-determination text for 5–10 clinicians or students. Conditions: reasoning only, plus confidence, or plus the internal warning. The analysis script is tested on simulated data, clearly labeled as simulated.
7. **Write-up.** README, workshop paper draft (`reports/paper.md`), and a reproducibility checklist, using only real results.

**Known limitations to state in all reports:**
- The internals results come from 4B only. Comparing outside behavior suggests they're relevant to bigger models but doesn't prove 27B has the same features.
- 4B may get questions wrong because it doesn't know the answer rather than because it's hiding its reasoning.
- Gemma Scope 2 was trained on general Gemma 3 4B, not on MedGemma.

## Working rules (follow these in every phase)

1. **Verify, don't assume.** Before using any library, model ID, dataset ID, or API, check its current documentation or Hugging Face page. This includes SAELens, TransformerLens, nnsight, the Gemma Scope 2 release names, MedGemma model IDs, and MPIB. If something doesn't exist or has changed, tell me rather than guessing. Check whether the Afolabi et al. and Tan et al. code is public before reimplementing anything.
2. **Never fabricate results.** Every number in a report, figure, or README must come from a script that ran successfully. If an experiment fails or is inconclusive, report that honestly; negative results are fine.
3. **Stop at the end of each phase.** Summarize what was built, what ran, the key results, any problems, and what you recommend for the next phase. Then wait for me to say "Begin Phase N."
4. **Start small, then scale.** Every experiment gets a `--debug` mode, roughly 20 examples on the small model, that runs in minutes. Only scale up after debug mode works.
5. **Config-driven and reproducible.** Use YAML configs, fixed random seeds, and saved outputs (JSONL/Parquet) with the config that produced them. Cache model outputs and activations so nothing expensive is recomputed unnecessarily.
6. **Secrets.** API keys and the Hugging Face token go in `.env` and never get committed. Add `.env.example` and a proper `.gitignore`.
7. **Track cost.** Log API token usage and estimated cost per run. Warn me before any run expected to cost more than $50 or take more than 2 GPU-hours.
8. **Code quality.** Python 3.11+, type hints, `uv` or `pip` with a lockfile, `ruff`, and `pytest` tests for the data processing and metric code. Commit with clear messages at logical checkpoints.
9. **Plain English in docs.** The README and reports should be understandable to someone outside the class.
