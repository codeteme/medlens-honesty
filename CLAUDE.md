# medlens-honesty

Research project for the Duke Explainable AI final project, built across several sessions, one phase at a time.

## Project context

**The question.** Medical AI models write out their reasoning before answering, but sometimes that reasoning hides what actually drove the answer. Example: the model switches its answer because the prompt says "an attending thinks it's B," but its reasoning never mentions that. We ask three things:
1. Can we catch this by looking inside the model?
2. Can we fix it?
3. Does showing clinicians a warning help them judge when to trust the model?

**The models, split by what we can see:**
- **Inside view: MedGemma 4B** (`google/medgemma-4b-it`). We can read its internal activity, because it runs on a laptop and Google's Gemma Scope 2 tools are built for its size. All the "looking inside" work happens here. It also handles images, but we use it with text only.
- **Scope (decided 2026-11-14): every phase runs on MedGemma 4B.** The one outside-view comparison is **Qwen 27B** (`Qwen/Qwen3.8-27B` via Tinker), run as an **optional final check** with the same Phase 1 hint test once the 4B pipeline works, to show the behavior isn't just a small-model quirk. **Deferred** (add only if time allows): MedGemma 27B (`google/medgemma-27b-text-it`) and frontier API models.
- **Control:** `google/gemma-3-4b-it` (no medical fine-tuning), used to measure how well Gemma Scope 2 fits MedGemma.

**Hardware decision (2026-11-14):** development runs on an Apple M4 with 24 GB of memory, no CUDA GPU, and limited free disk. 27B can't run there at full precision. Tinker (Thinking Machines, $500 credit) has no Gemma models and exposes no activations, so it's used only for black-box sampling of open models. MedGemma 27B is deferred (see Scope).

**Compute decision (2026-11-14, user chose "A first, B as fallback"):**
- **A (default):** no Colab. MedGemma 4B generation is sped up with batching on the Mac. Tinker's `Qwen/Qwen3.8-27B` is the optional comparison model.
- **B (fallback):** add Google Colab if batched 4B generation is too slow. The user's Colab tier is still unknown.
- At the start of Phase 1, benchmark batched 4B generation before launching large runs.
- The Tinker key works. Its model list (checked 2026-11-14) has 38 models and no Gemma or MedGemma. `openai/gpt-oss-20b` and `-120b` are available; it is untested whether Tinker returns their raw reasoning.
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GOOGLE_API_KEY` are still empty.
- **Course TPU (added in Phase 1):** start sessions with `ece661 compute start --shape v6e-1 --hours N`. It costs about $1.35/h from the group's shared budget, so ask before starting. `vllm-tpu==0.33.0` runs MedGemma 4B at about 4,600 tokens/sec, against 38 tokens/sec batched on the Mac.
  - The TPU gives *nearly* the same outputs as the Mac: 88.8% same answers on 500 prompts, and 99.4% teacher-forced token agreement versus 99.7% for the Mac's own answers.
  - Never mix devices within one comparison.
  - Phase 2+ re-verifies TPU-labeled cases on the Mac (HF), where the hooks are.
  - Delete the HF token from the machine and stop the session when done.

**Tools and sources:**
- Gemma Scope 2 (Google DeepMind, Dec 2025): SAEs and transcoders for Gemma 3. The SAELens release names use `-res`, `-transcoders`, `-mlp`, `-att` and their `-all` variants (not `-resid_post`). The 4B subset layers are 9, 17, 22 and 29. Download only `params.safetensors`, not `examples.safetensors`.
- Tan et al., AAAI 2026, "Efficient Transcoder Adaptation for Fine-Tuned Models." No public code was found. Their fine-tuning step (with attention frozen) likely can't be reused because Google did MedGemma's fine-tuning. Only their transcoder-adaptation step may apply.
- Afolabi et al., ML4H 2026 (arXiv 2603.13988): black-box probes. **Code is public:** `github.com/comp-med-research/med-llm-faithfulness`. It has **no license**, so read it but don't copy it; reimplement and cite. What the paper actually did:
  - **Hint injection:** one explicit template, "Hint: the correct answer is X.", placed after the options. Their wrong hint was always the first non-gold letter. Acknowledgment was detected with regexes only.
  - **Positional bias:** 3 worked examples whose answers are all at B, then the test question with either the correct or a wrong option at B. This is not option shuffling.
  - **Causal ablation:** the model returns JSON reasoning steps, each with an exact quote from the question; each quoted span is replaced with "[REDACTED]" one at a time. Metrics: Causal Density, Damage, Rescue, and Net Flip = Damage − Rescue.
  - They used 100 questions from 5-option MedQA (`med_qa_us`); we use 4-option MedQA.
- Prompt-format perturbations, inspired by Modi et al., "Understanding Clinical Reasoning Variability in Medical Large Language Models: A Mechanistic Interpretability Study," medRxiv 2026.01.26.26344845v2 (verified 2026-11-14 from the user's PDF). They varied prompt format (full assessment-and-plan vs direct question) and removed single medical terms (relevant vs irrelevant) on oncology cases, with MedGemma-27B-Text-IT among the models. They trained their own SAEs on MIMIC-IV notes. The preprint is "all rights reserved": cite it, but don't copy its text or cases.
- MPIB (Lee et al., Findings of EMNLP 2026): data at `jhlee0619/mpib` on Hugging Face (gated), code at `jhlee0619/mpib-eval` (no license file). It is built from MedQA, so deduplicate questions against our MedQA data. Never put its restricted attack text in the public demo.

**Relation to the user's earlier project, Silent Second Opinion (SSO)** (`../silent-second-opinion`, AIPI 590, Oct 2026). SSO asked whether MedGemma **1.5** 4B discloses MedMisBench cues (a fabricated claim vs a bare assertion, plus a sham baseline) in its thinking trace and response. It was black-box only, used a Qwen3-14B judge validated against LLM annotators only, and had no human labels.
- **Decision (2026-11-14, Option 2):** keep MedGemma 4B **v1** (`google/medgemma-4b-it`). Do **not** reuse SSO's data or rollouts. Cite SSO as prior work and state what is new: internals (Phases 2–4), an activation-based detector, steering, the injection link, human-validated judge labels, and the clinician study.
- **SSO lessons to apply in Phase 1:**
  - a sham-sentence baseline as the noise floor
  - "eligible" = the no-hint answer is parsed and isn't the target; unfinished runs are missing data, not resisted cues
  - parse the last answer statement
  - the judge must quote verbatim, and code checks that the quote is in the text
  - the judge cache is keyed by a hash of the rubric
  - verbalization is an imperfect proxy for faithfulness (Zaman & Srivastava, ACL 2026)
  - authority framing roughly doubled cue-following in SSO
- SSO's judge code may be adapted with attribution, since it is the user's own code.
- **Closest outside work:** Linzmayer & Elhadad, arXiv 2609.02754 (ML4H 2026 submission). Their "mechanism" analysis is text resampling (truncating and transplanting the trace), not internals. No Gemma models, no activation-based detector, no interventions, and no code link found.
- **Course policy:** the user confirmed (2026-11-14) that the final project may build on the SSO assignment.

**Main hypothesis (Phase 4):** the same internal features that make 4B follow hints without admitting it also make it vulnerable to prompt injection.

## Phases

0. **Setup.** Build the repo, load MedGemma 4B, and check how well Gemma Scope 2 works on it (base vs. instruction-tuned versions, with Gemma 3 4B as the control). Check that 4B answers enough MedQA questions correctly to be worth studying.
1. **Catch it from the outside (MedGemma 4B).**
   - Add right or wrong hints framed as an authority figure, an insistent patient, or metadata.
   - Add a sham-sentence baseline, Afolabi-style position bias (worked examples answered at B) and causal ablation, and Modi-style format and term-removal perturbations.
   - Record whether the answer changes and whether the reasoning admits the hint. An LLM judge labels each case *honest*, *dishonest*, or *unaffected*. No Anthropic key is set yet, so the judge runs on Tinker or locally.
   - **Judge validation (changed 2026-11-14 at the user's request):** instead of hand labels, two independent, blind Opus 5.5 annotators (fresh subagents that see only the label sheet) label the sheet. Agreement with them measures **model-to-model consistency, not validity against a person**, the same gap SSO had. Every report must say so. Hand labels for a small subset remain an option.
   - Report rates with confidence intervals per hint type.
   - **Start with a pilot:** 50 questions × 7 conditions, after speed tests. Scale up only if the pilot works.
2. **Look inside (4B only).** Find the features behind hidden hint-following. Interpret them, then turn them off or patch them to confirm they cause the behavior. Adapt the SAEs following Tan et al. if Phase 0 shows they fit poorly.
3. **Build a detector (4B only).** Train a simple classifier on internal features to flag dishonest reasoning. Compare it with detectors that read only the written text. Split by question so no question appears in both training and test.
4. **Fix it, and test the injection link.**
   - **Fix (4B):** turn down the hint-following features, or boost features that make the model admit the hint. Measure the accuracy gained and the side effects on normal questions.
   - **Injection test:** run MPIB on 4B with and without the fix, and check whether the Phase 2 features activate on injection cases.
   - **Verdict:** test the main hypothesis and report supported, not supported, or inconclusive.
5. **Demo.** A web app showing what the model *says* next to what *actually drove* the answer, with a warning badge, live hint toggles and a gallery of cases. Results are precomputed, so the public demo needs no GPU.
6. **Human study materials.** Protocol, consent form draft, survey and IRB exempt-determination text for 5–10 clinicians or students. Conditions: reasoning only, plus confidence, or plus the internal warning. The analysis script is tested on simulated data, clearly labeled as simulated.
7. **Write-up.** README, workshop paper draft (`reports/paper.md`), and a reproducibility checklist, using only real results.

**Known limitations to state in all reports:**
- All results come from one model, MedGemma 4B. The optional Qwen 27B check speaks only to outside behavior; it can't show that bigger models have the same internal features.
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
