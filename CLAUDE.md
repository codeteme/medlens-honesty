# medlens-honesty

Research project for the Duke Explainable AI final project, built across several sessions, one phase at a time.

## Project context

**The question.** Medical AI models write out their reasoning before answering, but sometimes that reasoning hides what actually drove the answer. Example: the model switches its answer because the prompt says "an attending thinks it's B," but its reasoning never mentions that. We ask three things:
1. Can we catch this by looking inside the model?
2. Can we fix it?
3. ~~Does showing clinicians a warning help them judge when to trust the model?~~ Dropped
   (2026-11-16): the user will not run a human study. The demo (Phase 5) shows the warning.

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
- **Decision (2026-11-14, Option 2):** keep MedGemma 4B **v1** (`google/medgemma-4b-it`). Do **not** reuse SSO's data or rollouts. Cite SSO as prior work and state what is new: internals (Phases 2–4), an activation-based detector, steering, the injection link, human-validated judge labels, and the demo.
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

**Phase 2 results to build on (2026-11-14, see reports/phase2.md):**
- Captured features and residuals for 375 questions × 6 conditions are in `outputs/phase2_capture/20261114-194828`: `resid_L{9,17,22,29}`, `features_L*`, and `readout.parquet`. Phase 3 should reuse them.
- The one-pass readout (prefill "Final answer:", read P(hinted letter)) predicts Phase 1 unfaithful vs unaffected with AUROC 0.83–0.91. It is the baseline for Phase 3.
- **Causal features are answer-letter / option-label features on the hint tokens:** for example 22/2710, 17/1894, 17/12508, 22/146 (pt 16k residual SAEs).
  - Removing the top 10 at layer 17 on the hint tokens changed the readout's log-P(hinted letter) by −0.83, and following fell from 87% to 75%.
  - In full step-by-step regeneration the decisions did NOT change (35/40 before and after).
  - Correlation-ranked features at the answer point are not causal (−0.07).
  - No authority features were found.
- **For Phase 4:** single-layer removal of 10 features is too weak for step-by-step answers. Try several layers at once, more features, negative steering, or attention masking to the hint's letter. Mac memory: call `torch.mps.empty_cache()` between cases (swap thrashing happened twice).
- **Related work found in Phase 2:**
  - Ayadi & Hautli-Janisz, ACL 2026 SRW (MedGemma-27B, Gemma Scope 2, False Consensus hint, letter-only answers): feature ablation gave 0% mitigation, and global steering at layer 29 fixed 21.9% of flips. GitHub code is in the paper.
  - Steering for cue acknowledgment: arXiv 2607.29062.
  - Activation-based unfaithfulness detectors: VERITAS (2610.23275), arXiv 2605.25603. Phase 3 must position itself against these.

**Phase 3 results to build on (2026-11-15, see reports/phase3.md):**
- Hint-blind linear probes (never see the hint text or letter), 5 folds grouped by question.
  - After reasoning (Phase 1 TPU reasoning teacher-forced on the Mac, read at the appended
    "Final answer:"), raw resid L22: AUROC 0.982, within-hint-type 0.978, ECE 0.015, no-hint
    false alarms 0.8% at 80% recall. Held-out hint types 0.98–0.99; held-out specialties 0.98–1.00.
  - Forced "Final answer:" before reasoning (Phase 2 capture), raw resid L22: AUROC 0.871, within type 0.821.
  - End-of-prompt probe (0.76) mostly reads hint type and question: question-only control 0.705.
  - Baselines: hint type alone 0.709; LLM judge (Qwen 27B, k=3) 0.717; TF-IDF 0.564; readout (uses letter) 0.893.
  - Raw activations beat SAE features at every layer and position.
- **The probe detects hint-following, not silence:** it flags 76% of the 21 runs that followed AND
  admitted the hint. Pair it with a text acknowledgment check for an unfaithfulness detector.
- Saved probes: `outputs/phase3_probes/20261115-153757/probe_{pre,post}.joblib`. Post capture:
  `outputs/phase3_capture_postcot/20261115-142843`.
- Captures before 2026-11-15 saved residuals as float16; channel 443 at L29 overflows to inf
  (probes drop non-finite channels). Captures now save bfloat16.
- For Phase 4: use the L22 post-reasoning probe as a cheap outcome measure, and try its weight
  vector as a steering direction; run it on MPIB injection cases for the injection link.

**Phase 4 results to build on (2026-11-16, see reports/phase4.md):**
- Fix: mean-difference directions (unfaithful − unaffected, dev half) removed or steered against at
  every token. Quick answers: hint-following 45% → 28% (remove, L22), random arrows at L22 do
  nothing, L17 is fragile (random arrows break it). Step-by-step (dev, 24 questions): **no fix
  lowered silent following** (31% no fix; 31–38% with fixes). Test-half confirmation and the MPIB
  fix arm were skipped (user decision; reported as deviations).
- MPIB V1 (150 cases, attached to original MedQA options): 26% of answers moved, accuracy
  unchanged (64% vs 65%); Phase 1 sham moves 14%. Phase 3 post probe on moved vs not: AUROC 0.48.
  Feature 22/146 is far more active on injected text (11.0 vs 0.06). Verdict: **inconclusive**
  (pre-registered), evidence leans against.
- For Phase 5: present the detector as "flags hint-following", not injection. Never show MPIB
  attack text. Runs: `outputs/phase4_{directions,screen,cot_dev,mpib}/`.

**Phase 5 results to build on (2026-11-16, see reports/phase5.md):**
- Static demo in `demo/` (vanilla HTML/JS, precomputed JSON in `demo/data/`, committed).
  Scores: `outputs/phase5_score_runs/20261116-171012` (first 40 of the 375 questions, all 10
  conditions; user chose 40 to save Mac time). Build: `phase5_build_demo.py`; serve with
  `python -m http.server` from `demo/`.
- Badge threshold 0.86 (80% recall, OOF over all 1,272 runs). It fires on 54% of right-answer
  hints the model already agreed with: it reads "answer matches a suggested letter".
- MedQA license is "unknown" on HF, but the **user confirmed (2026-11-16) MedQA questions may
  be shared**; `demo/data/` is committed. HAI-DEF terms allow showing precomputed outputs; keep
  the "AI-generated, not medical advice" notice. Never show MPIB attack text.
- Published as a private claude.ai artifact (owner shares it via the page's Share menu):
  https://claude.ai/artifact/Voko8MSzjQnDq1jkEMYypa. The published page is `demo/index.html`
  without the doctype/html/head/body tags, plus `style.css`, `app.js` and `data/`.
- For Phase 7: report the agreement finding (badge fires when the answer merely matches the hint).

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
6. ~~**Human study materials.**~~ **Dropped (2026-11-16):** the user does not plan to run a human study, so no protocol, consent form, survey or IRB text. After Phase 5, the next phase is Phase 7.
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
