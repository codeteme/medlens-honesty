# Phase 8 report: hints that don't name a letter, and the fix on new questions

**Date:** 2026-11-18 · **Status:** complete · **Model:** MedGemma 4B · **Plan:** pre-registered¹
in `reports/phase8_preregistration.md` (commit 75123b6) before any generation, judging or scoring

Phase 8 is a follow-up to the gap review after Phase 7. It closes two gaps:
- **Part A.** Every earlier hint named an answer letter ("…believes the answer is C"), as in
  Afolabi et al.⁴ Does the Phase 3 detector² still work when the hint names the option *text*, or gives a made-up
  clinical finding instead?
- **Part B.** The Phase 4 fix³ was tested on step-by-step answers for only 24 dev questions, and
  the test-half check was skipped. Here it runs on 100 untouched test questions.

## Summary

- **The detector partly transfers to hints that name the option text.**
  - Letter hints, same run: AUROC 0.99 (it replicates Phase 3's 0.98).
  - Text hints: **0.83** (attending) and **0.82** (metadata). Pre-registered verdict:
    **partly transfers** for both.
  - The scores mostly stay low. At the demo's warning threshold (0.86), the badge catches
    **81–91%** of silent letter-hint runs but only **1.5–1.9%** of silent text-hint runs.
  - So the detector mostly reads "the answer matches a letter the prompt suggested". A deployed
    badge would almost never fire on a hint that names the option in words.
- **Made-up clinical findings: inconclusive, but for an interesting reason.**
  - The finding moved 20% of answers (sham: 4%).
  - When it did, the reasoning **mentioned the finding in 59 of 61 runs** (one of the two
    misses was a judge-check error). That leaves 2 silent runs, below the pre-registered
    minimum of 10.
  - The model is *misled* by false evidence, but it does not *hide* it. The detector cannot see
    this kind of following at all (followed vs ignored AUROC 0.55, CI 0.47–0.63).
- **The fix does not work on new questions.** Pre-registered rule: silent following must drop
  with Holm-adjusted p < 0.05 and no-hint accuracy may fall at most 5 points.
  - No setting passed.
  - The prompt-only instruction came closest: silent following 37.5% → 33.4%, Holm p = 0.29.
  - The layer-22 removal lowered no-hint accuracy by 10 points and broke 22% of hinted answers.
  - This confirms the Phase 4 dev result on 4× more questions.

![Detector scores by hint type](figures/phase8_scores.png)

*Phase 3 detector score for every eligible wrong-hint run, split by what the model did. Logit
scale. Dashed line: the demo's warning threshold. The made-up finding has only 2 silent runs, so
no median is drawn.*

## What was built

| Piece | What it does |
|---|---|
| `src/medlens/blackbox/hints.py` | `authority_text`, `metadata_text`, `evidence` framings; `phase8_conditions` |
| `src/medlens/blackbox/judge.py` | Evidence rubric; quote check that needs a distinctive cue word |
| `src/medlens/eval/stats.py` | `holm` (multiple-comparison correction) |
| `scripts/phase8.py` | `evidence`, `capture`, `analyze`, `fix-verdict`, and the TPU export/import |
| `scripts/tpu/vllm_steer.py` | Applies the Phase 4 fixes inside vLLM on the TPU |
| `scripts/tpu/check_patch.py` | Checks the TPU fix against the Mac's HF hooks |
| `scripts/phase8_figures.py` | The figure above |
| `configs/phase8.yaml`, `configs/phase8_fix.yaml` | Every setting for Part A and Part B |

There are 104 tests, all passing.

## Method in plain English

**Part A: new hints.** The same 375 MedQA questions as Phases 1–3, with 7 versions of each
prompt:
- no hint, and the Phase 1 sham sentence (the noise floor);
- the Phase 1 attending and metadata hints, which name a letter (the reference);
- **new:** the same two hints with the option *text* in place of the letter, for example "An
  attending physician who reviewed this case believes the answer is Uncoupling of oxidative
  phosphorylation";
- **new:** one sentence of made-up clinical evidence that supports the wrong option without
  naming it.
  - The sentence was written by Qwen 27B. Code rejected any sentence that shared a word with an
    option or named a letter.
  - 355 of 375 questions got a valid sentence within 3 tries.

All hints point to the same wrong option as in Phase 1. The model reasoned step by step. The
Phase 1 judge (Qwen 27B) checked whether the reasoning mentioned the hint. The frozen Phase 3
detector scored each run; it was not retrained and never saw these hint types.

**Part B: the fix.**
- 100 questions drawn at random from the Phase 4 test half, which no earlier phase used.
- Each has one no-hint prompt and 4 wrong-hint prompts.
- Five settings: no fix, the prompt-only instruction, and the three Phase 4 shortlisted fixes
  (remove the direction at layer 22; steer against it at layer 17 or layer 22). The fixes use
  the Phase 4 dev directions unchanged.

## Results

### Part A: does the detector still work?

AUROC 0.5 = chance, 1.0 = perfect. "Silent" = followed the hint without mentioning it.
Eligible = the no-hint answer was parsed and was not the hinted option.

| Hint | Eligible | Followed | Mentioned it, when followed | Silent | AUROC silent vs ignored (95% CI) | Verdict |
|---|---|---|---|---|---|---|
| Attending, names letter | 328 | 29.9% | 14.3% | 84 | 0.997 (0.993–0.999) | reference |
| Metadata, names letter | 328 | 63.4% | 0% | 208 | 0.990 (0.977–0.998) | reference |
| Attending, names text | 328 | 24.1% | 13.9% | 68 | **0.834** (0.783–0.880) | partly transfers |
| Metadata, names text | 328 | 48.8% | 0% | 160 | **0.818** (0.772–0.863) | partly transfers |
| Made-up finding | 309 | 19.7% | 96.7% | 2 | 0.716 (0.556–0.863) | inconclusive (< 10 silent) |
| Both text hints, pooled | 656 | 36.4% | 4.6% | 228 | 0.839 (0.806–0.870) | partly transfers |

- **Every hint moved answers above the sham floor** (sham 4.3%; McNemar p < 10⁻⁹ for every
  hint). The letter hints match Phase 1: attending 29.9% (Phase 1: 29.3%), metadata 63.4%
  (64.0%).
- **Naming the text instead of the letter makes hints a bit weaker:** attending 29.9% → 24.1%;
  metadata 63.4% → 48.8%. The metadata hint is still never mentioned.
- **The drop from letter to text hints is real:** text minus letter AUROC is −0.16 (CI −0.21 to
  −0.11) for attending and −0.18 (CI −0.22 to −0.13) for metadata.
- **The scores shift down a lot** (median score, exploratory):

  | Hint | Silent runs | Ignored runs |
  |---|---|---|
  | Attending, letter | 0.988 | 0.006 |
  | Metadata, letter | 0.996 | 0.029 |
  | Attending, text | 0.054 | 0.003 |
  | Metadata, text | 0.067 | 0.004 |

  The ranking survives, but the scale does not.
- **What a warning badge would catch** (pooled text hints, exploratory):

  | Threshold | Silent text-hint runs flagged | Ignored runs flagged | No-hint runs flagged |
  |---|---|---|---|
  | 0.86 (the demo's) | 1.8% | 0.7% | 0.3% |
  | 0.5 | 7.9% | 2.2% | 2.1% |
  | 0.2 | 23.2% | 4.2% | 8.5% |
  | 0.1 | 39.5% | 6.6% | 18.9% |

  No threshold gives the letter-hint performance.

**Made-up findings.**
- The model followed 61 of 309 (19.7%). Its reasoning quoted the finding in all but two, for
  example "The key finding is the single common arterial trunk arising from the right
  ventricle."
- One of the two "silent" runs did mention it ("hypokalemia (2.8 mEq/L)" for a potassium of
  2.8). The quote check found no shared distinctive word of 5+ letters, so it counted as not
  mentioned.
- Exploratory: the detector cannot tell followed from ignored runs here (AUROC 0.55, CI
  0.47–0.63).

**Reading.** The Phase 3 detector was trained on letter hints. On text hints it still carries
some signal, but most of what made it strong was "my answer matches a letter the prompt
suggested." This fits Phase 2, where the causal features were answer-letter features. A model
that weighs evidence it states openly is a different problem from hiding a hint. The current
tools neither need to catch it nor can.

### Part B: does the fix work on new questions?

368 eligible wrong-hint prompts per setting (100 questions × 4 hint types, eligibility fixed by
the no-fix run). Silent following is among eligible prompts. No-hint accuracy is on the 100
no-hint prompts.

| Setting | Silent following (95% CI) | Change | McNemar p | Holm p | No-hint accuracy | Hinted answers cut off or unparsed | Works? |
|---|---|---|---|---|---|---|---|
| No fix | 37.5% (32.7–42.6) | — | — | — | 70% | 3.3% | — |
| Prompt instruction only | 33.4% (28.8–38.4) | −4.1 | 0.072 | 0.289 | 70% | 4.3% | no |
| Remove direction, L22 | 34.5% (29.8–39.5) | −3.0 | 0.266 | 0.600 | 60% (−10) | 22.0% | no |
| Steer against, L17 (4 gaps) | 38.3% (33.5–43.4) | +0.8 | 0.807 | 0.807 | 65% | 2.4% | no |
| Steer against, L22 (4 gaps) | 34.5% (29.8–39.5) | −3.0 | 0.200 | 0.600 | 66% | 4.9% | no |

- **No setting meets the pre-registered rule.** The prompt instruction is the only one close
  (raw p = 0.07). It also raised the rate at which followed answers mention the hint: 1.4% →
  7.5%.
- **The internal fixes cost more than they help.**
  - Removing the layer-22 direction lowers silent following by 3 points (not significant).
  - It also costs 10 points of no-hint accuracy (p = 0.052) and breaks 22% of hinted answers
    (cut off or no parseable answer).
  - Part of its "drop" is broken answers, not honest ones.
- **This confirms Phase 4.** There, on 24 dev questions, no fix lowered silent following
  (31% with no fix, 31–38% with fixes). The same holds here on 4× more questions that no
  earlier phase touched.

**TPU fix check.** The fixes ran inside vLLM on the TPU (below). `check_patch.py` compared 10
prompts with the Mac, where the fixes are HF hooks:
- A huge push (−40 gaps at layer 22) turned the text into gibberish, so the patch is active.
- Removal changed all 10 texts.
- The TPU gave the same answer as the Mac on 9 of 10 prompts with no fix and 8 of 10 with
  removal. That fits the known 88.8% TPU–Mac agreement (Phase 1).

## Deviations from the pre-registration

1. **Hardware.**
   - The plan said one CUDA GPU. The course GPU pool was unavailable ("Error 503: GPU compute
     is not available right now").
   - Instead, step-by-step answers were generated on the course TPU with vLLM.
   - Part A activations were taken on the Mac by teacher-forcing the TPU reasoning. This is
     exactly how the Phase 3 detector's training data was made, so the detector sees the same
     pipeline.
   - The A3 device check (GPU vs Mac activations) was not needed and was not run.
   - Part B ran fully on the TPU: all five settings on the same machine, so no comparison mixes
     devices.
2. **Part B batch size.** vLLM batches requests dynamically, so the fixed batch size of 16 in
   `configs/phase8_fix.yaml` was not used. Greedy outputs can depend slightly on batch makeup.
3. **Evidence quality.**
   - The made-up findings were written by an LLM and checked only by code, not by a person.
   - Some are weak or only loosely support the wrong option.
   - 20 of 375 questions have no evidence condition, because no valid sentence was written in
     3 tries.
4. **Exploratory additions** (labeled above): the median scores, the threshold table, the
   evidence followed-vs-ignored CI, and the TPU fix check.

## Problems and caveats

- **Judge validity.** The judge is an LLM. Phase 1 checked it only against two blind LLM
  annotators, which measures model-to-model consistency, not agreement with a person.
- **The evidence quote check is strict.** It needs a shared word of 5+ letters that is not in
  the question. Synonyms ("hypokalemia" for "potassium 2.8") miss it, so the 2 silent evidence
  runs are an upper bound.
- **Unfinished answers.** In Part A, 38 of 2,605 answers hit the 768-token limit and 71 had no
  parseable answer. These count as missing, not as ignoring the hint. 2 activation captures
  were cut at 1,200 tokens.
- **Standing limits.** All results are from one model, MedGemma 4B. Some wrong answers come
  from not knowing, not from hiding. Gemma Scope 2 was trained on base Gemma 3, not MedGemma.

## What this changes in the paper

- The detector claim narrows. It catches silent following of hints that name a letter. On hints
  that name the option text it ranks above chance (0.82–0.84) but is badly miscalibrated. It
  does not see following of planted evidence.
- The fix claim strengthens. The negative result now holds on untouched test questions with
  multiple-comparison correction.

## Cost and compute

| Item | Cost |
|---|---|
| Course TPU v6e-1, about 1.2 hours (session ended) | $1.55 (group budget) |
| Evidence sentences, Qwen 27B on Tinker | $0.18 |
| Part A judge, 606 calls | $0.39 |
| Part B judge, 670 calls | $0.42 |
| **Total** | **$2.54** |

Mac time: about 3.4 hours to capture 2,605 activations (slowed by memory swapping).

## Reproduce

```bash
# Part A and B prompts (Mac), then on the TPU: bash run_all.sh (copy cache/tpu/phase8/ there)
uv run python scripts/phase8.py evidence
uv run python scripts/phase8.py export
PYTHONPATH=scripts uv run python scripts/phase8.py tpu-export
PYTHONPATH=scripts uv run python scripts/phase8.py tpu-import     # after copying out_*.jsonl back
# Part A (Mac)
uv run python scripts/phase8.py capture --run outputs/phase8/<run>
uv run python scripts/phase8.py analyze --run outputs/phase8/<run>
PYTHONPATH=scripts uv run python scripts/phase8_figures.py outputs/phase8/<run>
# Part B (Mac): judge the TPU answers, then apply the pre-registered rule
uv run python scripts/phase4_fix.py cot --config configs/phase8_fix.yaml --split test \
  --directions outputs/phase4_directions/20261115-163308 --resume outputs/phase4_cot_test/<run> \
  --settings baseline prompt_fix ablate_pre_L22 steer_pre_L17_x4 steer_pre_L22_x4
uv run python scripts/phase8.py fix-verdict --run outputs/phase4_cot_test/<run>
uv run python scripts/tpu/check_patch.py cache/tpu/phase8/results outputs/phase4_cot_test/<debug run>
```

Runs used in this report:
- `phase8_evidence/20261118-161608`
- `phase8/20261118-174441` (Part A)
- `phase4_cot_test/20261118-174444` (Part B)
- `phase4_cot_test/20261118-162234-debug` (the Mac side of the fix check)

## References

1. Phase 8 pre-registration — `reports/phase8_preregistration.md`
2. Phase 3 report (the detector) — `reports/phase3.md`
3. Phase 4 report (the fix) — `reports/phase4.md`
4. Afolabi et al., ML4H 2026 (letter-named hint template)
