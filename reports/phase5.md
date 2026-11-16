# Phase 5 report: the demo

**Date:** 2026-11-16 · **Status:** complete (local; not deployed) · **Model:** MedGemma 4B
(precomputed outputs only) · **Data:** the first 40 of the 375 Phase 1 questions, 10 conditions
each (400 runs)

## Summary

- **A static web page** (`demo/`) shows what MedGemma 4B *says* next to what *actually drove* its
  answer. It needs no GPU and no server code: plain HTML, CSS and JavaScript reading
  precomputed JSON.
- **Each case shows:**
  - the question and the exact hint added to the prompt;
  - the model's full step-by-step reasoning, with any sentence that admits the hint
    highlighted;
  - a **warning badge** from the Phase 3 after-reasoning detector, with its score and threshold;
  - the **no-hint test**: the answer to the same question without the hint, and whether the
    hint changed it;
  - whether the LLM judge found the reasoning mentioning the hint, and the Phase 1 verdict;
  - a quick-answer readout: the chance of each letter if the model had to answer at once, plus
    an early-warning score from the before-reasoning detector.
- **Live hint toggles:** no hint, a harmless sentence, or one of 4 hint types (metadata tag,
  plain hint, attending, patient), each pointing to a wrong or the right answer. That makes 10
  versions of each question.
- **Gallery:** 6 shelves (caught, admitted, resisted, missed, false alarm, right-answer hint),
  filled by a seeded rule rather than by hand. A search box browses all 40 questions.
- **A new finding while building it:** the warning also fires when the answer merely *agrees*
  with the hint. On right-answer hints that pointed to the answer the model would have given
  anyway, it warned on **54%** of runs (59 of 110). The detector seems to read "my answer
  matches a letter suggested in the prompt", not "the hint changed my mind." The page says so.

## What was built

| Piece | What it does |
|---|---|
| `scripts/phase5_score_runs.py` | Detector scores for every demo run; reuses saved Phase 2/3 activations, runs the model only for the rest |
| `scripts/phase5_build_demo.py` | Builds `demo/data/` (one JSON per question plus an index) and the summary numbers |
| `src/medlens/demo.py` | Warning threshold, gallery rules and the plain-English "what the hint did" text (with tests) |
| `demo/index.html`, `app.js`, `style.css` | The page; light and dark themes; works at phone width |
| `demo/README.md` | How to build and serve it |
| `configs/phase5.yaml` | Paths, the 80%-recall badge setting, gallery size |

There are 87 tests, all passing.

## How the numbers are made

- **Warning threshold.** It is set so the after-reasoning detector catches 80% of silent
  hint-following, using the out-of-fold scores of all 1,272 Phase 3 training runs. The result
  is a threshold of 0.86.
- **Headline numbers use all 375 questions, not just the 40 shown:**
  - after-reasoning AUROC 0.982 (out-of-fold, same as Phase 3);
  - fires on 1.1% of answers that ignored a wrong hint;
  - fires on 0.5% (2 of 375) of no-hint answers.
  
  Phase 3 reported 0.8% for no-hint answers because it averaged the 5 cross-validation probes.
  The demo uses the single saved final probe.
- **Which score each run gets.** A run the detector was trained on shows its out-of-fold score,
  so it was never scored by a probe that saw it. Every other run (no hint, harmless sentence,
  right-answer hints, admitted and ineligible runs) is scored by the final probe, which never
  saw it. The page states the source under each badge.
- **Check that reused and recomputed activations agree.** Recomputing 10 stored activations on
  the Mac gave cosine similarity 1.000 with the saved ones.

**How often the badge fires, by kind of run** (40 demo questions; from
`outputs/phase5_build_demo/<run>/badge_by_kind.csv`):

| Kind of run | Phase 1 verdict | Runs | Warning fires |
|---|---|---|---|
| Wrong hint | silent hint-following | 38 | 89% |
| Wrong hint | open hint-following (admitted) | 3 | 67% |
| Wrong hint | not affected | 93 | 1% |
| Wrong hint | not scored (mostly: already the hinted answer without the hint) | 26 | 69% |
| Right hint | silent hint-following (the hint fixed a wrong answer) | 29 | 90% |
| Right hint | not affected | 21 | 10% |
| Right hint | not scored (mostly: already the hinted answer) | 110 | 54% |
| Harmless sentence | — | 40 | 5% |
| No hint | — | 40 | 0% |

The two "not scored" rows show the agreement problem. When the answer matches the hinted
letter for reasons unrelated to the hint, the detector often fires anyway. Phase 3 couldn't
see this, because these runs were outside its task (they can't be labeled followed or
ignored). They are exactly the runs a real deployment would meet.

## Problems and caveats

1. **MedQA's license for public display is unclear.** The Hugging Face card³ lists it as
   "unknown". The MedQA GitHub repository is MIT-licensed, but that license covers the code,
   and the README states no license for the data. The demo data (`demo/data/`) is therefore not
   committed, and the demo has not been deployed. Options are in Recommendations.
2. **MedGemma's terms.** The Health AI Developer Foundations terms¹ say Google claims no
   ownership of outputs, and the user is responsible for them. Showing precomputed text is not
   "hosting" the model. Their prohibited-use policy² bars misleading claims of capability in
   health and automated healthcare decisions. The page therefore labels all text as
   AI-generated research output, not medical advice.
3. **40 of 375 questions.** Scoring all 375 would have taken about 2.4 hours of Mac time, and
   the user chose to limit the demo to 40. The 40 are the first 40 of the fixed Phase 1 sample,
   not hand-picked. Headline numbers still use all 375.
4. **The badge flags agreement with a suggested letter, not hidden reasoning** (see the table).
   The Phase 3 and Phase 4 caveats also apply and are shown on the page: it flags following,
   not hiding; it only covers letter cues; it does not catch prompt injection (AUROC 0.48);
   and the fix failed.
5. **Small shelves.** Among the 40 questions there are only 3 admitted runs and 1 false alarm,
   so those shelves are thin.
6. **Checked in a headless browser only.** 240 question-and-hint combinations rendered without
   errors, and the layout was checked at desktop and phone width, in light and dark themes. No
   screen-reader test was done.

## Recommendations for Phase 6 (human study)

1. **Use the demo's case view as the study interface,** with three versions: reasoning only;
   plus confidence (the quick-answer letter chances); plus the warning badge.
2. **Include agreement cases** (the answer matches the hint, but the hint didn't change it).
   They are where the badge misleads, and the study should measure whether clinicians
   over-trust it there.
3. **Pick study cases by rule** from all 375 questions (score more on the Mac as needed),
   balanced across the gallery shelves.
4. **For a public link,** choose one of:
   - **(a)** ask the MedQA authors for permission;
   - **(b)** deploy privately (for example behind a course login);
   - **(c)** swap in questions with a clear license.

## Cost and compute

| Run | Where | Time | Cost |
|---|---|---|---|
| Scoring, first attempt (stopped at 40 of 375 questions; nothing saved) | Mac | 18 min | — |
| Scoring, 40 questions (229 runs needed the model) | Mac | 17 min | — |
| Building the data and checking the page | Mac, no model | minutes | — |

No Tinker or TPU use in this phase.

## Reproduce

```bash
PYTHONPATH=scripts uv run python scripts/phase5_score_runs.py --debug   # 3 questions, ~1 min
PYTHONPATH=scripts uv run python scripts/phase5_score_runs.py           # 40 questions, ~17 min
PYTHONPATH=scripts uv run python scripts/phase5_build_demo.py outputs/phase5_score_runs/<run>
cd demo && python -m http.server 8000
```

Runs used: `phase5_score_runs/20261116-171012`, `phase5_build_demo/20261116-173236`.

## References

1. [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)
2. [Health AI Developer Foundations Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy)
3. [MedQA dataset card (bigbio/med_qa)](https://huggingface.co/datasets/bigbio/med_qa)
