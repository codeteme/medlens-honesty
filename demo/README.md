# MedLens demo

A static web page: what MedGemma 4B *says* (its step-by-step reasoning) next to what *actually
drove* its answer (the no-hint comparison, the LLM judge's check for a mention of the hint, and
a warning badge from the Phase 3 activation detector). Toggle the hint type and whether it
points to a wrong or the right answer. Everything is precomputed; nothing here runs a model.

Build the data, then serve this folder (browsers block `fetch` from `file://`):

```bash
PYTHONPATH=scripts uv run python scripts/phase5_score_runs.py          # Mac, ~15 min for 40 questions
PYTHONPATH=scripts uv run python scripts/phase5_build_demo.py outputs/phase5_score_runs/<run>
cd demo && python -m http.server 8000                                  # open http://localhost:8000
```

`demo/data/` is committed so the page works as-is; rebuild it after changing the scripts.
Links: `#q=<question id>&c=<condition>` opens a specific case.
