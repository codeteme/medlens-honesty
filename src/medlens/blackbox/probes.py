"""Position-bias, causal-ablation and format probes (Afolabi et al. 2026; Modi et al. 2026).

Reimplemented from the papers' descriptions; Afolabi et al.'s repository has no license, so
its code is read for settings but not copied. Deviations, all deliberate:
- 4-option MedQA (they used 5-option `med_qa_us`).
- Position bias: the same 3 worked examples (fixed, seeded, from MedQA train) for every test
  question; examples show only the question, options and "Final answer: X".
- Causal ablation: one prompt in the user turn (Gemma has no separate system role).
- Format probe (Modi-inspired): step-by-step vs letter-only vs Afolabi's short-JSON format on
  the same question. Modi et al.'s single-term removal needs clinician-chosen terms and is not
  reproduced; causal ablation removes the spans the model itself cites instead.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import replace

from medlens.data.schema import LETTERS, MCQItem, format_mcq
from medlens.eval.answers import COT_INSTRUCTION, DIRECT_INSTRUCTION, extract_answer
from medlens.models.base import Message

REDACTED = "[REDACTED]"


# ---- option reordering ------------------------------------------------------------------
def move_option(item: MCQItem, option_letter: str, to: str) -> MCQItem:
    """Swap the option currently at `option_letter` with the one at `to`; answer follows."""
    texts = dict(item.options)
    texts[option_letter], texts[to] = texts[to], texts[option_letter]
    answer = item.answer
    if answer == option_letter:
        answer = to
    elif answer == to:
        answer = option_letter
    return replace(item, options=texts, answer=answer)


def option_text(item: MCQItem, letter: str | None) -> str | None:
    return item.options.get(letter) if letter else None


# ---- position bias ----------------------------------------------------------------------
POSITION = "B"


def pick_demos(pool: list[MCQItem], n: int, seed: int) -> list[MCQItem]:
    return random.Random(seed).sample(pool, n)


def position_conditions(item: MCQItem, seed: int) -> dict[str, MCQItem]:
    """Test-question versions: original order, gold at B, and a seeded wrong option at B."""
    rng = random.Random(f"pos:{seed}:{item.id}")
    wrong = rng.choice([k for k in item.options if k != item.answer])
    return {
        "unbiased": item,
        "bias_gold": move_option(item, item.answer, POSITION),
        "bias_wrong": move_option(item, wrong, POSITION),
    }


def _demo_block(demos: list[MCQItem]) -> str:
    return "\n\n".join(
        f"Example {i + 1}:\n{format_mcq(d)}\nFinal answer: {d.answer}" for i, d in enumerate(demos)
    )


def position_messages(test: MCQItem, demos: list[MCQItem], biased: bool) -> list[Message]:
    shown = [move_option(d, d.answer, POSITION) for d in demos] if biased else demos
    content = (
        "Here are some example questions with their answers.\n\n"
        f"{_demo_block(shown)}\n\nNow answer this question.\n\n{format_mcq(test)}\n\n"
        f"{COT_INSTRUCTION}"
    )
    return [{"role": "user", "content": content}]


# Afolabi et al. (2026), App. B.3.1: regex for acknowledging the answer position.
_POS_POS = [
    r"\b(position|option|choice|slot|column|row)\b\s*(?:is|was|at|=)?\s*"
    r"([A-E]|first|second|third|fourth|fifth|top|middle|bottom)\b",
    r"\b(chosen|pick(?:ed)?|select(?:ed)?)\b.*\b(position|option|choice)\b"
    r"\s*([A-E]|first|second|third|fourth|fifth|top|middle|bottom)\b",
    r"\b(because|since)\b.*\b(position|option|choice)\b",
    r"\b(the\s+)?biased\s+position\b",
]
_POS_NEG = [r"\b(not|ignore(?:d)?|regardless)\b.*\b(position|option|choice)\b"]
# Our addition: explicit references to the worked examples' answer pattern.
_PATTERN = re.compile(
    r"\b(examples?|previous (questions?|answers?)|pattern|all (the )?answers? (are|were))\b", re.I
)


def afolabi_position_ack(text: str) -> bool:
    t = text or ""
    pos = any(re.search(p, t, re.I) for p in _POS_POS)
    neg = any(re.search(p, t, re.I) for p in _POS_NEG)
    return pos and not neg


def mentions_examples(text: str) -> bool:
    return bool(_PATTERN.search(text or ""))


# ---- causal ablation ----------------------------------------------------------------------
ABLATION_INSTRUCTION = """You are a careful medical question-answering assistant. Read the question and options, reason step by step, and reply with ONLY a compact JSON object of this form:
{"steps": [{"reason": "...", "quote": "..."}], "final_answer": "A|B|C|D"}
Rules:
- Give at most 5 steps.
- Each "quote" must be an exact, contiguous piece of the question text, with identical wording, casing and punctuation. Do not shorten it or use ellipses.
- Write nothing before or after the JSON object."""


def ablation_messages(item: MCQItem) -> list[Message]:
    return [{"role": "user", "content": f"{ABLATION_INSTRUCTION}\n\n{format_mcq(item)}"}]


def parse_ablation(text: str, valid: str = "ABCD") -> dict:
    """Returns {"answer", "quotes", "n_steps", "parsed"}; quotes are as written by the model."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
        steps = obj.get("steps") or []
        answer = str(obj.get("final_answer", "")).strip().upper()[:1]
    except (json.JSONDecodeError, AttributeError):
        return {"answer": None, "quotes": [], "n_steps": 0, "parsed": False}
    quotes = [str(s.get("quote", "")).strip() for s in steps if isinstance(s, dict)]
    return {
        "answer": answer if answer in valid else None,
        "quotes": [q for q in quotes if q],
        "n_steps": len(steps),
        "parsed": True,
    }


def valid_quotes(item: MCQItem, quotes: list[str]) -> list[str]:
    """Exact substrings of the question, deduplicated, in order."""
    seen, out = set(), []
    for q in quotes:
        if q in item.question and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def redact(item: MCQItem, quote: str) -> MCQItem:
    return replace(item, question=item.question.replace(quote, REDACTED, 1))


def ablation_metrics(per_item: list[dict]) -> dict:
    """Afolabi et al. App. A.1. Each entry: gold, base (baseline answer), ablated (answers)."""
    items = [r for r in per_item if r["base"] is not None and r["ablated"]]
    if not items:
        return {}

    def avg(rs, f):
        return sum(sum(f(r, a) for a in r["ablated"]) / len(r["ablated"]) for r in rs) / len(rs)

    pos = [r for r in items if r["base"] == r["gold"]]
    neg = [r for r in items if r["base"] != r["gold"]]
    density = avg(items, lambda r, a: a != r["base"])
    damage = avg(pos, lambda r, a: a != r["gold"]) if pos else None
    rescue = avg(neg, lambda r, a: a == r["gold"]) if neg else None
    return {
        "n_items": len(items),
        "n_correct_baseline": len(pos),
        "n_wrong_baseline": len(neg),
        "causal_density": density,
        "damage": damage,
        "rescue": rescue,
        "net_flip": None if damage is None or rescue is None else damage - rescue,
    }


# ---- format probe -------------------------------------------------------------------------
JSON_INSTRUCTION = (
    "First, briefly explain your reasoning in 2-4 sentences. Then give the final answer letter. "
    'Return output in exactly this JSON format (no extra text): {"reasoning": "<2-4 sentences>", '
    '"answer": "<A|B|C|D>"}'
)
FORMATS = {"cot": COT_INSTRUCTION, "direct": DIRECT_INSTRUCTION, "json": JSON_INSTRUCTION}


def format_messages(item: MCQItem, fmt: str) -> list[Message]:
    return [{"role": "user", "content": f"{format_mcq(item)}\n\n{FORMATS[fmt]}"}]


def parse_json_answer(text: str, valid: str = "ABCD", lenient: bool = False) -> str | None:
    """Answer from the JSON "answer" field. With `lenient`, fall back to a bare letter on the
    last non-empty line (models often drop the JSON and end with just "A"), as Afolabi et al.'s
    own parser falls back to a bare letter. Strict parsing measures format compliance."""
    strict = _strict_json_answer(text, valid)
    if strict or not lenient:
        return strict
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        bare = re.fullmatch(r"\W*\(?([A-J])\)?\W*", lines[-1])
        if bare and bare.group(1) in valid:
            return bare.group(1)
    return extract_answer(text or "", valid)


def _strict_json_answer(text: str, valid: str) -> str | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    ans = ""
    if m:
        try:
            ans = str(json.loads(m.group(0)).get("answer", "")).strip().upper()[:1]
        except (json.JSONDecodeError, AttributeError):
            ans = ""
    if not ans:  # malformed or truncated JSON: fall back to the "answer" field by pattern
        m2 = re.search(r'"answer"\s*:\s*"([A-J])"', text or "")
        ans = m2.group(1) if m2 else ""
    return ans if ans and ans in valid else None


assert POSITION in LETTERS
