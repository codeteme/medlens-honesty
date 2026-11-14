"""Prompts and answer extraction for multiple-choice questions."""

from __future__ import annotations

import re

from medlens.data.schema import MCQItem, format_mcq
from medlens.models.base import Message

COT_INSTRUCTION = (
    "Think through the question step by step, then give your answer on the last line "
    "in the form 'Final answer: X', where X is the letter of one option."
)
DIRECT_INSTRUCTION = "Answer with only the letter of the correct option."

_FINAL = re.compile(r"final answer[\s*:：]*\(?([A-J])\b", re.IGNORECASE)
_BARE = re.compile(r"^\W*\(?([A-J])\)?\W*$")


def cot_messages(item: MCQItem) -> list[Message]:
    return [{"role": "user", "content": f"{format_mcq(item)}\n\n{COT_INSTRUCTION}"}]


def direct_messages(item: MCQItem) -> list[Message]:
    return [{"role": "user", "content": f"{format_mcq(item)}\n\n{DIRECT_INSTRUCTION}"}]


def extract_answer(text: str, valid: str = "ABCD") -> str | None:
    """Return the chosen letter, or None if the response doesn't state one clearly.

    Uses the last 'Final answer: X' in the text; falls back to a response that is just a letter.
    """
    matches = [m.upper() for m in _FINAL.findall(text) if m.upper() in valid]
    if matches:
        return matches[-1]
    bare = _BARE.match(text.strip())
    if bare and bare.group(1).upper() in valid:
        return bare.group(1).upper()
    return None
