"""Common format for multiple-choice medical questions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class MCQItem:
    """One multiple-choice question, normalized across datasets.

    `options` maps a letter ("A", "B", ...) to the option text, in display order.
    `answer` is the letter of the correct option.
    """

    id: str
    source: str
    question: str
    options: dict[str, str]
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer not in self.options:
            raise ValueError(
                f"{self.id}: answer {self.answer!r} not in options {list(self.options)}"
            )
        if list(self.options) != list(LETTERS[: len(self.options)]):
            raise ValueError(f"{self.id}: option keys must be A, B, C... in order")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_mcq(item: MCQItem) -> str:
    """Render a question and its lettered options as plain text."""
    lines = [item.question.strip(), ""]
    lines += [f"{letter}. {text.strip()}" for letter, text in item.options.items()]
    return "\n".join(lines)
