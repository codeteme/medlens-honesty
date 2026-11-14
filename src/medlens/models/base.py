"""Common interface shared by local Hugging Face models and API models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None when the model is local or the price is unknown


@dataclass
class Generation:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    cached: bool = False  # True if served from the on-disk cache (no new cost incurred)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Generation:
        d = dict(d)
        d["usage"] = Usage(**d.get("usage", {}))
        return cls(**d)


class TextModel(Protocol):
    """Anything that turns a chat into text. Black-box probes only need this."""

    name: str

    def generate(self, messages: list[Message], max_new_tokens: int = 1024) -> Generation: ...


@dataclass
class CostTracker:
    """Accumulates token usage and cost over a run."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0

    def add(self, gen: Generation) -> None:
        self.calls += 1
        if gen.cached:
            self.cached_calls += 1
            return
        self.input_tokens += gen.usage.input_tokens
        self.output_tokens += gen.usage.output_tokens
        if gen.usage.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += gen.usage.cost_usd

    def summary(self) -> dict[str, Any]:
        return asdict(self)
