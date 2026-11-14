"""Claude via the official Anthropic SDK.

Notes for this study:
- Claude never returns its raw internal reasoning, so the "CoT" we analyse is the
  reasoning the model writes in its visible answer.
- Server-side refusal fallbacks are OFF by default: a fallback silently swaps in another
  model, which would contaminate per-model results. Refusals are recorded as refusals.
- Sampling parameters (temperature etc.) are not sent; the API rejects them.
"""

from __future__ import annotations

from typing import Any

import anthropic

from medlens.models.base import Generation, Message, Usage
from medlens.models.cache import ResponseCache
from medlens.models.pricing import cost_usd


class AnthropicModel:
    def __init__(
        self,
        model_id: str = "claude-opus-5-5",
        effort: str = "medium",
        cache: ResponseCache | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = f"anthropic/{model_id}"
        self.effort = effort
        self.cache = cache
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _request(self, messages: list[Message], max_new_tokens: int) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        request: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_new_tokens,
            "messages": [m for m in messages if m["role"] != "system"],
            "output_config": {"effort": self.effort},
        }
        if system:
            request["system"] = system
        return request

    def generate(self, messages: list[Message], max_new_tokens: int = 16000) -> Generation:
        request = self._request(messages, max_new_tokens)
        if self.cache is not None and (hit := self.cache.get(request)) is not None:
            gen = Generation.from_dict(hit)
            gen.cached = True
            return gen

        response = self.client.messages.create(**request)
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        usage.cost_usd = cost_usd(
            "anthropic", self.model_id, usage.input_tokens, usage.output_tokens
        )
        meta: dict[str, Any] = {"request_id": response._request_id}
        if response.stop_reason == "refusal" and response.stop_details is not None:
            meta["refusal_category"] = response.stop_details.category
        gen = Generation(
            text=text, model=self.name, usage=usage, stop_reason=response.stop_reason, meta=meta
        )
        if self.cache is not None:
            self.cache.put(request, gen.to_dict())
        return gen
