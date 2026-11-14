"""Open models hosted on Tinker (Thinking Machines), sampled with the official `tinker` SDK.

Verified against tinker 0.39.0 on 2026-11-14: `ServiceClient().create_sampling_client(
base_model=...)`, `get_tokenizer()`, `sample(prompt=ModelInput.from_ints(ids), num_samples,
sampling_params=SamplingParams(...))` returning an APIFuture with `.result()`. Each `sample()`
call is submitted immediately, so submitting many before waiting runs them concurrently.
Tinker does not return a usage object; input tokens are counted from the prompt and output
tokens from the returned sequence.
"""

from __future__ import annotations

from typing import Any

from medlens.models.base import Generation, Message, Usage
from medlens.models.cache import ResponseCache
from medlens.models.pricing import cost_usd


class TinkerModel:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.8-27B",
        temperature: float = 0.0,
        enable_thinking: bool = False,
        cache: ResponseCache | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = f"tinker/{model_id}"
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.cache = cache
        self._client: Any = None
        self._tokenizer: Any = None

    def _ensure(self) -> None:
        if self._client is None:
            import tinker

            self._client = tinker.ServiceClient().create_sampling_client(base_model=self.model_id)
            self._tokenizer = self._client.get_tokenizer()

    def _prompt_text(self, messages: list[Message]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _request(self, messages: list[Message], max_new_tokens: int) -> dict[str, Any]:
        return {
            "provider": "tinker",
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
        }

    def generate(self, messages: list[Message], max_new_tokens: int = 1024) -> Generation:
        return self.generate_many([messages], max_new_tokens)[0]

    def generate_many(
        self, batch: list[list[Message]], max_new_tokens: int = 1024
    ) -> list[Generation]:
        """Submit every uncached request at once, then collect results in order."""
        from tinker import types

        results: list[Generation | None] = [None] * len(batch)
        pending = []
        for i, messages in enumerate(batch):
            request = self._request(messages, max_new_tokens)
            if self.cache is not None and (hit := self.cache.get(request)) is not None:
                gen = Generation.from_dict(hit)
                gen.cached = True
                results[i] = gen
                continue
            self._ensure()
            ids = self._tokenizer.encode(self._prompt_text(messages), add_special_tokens=False)
            future = self._client.sample(
                prompt=types.ModelInput.from_ints(ids),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=max_new_tokens, temperature=self.temperature
                ),
            )
            pending.append((i, request, len(ids), future))

        for i, request, n_in, future in pending:
            seq = future.result().sequences[0]
            n_out = len(seq.tokens)
            usage = Usage(input_tokens=n_in, output_tokens=n_out)
            usage.cost_usd = cost_usd("tinker", self.model_id, n_in, n_out)
            gen = Generation(
                text=self._tokenizer.decode(seq.tokens, skip_special_tokens=True),
                model=self.name,
                usage=usage,
                stop_reason=str(seq.stop_reason),
            )
            if self.cache is not None:
                self.cache.put(request, gen.to_dict())
            results[i] = gen
        return [g for g in results if g is not None]
