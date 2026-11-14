"""Local Hugging Face model with activation capture and activation replacement.

Hook sites match Gemma Scope 2's `hf_hook_point_in/out` (read from the SAE config.json files):
- "resid_post": output of decoder layer L                     (`model.layers.L.output`)
- "mlp_in":     output of layer L's pre_feedforward_layernorm  (transcoder input)
- "mlp_out":    output of layer L's post_feedforward_layernorm (transcoder output)

MedGemma 4B is an image+text model (Gemma3ForConditionalGeneration); we use it with
text only. `get_decoder()` returns the language model for both model classes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from medlens.models.base import Generation, Message, Usage

logger = logging.getLogger(__name__)

SITES = ("resid_post", "mlp_in", "mlp_out")
HookFn = Callable[[torch.Tensor], torch.Tensor]


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def site_module(layer: nn.Module, site: str) -> nn.Module:
    if site == "resid_post":
        return layer
    if site == "mlp_in":
        return layer.pre_feedforward_layernorm
    if site == "mlp_out":
        return layer.post_feedforward_layernorm
    raise ValueError(f"Unknown site {site!r}; expected one of {SITES}")


class HFLocalModel:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        model: nn.Module | None = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.name = f"hf/{model_id}"
        self.device = pick_device(device)
        if model is None:
            config = AutoConfig.from_pretrained(model_id)
            arch = (config.architectures or [""])[0]
            auto_cls = (
                AutoModelForImageTextToText
                if "ConditionalGeneration" in arch
                else AutoModelForCausalLM
            )
            logger.info("Loading %s (%s) on %s as %s", model_id, arch, self.device, dtype)
            model = auto_cls.from_pretrained(model_id, dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    # ---- structure -------------------------------------------------------
    @property
    def layers(self) -> nn.ModuleList:
        return self.model.get_decoder().layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    # ---- text ------------------------------------------------------------
    def chat_prompt(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def encode(self, text: str, chat_formatted: bool) -> torch.Tensor:
        # Chat templates already include <bos>; raw text needs the tokenizer to add it.
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=not chat_formatted)
        return ids["input_ids"].to(self.device)

    @torch.no_grad()
    def generate(self, messages: list[Message], max_new_tokens: int = 1024) -> Generation:
        """Greedy decoding, so local generations are deterministic."""
        return self.generate_batch([messages], max_new_tokens)[0]

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        ids = set(eos if isinstance(eos, list) else [eos])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    @torch.no_grad()
    def generate_batch(
        self, batch: list[list[Message]], max_new_tokens: int = 1024
    ) -> list[Generation]:
        """Greedy decoding of several chats at once (left-padded).

        Results are deterministic for a fixed batch composition. Padding can change floating
        point details slightly, so the same prompt may decode differently in a different batch;
        experiments therefore fix the batch size in their config.
        """
        texts = [self.chat_prompt(m) for m in batch]
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = old_side
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        eos, pad = self._eos_ids(), self.tokenizer.pad_token_id
        gens = []
        for i in range(len(batch)):
            new = out[i, enc["input_ids"].shape[1] :].tolist()
            end = next((j for j, t in enumerate(new) if t in eos), None)
            tokens = [t for t in (new if end is None else new[:end]) if t != pad]
            gens.append(
                Generation(
                    text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                    model=self.name,
                    usage=Usage(
                        input_tokens=int(enc["attention_mask"][i].sum()),
                        output_tokens=len(tokens),
                    ),
                    stop_reason="end_turn" if end is not None else "max_tokens",
                )
            )
        return gens

    # ---- activations -----------------------------------------------------
    @contextmanager
    def hooks(
        self,
        capture: Sequence[tuple[int, str]] = (),
        replace: dict[tuple[int, str], HookFn] | None = None,
    ) -> Iterator[dict[tuple[int, str], torch.Tensor]]:
        """Capture and/or replace activations at (layer, site) points during forward passes.

        Captured tensors are [batch, seq, d] on CPU in float32. A replacement function gets
        the original activation and returns the tensor to use instead.
        """
        store: dict[tuple[int, str], torch.Tensor] = {}
        handles = []
        replace = replace or {}

        def make_hook(key: tuple[int, str]):
            def hook(_mod: nn.Module, _inp: Any, out: Any) -> Any:
                act = out[0] if isinstance(out, tuple) else out
                if key in replace:
                    new = replace[key](act).to(act.dtype)
                    out = (new, *out[1:]) if isinstance(out, tuple) else new
                    act = new
                if key in capture:
                    store[key] = act.detach().float().cpu()
                return out

            return hook

        for key in set(capture) | set(replace):
            layer, site = key
            handles.append(
                site_module(self.layers[layer], site).register_forward_hook(make_hook(key))
            )
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def forward_logits(
        self,
        input_ids: torch.Tensor,
        capture: Sequence[tuple[int, str]] = (),
        replace: dict[tuple[int, str], HookFn] | None = None,
    ) -> tuple[torch.Tensor, dict[tuple[int, str], torch.Tensor]]:
        with self.hooks(capture=capture, replace=replace) as store:
            logits = self.model(input_ids=input_ids).logits
        return logits.float(), store
