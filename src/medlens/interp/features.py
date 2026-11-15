"""Feature capture for Phase 2: SAE features at chosen positions, plus a cheap answer readout.

Readout: the assistant turn is prefilled with "Final answer:" and we read the probability of
each option letter (" A", " B", ... are single tokens) at that point. One forward pass gives
how strongly the model leans toward each option before writing any reasoning.

Positions captured for every prompt:
- "gen_start": the last prompt token, where step-by-step generation would begin
- "final":     the ":" of the prefilled "Final answer:", where the readout is taken
- "hint":      mean over the tokens of the inserted cue sentence (absent for no-hint prompts)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from medlens.models.hf_local import HFLocalModel

PREFILL = "Final answer:"
POSITIONS = ("gen_start", "final", "hint")


@dataclass
class Captured:
    letter_logprobs: dict[str, float]  # over the option letters, renormalised
    resid: dict[int, dict[str, torch.Tensor]]  # layer -> position -> [d_model] float32
    n_tokens: int
    hint_tokens: int


def letter_token_ids(model: HFLocalModel, letters: str) -> list[int]:
    ids = []
    for letter in letters:
        toks = model.tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"' {letter}' is not a single token: {toks}")
        ids.append(toks[0])
    return ids


def char_span_to_tokens(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Token indices whose character span overlaps [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]


@torch.no_grad()
def capture(
    model: HFLocalModel,
    messages: list[dict],
    letters: str,
    layers: list[int],
    cue: str | None,
    replace: dict | None = None,
) -> Captured:
    prompt = model.chat_prompt(messages)
    text = prompt + PREFILL
    enc = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = torch.tensor([enc["input_ids"]], device=model.device)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    n_prompt = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    hint_idx: list[int] = []
    if cue:
        start = prompt.find(cue)
        if start < 0:
            raise ValueError("cue text not found in prompt")
        hint_idx = char_span_to_tokens(offsets, start, start + len(cue))

    logits, acts = model.forward_logits(
        ids, capture=[(layer, "resid_post") for layer in layers], replace=replace
    )
    lt = letter_token_ids(model, letters)
    lp = torch.log_softmax(logits[0, -1, lt], dim=-1)
    resid = {}
    for layer in layers:
        a = acts[(layer, "resid_post")][0]
        resid[layer] = {"gen_start": a[n_prompt - 1], "final": a[-1]}
        if hint_idx:
            resid[layer]["hint"] = a[hint_idx].mean(dim=0)
    return Captured(
        letter_logprobs={k: float(v) for k, v in zip(letters, lp, strict=True)},
        resid=resid,
        n_tokens=ids.shape[1],
        hint_tokens=len(hint_idx),
    )


def prompt_positions(
    model: HFLocalModel, messages: list[dict], cue: str | None, n_final: int = 5
) -> dict[str, list[int]]:
    """Token positions of the cue sentence and of the last `n_final` tokens (prompt + PREFILL)."""
    prompt = model.chat_prompt(messages)
    enc = model.tokenizer(prompt + PREFILL, return_offsets_mapping=True, add_special_tokens=False)
    n = len(enc["input_ids"])
    hint: list[int] = []
    if cue:
        start = prompt.find(cue)
        hint = char_span_to_tokens(
            [tuple(o) for o in enc["offset_mapping"]], start, start + len(cue)
        )
    return {"hint": hint, "final": list(range(n - n_final, n)), "all": list(range(1, n))}
