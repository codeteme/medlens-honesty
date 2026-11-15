"""SAE-feature attribution graphs via attribution patching (gradient x activation).

This is not circuit-tracer (which needs transcoders for every layer and transformers<=4.57).
SAEs are spliced into a few residual-stream layers, with each SAE's reconstruction error held
fixed, so the forward pass is unchanged. Gradients flow through the SAE features (not through
the error terms). Then:
- node attribution = activation x d(metric)/d(activation), a first-order estimate of how much
  removing that feature at that token would change the metric;
- edge attribution (upstream u -> downstream v) = a_u x d(a_v)/d(a_u), computed through the
  real model between the two layers.
The metric is logp(hinted letter) - logp(correct letter) after a prefilled "Final answer:".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from medlens.interp.features import PREFILL, letter_token_ids
from medlens.models.hf_local import HFLocalModel


@dataclass
class Node:
    layer: int
    pos: int
    feature: int
    activation: float
    attribution: float
    token: str


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[tuple[int, int, float]]  # (upstream node index, downstream node index, weight)
    metric: float
    tokens: list[str]
    meta: dict = field(default_factory=dict)


def _splice_hooks(saes: dict[int, Any], store: dict):
    """Replace resid_post at each layer with decode(a) + err, where a = encode(x) stays in the
    autograd graph and err = x - decode(a) is held constant. The forward value is unchanged;
    gradients reach every layer's features, including paths through later spliced layers
    (total effects). Gradient does not flow through the error terms, a standard approximation.
    `store[layer]` holds `a` for that layer."""

    def make(layer: int):
        sae = saes[layer]

        def hook(x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            a = sae.encode(xf)
            err = (xf - sae.decode(a)).detach()
            store[layer] = a
            return (sae.decode(a) + err).to(x.dtype)

        return hook

    return {(layer, "resid_post"): make(layer) for layer in saes}


def build_graph(
    model: HFLocalModel,
    saes: dict[int, Any],
    messages: list[dict],
    letters: str,
    target: str,
    gold: str,
    top_nodes_per_layer: int = 8,
    edge_top: int = 3,
) -> Graph:
    prompt = model.chat_prompt(messages)
    ids = model.encode(prompt + PREFILL, chat_formatted=True)
    tokens = [model.tokenizer.decode([t]) for t in ids[0].tolist()]
    lt = dict(zip(letters, letter_token_ids(model, letters), strict=True))
    layers = sorted(saes)
    store: dict = {}
    with torch.enable_grad(), model.hooks(replace=_splice_hooks(saes, store)):
        logits = model.model(input_ids=ids).logits
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        metric = lp[lt[target]] - lp[lt[gold]]
        leaves = [store[layer] for layer in layers]
        grads = torch.autograd.grad(metric, leaves, retain_graph=True)

        nodes: list[Node] = []
        for layer, leaf, g in zip(layers, leaves, grads, strict=True):
            attr = (leaf * g)[0]  # [T, F]
            attr[0] = 0  # ignore <bos>
            flat = attr.flatten()
            top = torch.topk(flat.abs(), top_nodes_per_layer).indices
            for t in top.tolist():
                pos, feat = divmod(t, attr.shape[1])
                nodes.append(
                    Node(layer, pos, feat, float(leaf[0, pos, feat]), float(flat[t]), tokens[pos])
                )

        edges: list[tuple[int, int, float]] = []
        for vi, v in enumerate(nodes):
            ups = [(ui, u) for ui, u in enumerate(nodes) if u.layer < v.layer and u.pos <= v.pos]
            if not ups:
                continue
            a_live = store[v.layer][0, v.pos, v.feature]
            if not a_live.requires_grad:
                continue
            up_leaves = [store[layer] for layer in layers if layer < v.layer]
            g_up = torch.autograd.grad(a_live, up_leaves, retain_graph=True, allow_unused=True)
            g_by_layer = dict(
                zip([layer for layer in layers if layer < v.layer], g_up, strict=True)
            )
            scored = []
            for ui, u in ups:
                g = g_by_layer[u.layer]
                if g is None:
                    continue
                w = float(store[u.layer][0, u.pos, u.feature] * g[0, u.pos, u.feature])
                scored.append((ui, vi, w))
            edges += sorted(scored, key=lambda e: -abs(e[2]))[:edge_top]
    return Graph(
        nodes, edges, float(metric), tokens, {"target": target, "gold": gold, "layers": layers}
    )
