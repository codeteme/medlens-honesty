from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from medlens.runs import REPO_ROOT

PRICING_PATH = REPO_ROOT / "configs" / "pricing.yaml"


@lru_cache(maxsize=1)
def _table() -> dict[str, Any]:
    with open(PRICING_PATH) as f:
        return yaml.safe_load(f)


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one call, or None if the model has no verified price in configs/pricing.yaml."""
    price = _table().get(provider, {}).get(model)
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
