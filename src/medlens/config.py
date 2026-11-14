"""YAML config loading with an optional `debug:` override block.

Every experiment config is a YAML file. If it contains a top-level `debug:` mapping,
those keys are deep-merged over the base config when the script runs with `--debug`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `base` with `override` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, debug: bool = False) -> dict[str, Any]:
    """Load a YAML config; apply its `debug:` block when `debug` is True."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    debug_block = raw.pop("debug", None) or {}
    cfg = deep_merge(raw, debug_block) if debug else raw
    cfg["debug"] = debug
    cfg.setdefault("seed", 0)
    return cfg
