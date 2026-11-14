"""Run directories: every script run saves its config, environment info, and outputs together."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

_TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "sae-lens",
    "datasets",
    "anthropic",
    "numpy",
    "pandas",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_info() -> dict[str, Any]:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def start_run(name: str, cfg: dict[str, Any], base_dir: Path = OUTPUTS_DIR) -> Path:
    """Create `outputs/<name>/<timestamp>[-debug]/`, save config + metadata, set up logging."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / name / (stamp + ("-debug" if cfg.get("debug") else ""))
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {"name": name, "started": stamp, "git": git_info(), "env": environment_info()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    setup_logging(run_dir / "run.log")
    set_seed(int(cfg.get("seed", 0)))
    logging.getLogger(__name__).info("Run directory: %s", run_dir)
    return run_dir


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
