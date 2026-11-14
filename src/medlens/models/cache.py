"""On-disk cache for model responses, keyed by a hash of the full request.

API models can't be made deterministic (Claude rejects `temperature` as of 2026-10-21),
so the cache is what makes reruns reproducible and free.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from medlens.runs import REPO_ROOT

DEFAULT_CACHE_PATH = REPO_ROOT / "cache" / "generations.sqlite"


def request_key(request: dict[str, Any]) -> str:
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, request TEXT, response TEXT)"
        )

    def get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT response FROM responses WHERE key = ?", (request_key(request),)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
            (request_key(request), json.dumps(request, default=str), json.dumps(response)),
        )
        self._conn.commit()
