"""Parquet read/write helpers and a small JSON metadata sidecar.

All tabular data is persisted as Parquet (per project conventions). A tiny
``_meta.json`` per processed table records the ``last_updated`` stamp shown in
the dashboard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import get_path


def _resolve(name: str, kind: str) -> Path:
    base = get_path(kind)
    return base / f"{name}.parquet"


def write_parquet(df: pd.DataFrame, name: str, kind: str = "processed") -> Path:
    """Write ``df`` to ``<kind>/<name>.parquet`` and stamp metadata."""
    path = _resolve(name, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    _write_meta(name, kind, {"rows": int(len(df)), "cols": list(map(str, df.columns))})
    return path


def read_parquet(name: str, kind: str = "processed") -> Optional[pd.DataFrame]:
    """Read a parquet table, or return ``None`` if it does not exist."""
    path = _resolve(name, kind)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def exists(name: str, kind: str = "processed") -> bool:
    return _resolve(name, kind).exists()


def _meta_path(name: str, kind: str) -> Path:
    return get_path(kind) / f"{name}_meta.json"


def _write_meta(name: str, kind: str, extra: Dict[str, Any]) -> None:
    meta = {"last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    _meta_path(name, kind).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(name: str, kind: str = "processed") -> Dict[str, Any]:
    p = _meta_path(name, kind)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def last_updated(name: str, kind: str = "processed") -> Optional[str]:
    return read_meta(name, kind).get("last_updated")
