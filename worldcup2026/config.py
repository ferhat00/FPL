"""Configuration loading and path resolution.

`load_config()` reads ``config.yaml`` from the repo root, falling back to
``config.example.yaml`` so the pipeline is runnable out of the box (in offline /
synthetic mode). All paths are resolved to absolute and the directories created.
"""
from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# repo root = two levels up from this file (worldcup2026/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
TOURNAMENT_YAML = PACKAGE_ROOT / "data" / "tournament_2026.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load merged configuration (example defaults <- user config.yaml <- env)."""
    example = REPO_ROOT / "config.example.yaml"
    user = REPO_ROOT / "config.yaml"

    cfg: Dict[str, Any] = {}
    if example.exists():
        cfg = yaml.safe_load(example.read_text(encoding="utf-8")) or {}
    if user.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(user.read_text(encoding="utf-8")) or {})

    # environment overrides for secrets (handy in CI / Streamlit Cloud)
    kg = cfg.setdefault("kaggle", {})
    kg["username"] = os.environ.get("KAGGLE_USERNAME", kg.get("username", ""))
    kg["key"] = os.environ.get("KAGGLE_KEY", kg.get("key", ""))

    # resolve + create paths
    paths = cfg.setdefault("paths", {})
    for name, default in {
        "raw": "worldcup2026/data/raw",
        "processed": "worldcup2026/data/processed",
        "models": "worldcup2026/data/models",
        "cache": "worldcup2026/data/cache",
        "output": "output",
    }.items():
        p = Path(paths.get(name, default))
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = str(p)

    return cfg


def get_path(name: str) -> Path:
    """Return a resolved path from config (raw/processed/models/cache/output)."""
    return Path(load_config()["paths"][name])


def kaggle_available() -> bool:
    """True if Kaggle credentials are present (config or ~/.kaggle/kaggle.json)."""
    cfg = load_config()
    kg = cfg.get("kaggle", {})
    if kg.get("username") and kg.get("key"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()
