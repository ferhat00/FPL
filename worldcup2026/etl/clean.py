"""Cleaning utilities: team-name normalization and match-frame canonicalization.

Different sources spell countries differently (USA/United States,
Czechia/Czech Republic, Turkiye/Turkey, ...). We map everything to the single
canonical name used in ``tournament_2026.yaml``.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable

import pandas as pd

from ..tournament import all_teams

# Canonical names are the 48 WC-2026 entrants (others pass through normalized).
# Map common source variants -> canonical.
NAME_MAP = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "iran": "Iran",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "curacao": "Curacao",
    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "china pr": "China",
    "republic of ireland": "Ireland",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_team(name: str) -> str:
    """Return the canonical team name for ``name`` (best-effort)."""
    if not isinstance(name, str):
        return name
    key = _strip_accents(name).strip().lower()
    if key in NAME_MAP:
        return NAME_MAP[key]
    # title-case fallback that still matches canonical WC names
    titled = _strip_accents(name).strip()
    canon = {t.lower(): t for t in all_teams()}
    return canon.get(titled.lower(), titled)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize a raw results frame to the standard schema.

    Output columns: date (datetime), home_team, away_team, home_score,
    away_score, tournament, neutral (bool).
    """
    df = df.copy()
    rename = {
        "home_team": "home_team", "away_team": "away_team",
        "home_score": "home_score", "away_score": "away_score",
        "date": "date", "tournament": "tournament", "neutral": "neutral",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    if "tournament" not in df.columns:
        df["tournament"] = "Unknown"
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "neutral"]]


def canonical_teams() -> Iterable[str]:
    return all_teams()
