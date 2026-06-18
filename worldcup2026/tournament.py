"""Load and expose the 2026 World Cup definition (``data/tournament_2026.yaml``).

Provides:
  * ``load_tournament()``  – raw dict
  * ``teams_table()``      – DataFrame of per-team static attributes
  * ``group_fixtures()``   – the 72 round-robin group matches (deterministic)
  * group / bracket accessors used by the simulator
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from .config import TOURNAMENT_YAML


@lru_cache(maxsize=1)
def load_tournament() -> Dict[str, Any]:
    return yaml.safe_load(TOURNAMENT_YAML.read_text(encoding="utf-8"))


def groups() -> Dict[str, List[str]]:
    return load_tournament()["groups"]


def all_teams() -> List[str]:
    out: List[str] = []
    for members in groups().values():
        out.extend(members)
    return out


def host_nations() -> List[str]:
    return load_tournament()["meta"]["hosts"]


def teams_table() -> pd.DataFrame:
    """Per-team static attributes as a DataFrame indexed by team name.

    Adds the group letter for each team. Values here are the *offline snapshot*;
    ETL merges in live Elo / market value when available.
    """
    t = load_tournament()
    rows = []
    team_to_group = {}
    for g, members in t["groups"].items():
        for team in members:
            team_to_group[team] = g
    for team, attrs in t["teams"].items():
        row = {"team": team, "group": team_to_group.get(team)}
        row.update(attrs)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("team")
    # guarantee expected columns exist
    for col, default in {
        "confederation": "UNK", "host": False, "elo": 1500.0,
        "fifa_rank": 100, "market_value_m": 50.0, "squad_age": 27.0,
    }.items():
        if col not in df.columns:
            df[col] = default
    return df


def group_fixtures() -> pd.DataFrame:
    """Deterministic round-robin: each group's 4 teams -> 6 matches (72 total).

    Columns: group, home, away. (home/away here is a label only — group games
    are at neutral/assigned venues; the model treats venue via the host flag.)
    """
    rows = []
    for g, members in groups().items():
        for a, b in combinations(members, 2):
            rows.append({"group": g, "home": a, "away": b})
    return pd.DataFrame(rows)


def bracket() -> Dict[str, List[Dict[str, str]]]:
    return load_tournament()["bracket"]


def meta() -> Dict[str, Any]:
    return load_tournament()["meta"]


def playoff_slots() -> List[str]:
    return load_tournament().get("playoff_slots", [])


def confederation_map() -> Dict[str, str]:
    return {team: attrs.get("confederation", "UNK")
            for team, attrs in load_tournament()["teams"].items()}
