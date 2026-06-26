"""Actual group-stage standings, third-place ranking and R32 qualifiers.

Pure, model-free tournament accounting computed from *played* results (the
``wc2026_live`` table). Standings are ranked by the same FIFA key the simulator
uses — points -> goal difference -> goals for -> a strength (Elo) tiebreak — so
the real table is consistent with ``model/bracket.py``. Counts accumulate only
from played matches, so every function is correct mid-tournament (unplayed games
simply contribute nothing yet).

This is the factual counterpart to ``model/simulate.py::expected_bracket`` (which
projects the *remaining* fixtures); the dashboard shows the two side by side.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .model.bracket import select_best_thirds
from .tournament import groups as groups_fn

STANDING_COLS = ["team", "Pld", "W", "D", "L", "GF", "GA", "GD", "Pts"]


def _team_group(groups: Dict[str, List[str]]) -> Dict[str, str]:
    return {t: g for g, members in groups.items() for t in members}


def results_by_group(live_df: Optional[pd.DataFrame],
                     groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Label every intra-group fixture with its group and a ``played`` flag.

    Columns: ``group, home, away, home_score, away_score, played, date``.
    Rows whose two teams are not in the same group (i.e. not group-stage games)
    are dropped. Works on the full ``wc2026_live`` table (played + scheduled).
    """
    cols = ["group", "home", "away", "home_score", "away_score", "played", "date"]
    if live_df is None or len(live_df) == 0:
        return pd.DataFrame(columns=cols)
    groups = groups or groups_fn()
    tg = _team_group(groups)
    rows = []
    for r in live_df.itertuples():
        g = tg.get(r.home_team)
        if g is None or g != tg.get(r.away_team):
            continue  # not an intra-group fixture
        played = pd.notna(r.home_score) and pd.notna(r.away_score)
        rows.append({
            "group": g, "home": r.home_team, "away": r.away_team,
            "home_score": r.home_score, "away_score": r.away_score,
            "played": bool(played), "date": getattr(r, "date", pd.NaT),
        })
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values(["group", "date"], kind="stable").reset_index(drop=True)
    return out


def group_standings(live_df: Optional[pd.DataFrame],
                    groups: Optional[Dict[str, List[str]]] = None,
                    strength: Optional[Dict[str, float]] = None) -> Dict[str, pd.DataFrame]:
    """Per-group standings table from played results.

    Returns ``{group_letter: DataFrame}`` with columns ``STANDING_COLS``, ranked
    best->worst by ``(Pts, GD, GF, strength)`` and indexed by position (1..4).
    """
    groups = groups or groups_fn()
    strength = strength or {}
    played = results_by_group(live_df, groups)
    played = played[played["played"]] if len(played) else played

    out: Dict[str, pd.DataFrame] = {}
    for g, members in groups.items():
        rec = {t: {"team": t, "Pld": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
               for t in members}
        sub = played[played["group"] == g] if len(played) else played
        for m in sub.itertuples():
            h, a = m.home, m.away
            hs, as_ = int(m.home_score), int(m.away_score)
            for t, gf, ga in ((h, hs, as_), (a, as_, hs)):
                rec[t]["Pld"] += 1
                rec[t]["GF"] += gf
                rec[t]["GA"] += ga
            if hs > as_:
                rec[h]["W"] += 1; rec[h]["Pts"] += 3; rec[a]["L"] += 1
            elif hs < as_:
                rec[a]["W"] += 1; rec[a]["Pts"] += 3; rec[h]["L"] += 1
            else:
                rec[h]["D"] += 1; rec[a]["D"] += 1
                rec[h]["Pts"] += 1; rec[a]["Pts"] += 1

        df = pd.DataFrame([rec[t] for t in members])
        df["GD"] = df["GF"] - df["GA"]
        df["_s"] = df["team"].map(lambda t: strength.get(t, 0.0))
        df = (df.sort_values(["Pts", "GD", "GF", "_s"], ascending=False, kind="stable")
                .drop(columns="_s")
                .reset_index(drop=True))
        df = df[STANDING_COLS]
        df.index = pd.RangeIndex(1, len(df) + 1, name="Pos")
        out[g] = df
    return out


def third_place_ranking(standings: Dict[str, pd.DataFrame],
                        strength: Optional[Dict[str, float]] = None,
                        n_qualify: int = 8) -> pd.DataFrame:
    """Rank the 12 third-placed teams; flag the best ``n_qualify`` as qualifiers.

    Uses the same ordering as ``model/bracket.py::select_best_thirds`` so the
    actual third-place table matches the simulator's selection rule.
    """
    strength = strength or {}
    third_rows: Dict[str, dict] = {}
    third_stats: Dict[str, tuple] = {}
    for g, df in standings.items():
        if len(df) < 3:
            continue
        row = df.loc[3]  # third position
        team = row["team"]
        third_rows[team] = {"group": g, **{c: row[c] for c in STANDING_COLS}}
        third_stats[team] = (int(row["Pts"]), int(row["GD"]), int(row["GF"]))

    ranked = select_best_thirds(third_stats, strength, n=len(third_stats))
    out = []
    for i, team in enumerate(ranked, start=1):
        rec = third_rows[team]
        out.append({"rank": i, **rec, "qualified": i <= n_qualify})
    cols = ["rank", "group"] + STANDING_COLS + ["qualified"]
    return pd.DataFrame(out, columns=cols)


def r32_qualifiers(standings: Dict[str, pd.DataFrame],
                   third_ranking: pd.DataFrame) -> pd.DataFrame:
    """The 32 teams through to the Round of 32: 12 winners + 12 runners-up + 8 thirds.

    Columns: ``team, group, route, seed`` (seed 1=winner, 2=runner-up, 3=best third).
    """
    rows = []
    for g, df in standings.items():
        if len(df) >= 1:
            rows.append({"team": df.loc[1, "team"], "group": g,
                         "route": "Group winner", "seed": 1})
        if len(df) >= 2:
            rows.append({"team": df.loc[2, "team"], "group": g,
                         "route": "Runner-up", "seed": 2})
    if len(third_ranking):
        for r in third_ranking[third_ranking["qualified"]].itertuples():
            rows.append({"team": r.team, "group": r.group,
                         "route": "3rd place", "seed": 3})
    return pd.DataFrame(rows, columns=["team", "group", "route", "seed"])
