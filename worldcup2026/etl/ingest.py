"""ETL orchestration — the work behind the dashboard's *Refresh Data* button.

Pulls historical + live results and per-team attributes, harmonises names, and
writes Parquet tables (``results``, ``team_attrs``) under ``data/processed``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from .. import persistence as pstore
from ..tournament import all_teams, teams_table
from . import sources

log = logging.getLogger("worldcup2026.etl")


def build_team_attrs() -> pd.DataFrame:
    """Merge live/snapshot attributes into one per-team table indexed by team."""
    base = teams_table().reset_index().rename(columns={"index": "team"})
    if "team" not in base.columns:  # set_index used name 'team' already
        base = teams_table().reset_index()
    base = base.rename(columns={base.columns[0]: "team"}) if base.columns[0] != "team" else base

    elo = sources.fetch_elo()
    rank = sources.fetch_fifa_ranking()
    mv = sources.fetch_market_values()
    wb = sources.fetch_worldbank()

    out = base.drop(columns=[c for c in ["elo", "fifa_rank", "market_value_m", "squad_age"]
                             if c in base.columns])
    out = (out
           .merge(elo, on="team", how="left")
           .merge(rank, on="team", how="left")
           .merge(mv, on="team", how="left")
           .merge(wb, on="team", how="left"))

    # fill any gaps from the offline snapshot
    snap = teams_table()
    for col in ["elo", "fifa_rank", "market_value_m", "squad_age"]:
        if col in out.columns:
            out[col] = out[col].fillna(out["team"].map(snap[col]))
    return out.set_index("team")


def run_etl(refresh: bool = False) -> Dict[str, Any]:
    """Run the full ETL. ``refresh`` appends new live results to existing data."""
    log.info("Starting ETL (refresh=%s)...", refresh)

    # fetch_results() is a FULL historical pull every time, so we always replace
    # the historical table and only union in the (incremental) live WC-2026 rows.
    # Live results are derived FROM the historical pull (World Cup games on/after
    # the 2026 opener), so the union below is a no-op after de-dup unless the
    # manual override CSV adds fixtures Kaggle has not yet ingested.
    results = sources.fetch_results()
    live = sources.fetch_live_results(results)
    results = pd.concat([results, live], ignore_index=True)

    # de-duplicate on (date, home, away)
    results = (results
               .dropna(subset=["date", "home_team", "away_team"])
               .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
               .sort_values("date")
               .reset_index(drop=True))

    team_attrs = build_team_attrs()

    pstore.write_parquet(results, "results", "processed")
    pstore.write_parquet(team_attrs, "team_attrs", "processed")
    pstore.write_parquet(live, "wc2026_live", "processed")

    summary = {
        "n_matches": int(len(results)),
        "date_min": str(results["date"].min().date()) if len(results) else None,
        "date_max": str(results["date"].max().date()) if len(results) else None,
        "n_live_wc2026": int(live["home_score"].notna().sum()) if len(live) else 0,
        "n_teams": len(team_attrs),
        "last_updated": pstore.last_updated("results"),
        "synthetic": bool((results["tournament"] == "Synthetic").any()),
    }
    log.info("ETL complete: %s", summary)
    return summary
