"""Feature engineering — Groll-style covariates for a per-team goals model.

Builds a *team-match* long table (each match -> two rows, one per team's
perspective) with leak-free dynamic Elo, rolling form, opponent form, squad
market value/age, FIFA rank, host & confederation flags, then assembles the
numeric feature matrix ``X`` (numpy) with an explicit ``feature_names`` list,
plus Dixon-Coles exponential time-decay sample weights.

References: Groll et al. (2019, arXiv:1806.03208); Dixon & Coles (1997).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..tournament import confederation_map, host_nations
from .elo import compute_elo_timeline

# Final model feature order (X columns).
FEATURE_NAMES: List[str] = [
    "elo_team", "elo_opp", "elo_diff",
    "rank_team", "rank_opp", "rank_diff",
    "mv_team", "mv_opp", "mv_diff",
    "age_team", "age_opp",
    "loggdp_team", "logpop_team",
    "host_team", "host_opp", "same_confed", "is_home",
    "form_gf", "form_ga", "form_ppg",
    "opp_form_gf", "opp_form_ga", "opp_form_ppg",
    "days_rest",
]


@dataclass
class FeatureContext:
    """Everything needed to build features for a *future* (predicted) match."""
    attrs: pd.DataFrame                       # per-team static attributes (indexed by team)
    latest_form: pd.DataFrame                 # per-team most-recent rolling form
    current_elo: Dict[str, float]             # per-team current Elo
    medians: pd.Series                        # training-set medians for imputation
    confed: Dict[str, str] = field(default_factory=confederation_map)
    hosts: List[str] = field(default_factory=host_nations)


# ---------------------------------------------------------------------------
# Dixon-Coles time decay
# ---------------------------------------------------------------------------
def dixon_coles_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Exponential half-life weighting; most recent match has weight ~1."""
    ref = dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    xi = np.log(2.0) / float(half_life_days)
    return np.exp(-xi * age_days)


# ---------------------------------------------------------------------------
# Long team-match table with form
# ---------------------------------------------------------------------------
def prepare_long(results: pd.DataFrame, rolling_window: int = 10,
                 init_elo: Dict[str, float] | None = None):
    """Return (long_df, current_elo, latest_form).

    ``long_df`` has, per played match, two rows with pre-match Elo, leak-free
    rolling form (shifted) and opponent form, ready for feature assembly.
    ``init_elo`` seeds the Elo engine so the training-feature scale matches the
    snapshot used at prediction time.
    """
    res, current_elo = compute_elo_timeline(results, init=init_elo)
    res = res.reset_index(drop=True)
    res["match_id"] = res.index

    played = res.dropna(subset=["home_score", "away_score"]).copy()

    def _side(is_home: bool) -> pd.DataFrame:
        if is_home:
            d = played.rename(columns={
                "home_team": "team", "away_team": "opp",
                "home_score": "gf", "away_score": "ga",
                "elo_home_pre": "elo_team", "elo_away_pre": "elo_opp"})
            d["is_home"] = (~d["neutral"]).astype(int)
        else:
            d = played.rename(columns={
                "away_team": "team", "home_team": "opp",
                "away_score": "gf", "home_score": "ga",
                "elo_away_pre": "elo_team", "elo_home_pre": "elo_opp"})
            d["is_home"] = 0
        return d[["match_id", "date", "team", "opp", "gf", "ga", "is_home",
                  "neutral", "elo_team", "elo_opp"]]

    long = pd.concat([_side(True), _side(False)], ignore_index=True)
    long["points"] = np.where(long["gf"] > long["ga"], 3,
                              np.where(long["gf"] == long["ga"], 1, 0))
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    g = long.groupby("team", group_keys=False)
    # leak-free (shifted) rolling form for training
    long["form_gf"] = g["gf"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean().shift(1))
    long["form_ga"] = g["ga"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean().shift(1))
    long["form_ppg"] = g["points"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean().shift(1))
    long["days_rest"] = g["date"].apply(lambda s: s.diff().dt.days)

    # opponent form via match_id swap (exactly two rows per match)
    fk = long[["match_id", "team", "form_gf", "form_ga", "form_ppg"]]
    merged = long.merge(fk, on="match_id", suffixes=("", "_o"))
    merged = merged[merged["team"] != merged["team_o"]]
    merged = merged.rename(columns={"form_gf_o": "opp_form_gf",
                                    "form_ga_o": "opp_form_ga",
                                    "form_ppg_o": "opp_form_ppg"})
    long = merged.drop(columns=["team_o"]).sort_values(["team", "date"]).reset_index(drop=True)

    # latest (non-shifted) form per team for prediction
    cur = long.copy().sort_values(["team", "date"])
    cg = cur.groupby("team", group_keys=False)
    cur["c_gf"] = cg["gf"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    cur["c_ga"] = cg["ga"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    cur["c_ppg"] = cg["points"].apply(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    latest_form = (cur.groupby("team")[["c_gf", "c_ga", "c_ppg"]].last()
                   .rename(columns={"c_gf": "form_gf", "c_ga": "form_ga", "c_ppg": "form_ppg"}))
    return long, current_elo, latest_form


# ---------------------------------------------------------------------------
# Feature assembly (shared by train + predict)
# ---------------------------------------------------------------------------
def _attr(attrs: pd.DataFrame, teams: pd.Series, col: str) -> pd.Series:
    s = teams.map(attrs[col]) if col in attrs.columns else pd.Series(np.nan, index=teams.index)
    return pd.to_numeric(s, errors="coerce")


def assemble_features(frame: pd.DataFrame, attrs: pd.DataFrame,
                      confed: Dict[str, str], hosts: List[str]) -> pd.DataFrame:
    """Map a frame with team/opp columns to the FEATURE_NAMES matrix (pre-impute)."""
    f = pd.DataFrame(index=frame.index)
    f["elo_team"] = frame["elo_team"].astype(float)
    f["elo_opp"] = frame["elo_opp"].astype(float)
    f["elo_diff"] = f["elo_team"] - f["elo_opp"]

    f["rank_team"] = _attr(attrs, frame["team"], "fifa_rank")
    f["rank_opp"] = _attr(attrs, frame["opp"], "fifa_rank")
    f["rank_diff"] = f["rank_opp"] - f["rank_team"]      # positive => team better ranked

    f["mv_team"] = _attr(attrs, frame["team"], "market_value_m")
    f["mv_opp"] = _attr(attrs, frame["opp"], "market_value_m")
    f["mv_diff"] = f["mv_team"] - f["mv_opp"]

    f["age_team"] = _attr(attrs, frame["team"], "squad_age")
    f["age_opp"] = _attr(attrs, frame["opp"], "squad_age")

    # derive log GDP/pop from raw columns when present (else NaN -> imputed)
    f["loggdp_team"] = np.log1p(_attr(attrs, frame["team"], "gdp_per_capita"))
    f["logpop_team"] = np.log1p(_attr(attrs, frame["team"], "population"))

    hostset = set(hosts)
    f["host_team"] = frame["team"].isin(hostset).astype(int)
    f["host_opp"] = frame["opp"].isin(hostset).astype(int)
    f["same_confed"] = (frame["team"].map(confed) == frame["opp"].map(confed)).astype(int)
    f["is_home"] = frame["is_home"].astype(int)

    for c in ["form_gf", "form_ga", "form_ppg", "opp_form_gf", "opp_form_ga", "opp_form_ppg"]:
        f[c] = frame[c].astype(float)
    f["days_rest"] = frame["days_rest"].astype(float)
    return f[FEATURE_NAMES]


def build_training_matrix(results: pd.DataFrame, attrs: pd.DataFrame,
                          rolling_window: int = 10, half_life_days: float = 1095.0):
    """Return (X, y, weights, feature_names, context, dates).

    X is a numpy array (rows = team-match perspectives); y = goals scored;
    weights = Dixon-Coles time decay; context carries imputation medians and the
    state needed to featurise future matches; dates orders rows in time (for
    TimeSeriesSplit). Rows are returned sorted by date.
    """
    init_elo = (attrs["elo"].to_dict() if "elo" in attrs.columns else None)
    long, current_elo, latest_form = prepare_long(results, rolling_window, init_elo=init_elo)
    long = long.sort_values("date").reset_index(drop=True)   # global time order
    confed = confederation_map()
    hosts = host_nations()

    Xdf = assemble_features(long, attrs, confed, hosts)
    medians = Xdf.median(numeric_only=True).fillna(0.0)  # all-NaN cols -> 0
    Xdf = Xdf.fillna(medians)

    y = long["gf"].to_numpy(dtype=float)
    w = dixon_coles_weights(long["date"], half_life_days)
    dates = long["date"].reset_index(drop=True)

    # At PREDICTION time, prefer the curated/live current Elo (snapshot or
    # eloratings.net) over the endpoint of the historical replay — the latter
    # is only used as leak-free per-match training features.
    if "elo" in attrs.columns:
        for t in attrs.index:
            v = attrs.loc[t, "elo"]
            if pd.notna(v):
                current_elo[t] = float(v)

    ctx = FeatureContext(attrs=attrs, latest_form=latest_form,
                         current_elo=current_elo, medians=medians,
                         confed=confed, hosts=hosts)
    return Xdf.to_numpy(dtype=float), y, w, FEATURE_NAMES, ctx, dates


def match_feature_row(team: str, opp: str, is_home: int, ctx: FeatureContext) -> np.ndarray:
    """Featurise a single (future) matchup using current Elo + latest form."""
    def form(t: str, col: str) -> float:
        if t in ctx.latest_form.index:
            return float(ctx.latest_form.loc[t, col])
        return float(ctx.medians.get(col, np.nan))

    frame = pd.DataFrame([{
        "team": team, "opp": opp, "is_home": is_home,
        "elo_team": ctx.current_elo.get(team, 1500.0),
        "elo_opp": ctx.current_elo.get(opp, 1500.0),
        "form_gf": form(team, "form_gf"), "form_ga": form(team, "form_ga"),
        "form_ppg": form(team, "form_ppg"),
        "opp_form_gf": form(opp, "form_gf"), "opp_form_ga": form(opp, "form_ga"),
        "opp_form_ppg": form(opp, "form_ppg"),
        "days_rest": float(ctx.medians.get("days_rest", 30.0)),
    }])
    X = assemble_features(frame, ctx.attrs, ctx.confed, ctx.hosts).fillna(ctx.medians)
    return X.to_numpy(dtype=float)[0]
