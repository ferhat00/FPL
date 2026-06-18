"""World Football Elo engine.

Computes each team's *pre-match* Elo as the result history is replayed
chronologically (so training features are leak-free), plus the final current
rating per team. Uses the standard eloratings.net update with a goal-difference
multiplier and home-field advantage.

Reference: World Football Elo Ratings methodology; used as the headline team-ability
covariate in Groll et al. (2019).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0


def _gd_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference weighting."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0  # 3 -> 1.75, 4 -> 1.875, ...


def compute_elo_timeline(
    results: pd.DataFrame, k: float = 30.0, hfa: float = 65.0,
    init: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return (results + elo_home_pre/elo_away_pre, current_elo_per_team).

    Unplayed rows (NaN score) contribute their pre-match Elo but do not update
    ratings — handy when live fixtures are present. ``init`` seeds starting
    ratings (e.g. the current snapshot) so the feature scale matches prediction;
    early matches are heavily down-weighted by Dixon-Coles decay regardless.
    """
    elo: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
    if init:
        elo.update({k_: float(v) for k_, v in init.items() if v == v})  # skip NaN
    pre_home = np.empty(len(results))
    pre_away = np.empty(len(results))

    home = results["home_team"].to_numpy()
    away = results["away_team"].to_numpy()
    hs = results["home_score"].to_numpy()
    as_ = results["away_score"].to_numpy()
    neutral = results["neutral"].to_numpy()

    for i in range(len(results)):
        h, a = home[i], away[i]
        eh, ea = elo[h], elo[a]
        pre_home[i], pre_away[i] = eh, ea
        if np.isnan(hs[i]) or np.isnan(as_[i]):
            continue
        ha = 0.0 if neutral[i] else hfa
        exp_h = 1.0 / (1.0 + 10 ** (-(eh + ha - ea) / 400.0))
        res_h = 1.0 if hs[i] > as_[i] else 0.5 if hs[i] == as_[i] else 0.0
        delta = k * _gd_multiplier(hs[i] - as_[i]) * (res_h - exp_h)
        elo[h] = eh + delta
        elo[a] = ea - delta

    out = results.copy()
    out["elo_home_pre"] = pre_home
    out["elo_away_pre"] = pre_away
    return out, dict(elo)
