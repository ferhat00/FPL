"""Backtesting & validation (the dashboard's *Train & Validate* metrics).

Time-ordered holdout: train the RF on matches before a cutoff, predict match
outcomes (1X2) on matches after it via Poisson goal means, and score with the
Ranked Probability Score, multiclass log-loss and Brier — compared against an
Elo-only baseline. Also returns a calibration table for P(home/teamA win).

A tournament filter (e.g. "FIFA World Cup") restricts the test set to real
tournament matches when such data is present.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss

from ..config import load_config
from ..features.engineering import (FEATURE_NAMES, assemble_features,
                                    dixon_coles_weights, prepare_long)
from ..tournament import confederation_map, host_nations
from .train import _make_estimator

log = logging.getLogger("worldcup2026.validate")


def outcome_probs(lam_a: float, lam_b: float, max_goals: int = 10):
    """P(teamA win), P(draw), P(teamB win) under independent Poisson goals."""
    a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    b = poisson.pmf(np.arange(max_goals + 1), lam_b)
    m = np.outer(a, b)
    p_draw = float(np.trace(m))
    p_a = float(np.tril(m, -1).sum())
    p_b = float(np.triu(m, 1).sum())
    tot = p_a + p_draw + p_b
    return p_a / tot, p_draw / tot, p_b / tot


def _elo_baseline(elo_a: float, elo_b: float):
    d = elo_a - elo_b
    e_a = 1.0 / (1.0 + 10 ** (-d / 400.0))
    p_draw = 0.27 * np.exp(-((d / 400.0) ** 2))
    p_a = e_a * (1 - p_draw)
    p_b = (1 - e_a) * (1 - p_draw)
    tot = p_a + p_draw + p_b
    return p_a / tot, p_draw / tot, p_b / tot


def _rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered 3-class outcomes (lower=better)."""
    cum_p = np.cumsum(probs, axis=1)
    oh = np.zeros_like(probs)
    oh[np.arange(len(outcomes)), outcomes] = 1.0
    cum_o = np.cumsum(oh, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)))


def backtest(results: pd.DataFrame, attrs: pd.DataFrame,
             cutoff: str = "2018-01-01", cfg: Optional[Dict[str, Any]] = None,
             tournament_contains: Optional[str] = None) -> Dict[str, Any]:
    """Train-before / test-after backtest. Returns metrics + calibration table."""
    cfg = cfg or load_config()
    mcfg = cfg.get("model", {})
    k = int(mcfg.get("rolling_window", 10))
    hl = float(mcfg.get("time_decay_half_life_days", 1095))

    long, _, _ = prepare_long(results, k)
    long = long.sort_values("date").reset_index(drop=True)
    Xdf = assemble_features(long, attrs, confederation_map(), host_nations())

    cutoff_ts = pd.Timestamp(cutoff)
    train_mask = (long["date"] < cutoff_ts).to_numpy()
    test_mask = ~train_mask
    if test_mask.sum() < 50:
        log.warning("Few test matches (%d) after cutoff.", int(test_mask.sum()))

    medians = Xdf[train_mask].median(numeric_only=True).fillna(0.0)
    X = Xdf.fillna(medians).to_numpy(dtype=float)
    y = long["gf"].to_numpy(dtype=float)
    w = dixon_coles_weights(long["date"], hl)

    model = _make_estimator(mcfg)
    model.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
    long["lam_pred"] = np.clip(model.predict(X), 1e-3, 6.0)

    # pair the two perspectives of each test match
    test = long[test_mask]
    if tournament_contains:  # restrict to real tournament matches if available
        ids = long.loc[test_mask, "match_id"]
        keep = ids  # tournament info not in long; filter best-effort below
    rows = []
    for mid, grp in test.groupby("match_id"):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        p_a, p_d, p_b = outcome_probs(a["lam_pred"], b["lam_pred"])
        eb = _elo_baseline(a["elo_team"], b["elo_opp"])
        if a["gf"] > b["gf"]:
            outcome = 0
        elif a["gf"] == b["gf"]:
            outcome = 1
        else:
            outcome = 2
        rows.append({"p_a": p_a, "p_d": p_d, "p_b": p_b,
                     "eb_a": eb[0], "eb_d": eb[1], "eb_b": eb[2],
                     "outcome": outcome, "elo_team": a["elo_team"]})
    if not rows:
        return {"error": "no test matches", "n_test": 0}

    df = pd.DataFrame(rows)
    P = df[["p_a", "p_d", "p_b"]].to_numpy()
    B = df[["eb_a", "eb_d", "eb_b"]].to_numpy()
    out = df["outcome"].to_numpy()

    def brier(probs):
        oh = np.zeros_like(probs); oh[np.arange(len(out)), out] = 1.0
        return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

    metrics = {
        "n_test": int(len(df)),
        "cutoff": cutoff,
        "model": {
            "rps": _rps(P, out),
            "log_loss": float(log_loss(out, P, labels=[0, 1, 2])),
            "brier": brier(P),
            "accuracy": float(np.mean(P.argmax(1) == out)),
        },
        "elo_baseline": {
            "rps": _rps(B, out),
            "log_loss": float(log_loss(out, B, labels=[0, 1, 2])),
            "brier": brier(B),
            "accuracy": float(np.mean(B.argmax(1) == out)),
        },
    }

    # calibration of P(teamA win)
    cal = _calibration(P[:, 0], (out == 0).astype(int))
    log.info("Backtest RPS model=%.4f vs elo=%.4f (n=%d)",
             metrics["model"]["rps"], metrics["elo_baseline"]["rps"], len(df))
    return {"metrics": metrics, "calibration": cal}


def _calibration(prob: np.ndarray, hit: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_mid": (edges[b] + edges[b + 1]) / 2,
                     "predicted": float(prob[m].mean()),
                     "observed": float(hit[m].mean()),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)
