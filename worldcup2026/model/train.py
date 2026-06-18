"""Train the Groll-style random-forest goals model.

Target: goals scored by a team in a match. A single ``RandomForestRegressor``
predicts (lambda_team) from team/opponent covariates; applied twice per fixture
it yields (lambda_home, lambda_away) for the Poisson simulation layer.

Validation uses ``TimeSeriesSplit`` only (never a random split), per project
conventions. The fitted bundle (model + feature context + metrics) is pickled.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import TimeSeriesSplit

from .. import persistence as pstore
from ..config import get_path, load_config
from ..features.engineering import FeatureContext, build_training_matrix, match_feature_row
from ..tournament import all_teams, host_nations

log = logging.getLogger("worldcup2026.model")

BUNDLE_PATH = "rf_goals_model.pkl"


@dataclass
class ModelBundle:
    model: Any
    ctx: FeatureContext
    feature_names: List[str]
    metrics: Dict[str, Any]
    importances: Dict[str, float]


def _make_estimator(mcfg: Dict[str, Any]):
    common = dict(
        n_estimators=int(mcfg.get("n_estimators", 500)),
        max_depth=mcfg.get("max_depth"),
        min_samples_leaf=int(mcfg.get("min_samples_leaf", 5)),
        random_state=int(mcfg.get("random_state", 42)),
        n_jobs=-1,
    )
    if mcfg.get("algorithm") == "extra_trees":
        return ExtraTreesRegressor(**common)
    return RandomForestRegressor(**common)


def train_model(results: pd.DataFrame, attrs: pd.DataFrame,
                cfg: Dict[str, Any] | None = None, persist: bool = True) -> ModelBundle:
    """Fit the RF goals model and return a :class:`ModelBundle`."""
    cfg = cfg or load_config()
    mcfg = cfg.get("model", {})

    X, y, w, names, ctx, dates = build_training_matrix(
        results, attrs,
        rolling_window=int(mcfg.get("rolling_window", 10)),
        half_life_days=float(mcfg.get("time_decay_half_life_days", 1095)),
    )
    log.info("Training matrix: %d rows x %d features.", X.shape[0], X.shape[1])

    # --- TimeSeriesSplit CV (goals MAE + Poisson deviance) ---
    tscv = TimeSeriesSplit(n_splits=5)
    maes, devs = [], []
    for tr, te in tscv.split(X):
        est = _make_estimator(mcfg)
        est.fit(X[tr], y[tr], sample_weight=w[tr])
        pred = np.clip(est.predict(X[te]), 1e-3, None)
        maes.append(mean_absolute_error(y[te], pred))
        try:
            devs.append(mean_poisson_deviance(y[te], pred))
        except ValueError:
            devs.append(np.nan)

    # --- fit final model on all data ---
    model = _make_estimator(mcfg)
    model.fit(X, y, sample_weight=w)

    importances = dict(sorted(
        zip(names, model.feature_importances_), key=lambda kv: kv[1], reverse=True))
    metrics = {
        "cv_goals_mae": float(np.mean(maes)),
        "cv_goals_mae_std": float(np.std(maes)),
        "cv_poisson_deviance": float(np.nanmean(devs)),
        "n_train_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_date_min": str(dates.min().date()),
        "train_date_max": str(dates.max().date()),
    }
    log.info("CV goals MAE=%.3f  Poisson deviance=%.3f",
             metrics["cv_goals_mae"], metrics["cv_poisson_deviance"])

    bundle = ModelBundle(model, ctx, names, metrics, importances)
    if persist:
        save_bundle(bundle)
    return bundle


def save_bundle(bundle: ModelBundle) -> None:
    path = get_path("models") / BUNDLE_PATH
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    pstore._write_meta("rf_goals_model", "models", bundle.metrics)  # stamp last_updated
    log.info("Saved model bundle -> %s", path)


def load_bundle() -> ModelBundle | None:
    path = get_path("models") / BUNDLE_PATH
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_goals(bundle: ModelBundle, team: str, opp: str, is_home: int) -> float:
    row = match_feature_row(team, opp, is_home, bundle.ctx).reshape(1, -1)
    return float(np.clip(bundle.model.predict(row)[0], 1e-3, 6.0))


def build_lambda_table(bundle: ModelBundle,
                       teams: List[str] | None = None) -> Tuple[List[str], np.ndarray]:
    """Precompute expected goals for every ordered pair (fast Monte-Carlo).

    ``lam[i, j]`` = expected goals team ``i`` scores vs team ``j``. Host nations
    carry home advantage (is_home=1) throughout the tournament.
    """
    teams = teams or all_teams()
    hosts = set(host_nations())
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    rows, meta = [], []
    for t in teams:
        for o in teams:
            if t == o:
                continue
            rows.append(match_feature_row(t, o, int(t in hosts), bundle.ctx))
            meta.append((idx[t], idx[o]))
    preds = np.clip(bundle.model.predict(np.array(rows)), 1e-3, 6.0)

    lam = np.zeros((n, n))
    for (i, j), p in zip(meta, preds):
        lam[i, j] = p
    return teams, lam
