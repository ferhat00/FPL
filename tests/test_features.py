"""Feature engineering: long-format shape, leak-free form, matrix dimensions."""
import numpy as np
import pytest

from worldcup2026.features import engineering as fe


def test_prepare_long_two_rows_per_match(small_results):
    n_played = small_results.dropna(subset=["home_score", "away_score"]).shape[0]
    long, current_elo, latest = fe.prepare_long(small_results, rolling_window=6)
    assert len(long) == 2 * n_played
    assert {"team", "opp", "gf", "ga", "elo_team", "elo_opp",
            "form_gf", "opp_form_gf"}.issubset(long.columns)
    assert len(current_elo) > 0


def test_form_is_leak_free(small_results):
    long, _, _ = fe.prepare_long(small_results, rolling_window=6)
    # shifted form -> exactly one NaN (the first match) per team that played
    n_nan = long["form_gf"].isna().sum()
    assert n_nan == long["team"].nunique()


def test_dixon_coles_weights_recent_heavier(small_results):
    w = fe.dixon_coles_weights(small_results["date"], half_life_days=365)
    s = small_results.reset_index(drop=True)
    assert w[s["date"].idxmax()] == pytest.approx(1.0, abs=1e-9)
    assert w.min() < w.max()


def test_build_training_matrix_dimensions(small_results, attrs):
    X, y, w, names, ctx, dates = fe.build_training_matrix(
        small_results, attrs, rolling_window=6, half_life_days=1095)
    assert X.shape[1] == len(fe.FEATURE_NAMES) == len(names)
    assert X.shape[0] == len(y) == len(w) == len(dates)
    assert not np.isnan(X).any()           # imputed
    assert dates.is_monotonic_increasing   # sorted by date


def test_match_feature_row_matches_width(small_results, attrs):
    _, _, _, names, ctx, _ = fe.build_training_matrix(
        small_results, attrs, rolling_window=6)
    row = fe.match_feature_row("Spain", "Panama", is_home=0, ctx=ctx)
    assert row.shape == (len(names),)
    assert not np.isnan(row).any()
