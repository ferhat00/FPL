"""Actual group-stage accounting: standings, third-place ranking, R32 qualifiers."""
import pandas as pd

from worldcup2026 import standings as gs
from worldcup2026.tournament import group_fixtures, groups as groups_fn


def _live(strength, partial=False):
    """Build a full WC-2026 live table where the stronger team always wins 1-0.

    If ``partial`` is True, the last fixture of each group is left unplayed (NaN).
    """
    groups = groups_fn()
    fx = group_fixtures()
    last_per_group = fx.groupby("group").tail(1).index if partial else []
    rows = []
    for i, r in enumerate(fx.itertuples()):
        if i in set(last_per_group):
            hs, as_ = (float("nan"), float("nan"))
        else:
            hs, as_ = (1, 0) if strength[r.home] > strength[r.away] else (0, 1)
        rows.append({"date": pd.Timestamp("2026-06-12"),
                     "home_team": r.home, "away_team": r.away,
                     "home_score": hs, "away_score": as_,
                     "tournament": "FIFA World Cup 2026", "neutral": True})
    return pd.DataFrame(rows)


def _strength():
    teams = [t for ms in groups_fn().values() for t in ms]
    return {t: float(1500 + i) for i, t in enumerate(teams)}  # all distinct


def test_results_by_group_covers_all_72_intra_group_fixtures():
    strength = _strength()
    rbg = gs.results_by_group(_live(strength))
    assert len(rbg) == 72
    assert rbg["played"].all()
    assert set(rbg["group"]) == set(groups_fn())


def test_group_standings_points_and_order():
    strength = _strength()
    standings = gs.group_standings(_live(strength), strength=strength)
    assert set(standings) == set(groups_fn())
    for g, df in standings.items():
        assert len(df) == 4
        # stronger-team-wins-1-0 -> points are exactly 9/6/3/0 top to bottom
        assert df["Pts"].tolist() == [9, 6, 3, 0]
        assert (df["Pld"] == 3).all()
        assert df["GD"].tolist() == [3, 1, -1, -3]
        # ranked by strength within group (since all points distinct here)
        ordered_by_strength = sorted(df["team"], key=lambda t: strength[t], reverse=True)
        assert df["team"].tolist() == ordered_by_strength


def test_third_place_ranking_flags_best_eight():
    strength = _strength()
    standings = gs.group_standings(_live(strength), strength=strength)
    thirds = gs.third_place_ranking(standings, strength)
    assert len(thirds) == 12
    assert int(thirds["qualified"].sum()) == 8
    # ranking is monotone non-increasing on the (Pts, GD, GF) key
    keys = list(zip(thirds["Pts"], thirds["GD"], thirds["GF"]))
    assert keys == sorted(keys, reverse=True)
    # all 12 thirds came from distinct groups
    assert thirds["group"].nunique() == 12


def test_r32_qualifiers_are_32_unique_with_correct_routes():
    strength = _strength()
    standings = gs.group_standings(_live(strength), strength=strength)
    thirds = gs.third_place_ranking(standings, strength)
    quals = gs.r32_qualifiers(standings, thirds)
    assert len(quals) == 32
    assert quals["team"].nunique() == 32
    counts = quals["route"].value_counts()
    assert counts["Group winner"] == 12
    assert counts["Runner-up"] == 12
    assert counts["3rd place"] == 8


def test_partial_results_do_not_error_and_count_only_played():
    strength = _strength()
    standings = gs.group_standings(_live(strength, partial=True), strength=strength)
    # one game per group unplayed -> the two teams in it have played 2, others 3
    for df in standings.values():
        assert df["Pld"].tolist().count(3) == 2
        assert df["Pld"].tolist().count(2) == 2
    thirds = gs.third_place_ranking(standings, strength)
    assert len(thirds) == 12
    assert int(thirds["qualified"].sum()) == 8
